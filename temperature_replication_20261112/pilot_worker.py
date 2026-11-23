"""Persistent single-MIG benchmark worker. Tasks are bounded by the controller."""
import argparse
import json
import math
import os
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/workspace/temperature-study')


def atomic(path, value):
    temp = path.with_suffix('.new')
    temp.write_text(json.dumps(value, allow_nan=False))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = json.loads((ROOT/'mig_uuids.json').read_text())[gpu]
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from transformers import AutoModel
    import probabilistic_extraction as pe
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = json.loads((ROOT/'pilot_inputs.json').read_text())
    model = AutoModel.from_pretrained(ROOT/'models/llada-base', trust_remote_code=True,
                                      local_files_only=True, torch_dtype=torch.bfloat16,
                                      low_cpu_mem_usage=True, device_map={'': 0}).eval()
    model.config.use_cache = False
    folder = ROOT/'worker_io'/str(gpu)
    folder.mkdir(parents=True, exist_ok=True)

    class SliceProxy:
        def __init__(self, positions, values):
            self.positions = positions
            self.values = values

        def __getitem__(self, index):
            batch, positions, vocab = index
            assert batch == 0 and vocab == slice(None)
            assert positions.tolist() == self.positions
            return self.values.to('cuda:0')

    class NativeCache(torch.nn.Module):
        """Reuse identical singleton forwards across temperatures/decoders.

        Only native BF16 logits at still-masked positions are cached. Probability
        routines and model forward shapes remain unchanged.
        """
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.cache = OrderedDict()
            self.forward_count = self.hits = self.bytes = 0
            self.limit = 1800 * 1024**2

        def forward(self, ids, attention_mask=None):
            assert ids.shape == (1, 100) and attention_mask is None
            key = ids.detach().cpu().numpy().tobytes()
            if key in self.cache:
                self.hits += 1
                positions, values = self.cache[key]
                self.cache.move_to_end(key)
                return SimpleNamespace(logits=SliceProxy(positions, values))
            output = self.base(ids, attention_mask=attention_mask)
            self.forward_count += 1
            positions = torch.nonzero(ids[0] == 126336, as_tuple=False).squeeze(-1)
            values = output.logits[0, positions, :].detach().to('cpu', copy=True)
            size = values.numel() * values.element_size()
            while self.cache and self.bytes + size > self.limit:
                _, (_, previous) = self.cache.popitem(last=False)
                self.bytes -= previous.numel() * previous.element_size()
            if size <= self.limit:
                self.cache[key] = (positions.tolist(), values)
                self.bytes += size
            return output

    @torch.inference_mode()
    def calculate(task, active_model):
        if (ROOT/'results/runtime_study_excluded.json').exists() and (task.get('purpose')=='runtime' or task.get('timing')):
            raise ValueError('The paper runtime study is excluded on this pod; A100 measurements are required.')
        torch.set_num_threads(task.get('threads', 3))
        sequence = torch.tensor([data['tokens'][task['document']+':'+str(task['window_number'])]], device='cuda:0')
        masked = data['masks'][task['mask_id']-1]
        if task.get('reduced'):
            masked = masked[::5]
        mode = task['method']
        scheme = task['scheme']
        common = dict(model=active_model, sequence_tokens=sequence, masked_indexes=masked,
                      steps=len(masked), attention_mask=None, mask_id=126336,
                      temperature=task['temperature'])
        options = dict(num_samples=task.get('samples', 10), seed=task.get('seed', 42))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if mode == 'exact':
            fn = (pe._exact_threshold_probability_dp_from_partially_masked if scheme == 'threshold'
                  else pe._exact_low_confidence_probability_dp_from_partially_masked)
            extra = {'confidence_threshold': .9} if scheme == 'threshold' else {}
            result = fn(**common, **extra)
        elif mode == 'path_sampling':
            fn = (pe._path_sampling_threshold_probability_fast_from_partially_masked if scheme == 'threshold'
                  else pe._path_sampling_low_confidence_probability_fast_from_partially_masked)
            extra = {'confidence_threshold': .9} if scheme == 'threshold' else {}
            result = fn(**common, **options, **extra, return_sample_times=task.get('timing', False))
        elif mode == 'monte-carlo':
            fn = (pe._monte_carlo_threshold_probability_fast_from_partially_masked if scheme == 'threshold'
                  else pe._monte_carlo_probability_temperature_fast_from_partially_masked)
            extra = {'confidence_threshold': .9} if scheme == 'threshold' else {}
            result = fn(**common, **options, **extra, decoding_scheme='full', k=1,
                        scale=task.get('scale', False), return_sample_times=task.get('timing', False))
            result = dict(probability=result.estimate, hits=result.hits)
        elif mode == 'random':
            result = pe._path_sampling_random_probability_from_partially_masked(
                **common, **options, decoding_scheme='full', k=1,
                batch_size=task.get('batch_size', 1), use_selected_logits=task.get('selected', False),
                max_path_samples=None, max_path_steps=None, stratified_paths=False)
        elif mode == 'greedy':
            result = pe._duel_low_confidence_probability_fast_from_partially_masked(
                **common, max_steps=None)
        else:
            raise ValueError(mode)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        probability = result.get('probability', result.get('estimate'))
        assert math.isfinite(probability) and 0 <= probability <= 1
        output = dict(task, gpu=gpu, seconds=elapsed, probability=probability,
                      peak_gib=torch.cuda.max_memory_allocated()/2**30,
                      reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        for name in ['log_probability', 'model_forward_calls', 'model_forward_rows',
                     'num_evaluated_states', 'num_skipped_unreachable_states', 'hits',
                     'evaluated_samples', 'evaluated_steps']:
            value = result.get(name)
            if value is not None:
                output[name] = value if not isinstance(value, float) or math.isfinite(value) else str(value)
        return output

    prop = torch.cuda.get_device_properties(0)
    atomic(folder/'ready.json', dict(gpu=gpu, name=prop.name, vram=prop.total_memory,
                                   pid=os.getpid(), torch=torch.__version__))
    print('READY', gpu, flush=True)
    while True:
        job_path = folder/'job.json'
        if not job_path.exists():
            time.sleep(.05)
            continue
        job = json.loads(job_path.read_text())
        job_path.unlink()
        if job.get('stop'):
            break
        try:
            if job.get('shared_cache'):
                cached = NativeCache(model)
            else:
                cached = model
            results = []
            for task in job['tasks']:
                output = calculate(task, cached)
                results.append(output)
                with (ROOT/'results/pilot_rows.jsonl').open('a') as handle:
                    handle.write(json.dumps(output)+'\n')
                print('DONE', task['id'], round(output['seconds'], 3), flush=True)
            if job.get('shared_cache'):
                summary = dict(native_forwards=cached.forward_count, cache_hits=cached.hits,
                               cache_gib=cached.bytes/2**30)
                del cached
            else:
                summary = {}
            atomic(folder/'result.json', dict(job_id=job['id'], rows=results, cache=summary))
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            torch.cuda.empty_cache()
            atomic(folder/'result.json', dict(job_id=job['id'], error=error))


if __name__ == '__main__':
    main()
