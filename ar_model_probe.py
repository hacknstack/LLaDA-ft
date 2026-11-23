"""Full-parameter AR training feasibility, using the frozen LLaDA articles.

Large files stay on the Colab VM. This script never starts a production run.
"""
import os
os.environ.update(HF_HUB_CACHE='/content/ar-ft-cache/hub', HF_HUB_DISABLE_XET='1',
                  TOKENIZERS_PARALLELISM='false', USE_TF='0', USE_FLAX='0')
import argparse
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import statistics
import time
import traceback

MODELS = {'llama2_7b': 'meta-llama/Llama-2-7b-hf',
          'llama3_8b': 'meta-llama/Meta-Llama-3-8B',
          'mistral7b_v01': 'mistralai/Mistral-7B-v0.1',
          'olmo7b': 'allenai/OLMo-7B-hf'}
ROOT = Path('/content/drive/MyDrive/LLaDA-ft')
OUT = Path('/content/ar-feasibility')
OUT.mkdir(exist_ok=True)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ar_loss(logits, ids):
    import torch
    if ids.shape[1] < 2:
        return logits[..., 0].float().sum() * 0.0
    return torch.nn.functional.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                                             ids[:, 1:].reshape(-1), reduction='mean')


def main(key, length):
    import numpy as np
    import psutil
    import torch
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ar_probe_optimizer import CpuMasterAdamW
    torch.set_num_threads(12)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    start = time.perf_counter()
    report = dict(key=key, model_id=MODELS[key], sequence_length=length, status='starting',
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  production_training_started=False, cpu_threads=12, updates=[])

    def emit(event, **data):
        print(json.dumps(dict(event=event, key=key, **data)), flush=True)
        (OUT / f'{key}_{length}.json').write_text(json.dumps(report, indent=2))

    try:
        info = HfApi().model_info(MODELS[key])
        report['model_revision'] = info.sha
        config_path = hf_hub_download(MODELS[key], 'config.json', revision=info.sha)
        native = json.loads(Path(config_path).read_text())
        report['native_config'] = native
        report['context_extrapolation'] = length > native.get('max_position_embeddings', length)
        emit('ACCESS_VERIFIED', revision=info.sha, context_extrapolation=report['context_extrapolation'])
        tokenizer = AutoTokenizer.from_pretrained(MODELS[key], revision=info.sha, use_fast=True)
        selected = ROOT / 'datasets/bbc_even_after_2025_02.jsonl'
        rows = [json.loads(line) for line in selected.open()]
        assert [r['filtered_index'] for r in rows] == list(range(0, 10643, 2))
        assert all(r['published_date'] >= '2025-03-01' for r in rows)
        assert sum(not r['content'].strip() for r in rows) == 7
        texts = [r['content'] for r in rows if r['content'].strip()]
        encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False,
                            return_token_type_ids=False)['input_ids']
        assert tokenizer.eos_token_id is not None
        stream = np.asarray([token for ids in encoded for token in ids + [tokenizer.eos_token_id]], dtype=np.int32)
        token_path = OUT / f'{key}_tokens.npy'
        np.save(token_path, stream)
        count = math.ceil(len(stream) / length)
        updates_per_epoch = math.ceil(count / 16)
        report.update(selected_articles=len(rows), nonempty_articles=len(texts),
            frozen_article_file_sha256=sha(selected), train_tokens=len(stream), blocks_per_epoch=count,
            tail_tokens=len(stream) % length or length, updates_per_epoch=updates_per_epoch,
            three_epoch_updates=updates_per_epoch*3, token_file_sha256=sha(token_path),
            tokenizer_class=type(tokenizer).__name__, eos_token_id=tokenizer.eos_token_id,
            native_tokenizer=True, bos_inserted=False)
        emit('TOKENIZATION_READY', tokens=len(stream), blocks=count, three_epoch_updates=updates_per_epoch*3)
        names = [s.rfilename for s in info.siblings if '/' not in s.rfilename]
        weight_files = [name for name in names if name.endswith('.safetensors')]
        assert weight_files, 'No safetensors release; not falling back to pickle weights'
        selected_files = weight_files + [name for name in names if name.endswith(('.json', '.model', '.txt', '.tiktoken'))]
        model_path = snapshot_download(MODELS[key], revision=info.sha, allow_patterns=selected_files, max_workers=4)
        report['model_path'] = model_path
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map={'': 0}, attn_implementation='sdpa')
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        parameters = list(model.parameters())
        assert all(p.requires_grad and p.dtype == torch.bfloat16 and p.is_cuda for p in parameters)
        nparams = sum(p.numel() for p in parameters)
        report.update(trainable_parameters=nparams, model_tensor_bytes=nparams*2,
            gpu=torch.cuda.get_device_name(0), gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
            initial_available_host_gib=psutil.virtual_memory().available/2**30,
            attention_implementation=model.config._attn_implementation,
            whole_layer_nonreentrant_checkpointing=bool(model.is_gradient_checkpointing))
        assert model.is_gradient_checkpointing
        assert psutil.virtual_memory().available >= (nparams*16 + 6*2**30), 'Insufficient available CPU RAM'
        config = dict(peak_learning_rate=1e-5, betas=[.9, .999], epsilon=1e-8,
                      weight_decay=.1, gradient_clip_norm=1.)
        opt = CpuMasterAdamW(parameters, config)
        # Exactly the LLaDA CPU-master implementation; an isolated process per model.
        # Exercise two full windows and the true final accumulation-window size.
        remainder = count % 16 or 16
        window_sizes = [16, 16, remainder]
        rng = np.random.default_rng(43)
        order = rng.permutation(count)
        torch.cuda.reset_peak_memory_stats()
        report['load_and_prepare_seconds'] = time.perf_counter()-start
        emit('MODEL_LOADED', parameters=nparams, remainder_window=remainder)
        for update, size in enumerate(window_sizes):
            opt.begin()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            losses = []
            if update < 2:
                blocks = order[update*16:(update+1)*16]
            else:
                blocks = list(range(count-size, count))
            lengths = []
            for block in blocks:
                ids = torch.tensor(stream[block*length:min((block+1)*length, len(stream))].astype(np.int64), device='cuda')[None, :]
                lengths.append(ids.shape[1])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(input_ids=ids, use_cache=False).logits
                    loss = ar_loss(logits, ids)
                assert torch.isfinite(loss)
                (loss / size).backward()
                losses.append(float(loss.detach()))
                del logits, loss, ids
            torch.cuda.synchronize()
            fb = time.perf_counter()-t0
            assert all(p.grad is None for p in parameters)
            assert len(opt.buffers) == len(parameters)
            outcome = opt.step(1e-5)
            record = dict(update=update+1, window_size=size, sequence_lengths=lengths,
                mean_loss=statistics.mean(losses), seconds=time.perf_counter()-t0,
                forward_backward_seconds=fb, gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,
                cpu_rss_gib=psutil.Process().memory_info().rss/2**30, **outcome)
            report['updates'].append(record)
            emit('UPDATE_PASSED', **record)
        dtypes = sorted({str(value.dtype) for state in opt.optimizer.state.values()
                         for value in state.values() if torch.is_tensor(value)})
        assert dtypes == ['torch.float32']
        # A one-token crop has no AR target; preserve the occurrence as zero loss.
        opt.begin()
        one = torch.tensor([[int(stream[0])]], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=one, use_cache=False).logits
            zero = ar_loss(logits, one)
        assert zero.item() == 0
        zero.backward()
        assert all(torch.count_nonzero(buffer).item() == 0 for buffer in opt.buffers.values())
        del logits, zero
        report.update(status='ok', optimizer_state_dtypes=dtypes, one_token_zero_loss_verified=True,
            peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            peak_cpu_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            steady_full_update_seconds=report['updates'][1]['seconds'],
            three_epoch_compute_hours=updates_per_epoch*3*report['updates'][1]['seconds']/3600,
            timing_limit='One warm full-window timing; short-window, diagnostics, downloads and uploads not extrapolated accurately',
            checkpoint_serialization_tested=False,
            probe_schedule='Constant 1e-5 for feasibility; production retains 5% warmup and 10% linear terminal decay',
            probe_random_crops='Disabled for worst-case full-length memory measurement; zero-target length-1 path tested separately')
        emit('PROBE_PASSED', peak_gpu_gib=report['peak_gpu_allocated_gib'],
             peak_cpu_gib=report['peak_cpu_rss_gib'], projected_compute_hours=report['three_epoch_compute_hours'])
    except Exception as error:
        report.update(status='blocked' if type(error).__name__ == 'GatedRepoError' else 'error',
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        emit('PROBE_FAILED', error_type=type(error).__name__)
    finally:
        report['elapsed_seconds'] = time.perf_counter()-start
        (OUT / f'{key}_{length}.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('--length', type=int, default=4096)
    args = parser.parse_args()
    main(args.model, args.length)
