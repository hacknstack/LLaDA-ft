"""Real GPU optimizer-step timing; no trained model/checkpoint is saved."""
import os
from pathlib import Path
ROOT = Path('/content/drive/MyDrive/LLaDA-ft')
os.environ['HF_HOME'] = str(ROOT/'hf-cache')
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'

import argparse
import csv
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
import time
import traceback

CONFIGS = [
    dict(name='full_adamw_bf16', kind='full', optimizer='adamw'),
    dict(name='full_adamw8bit', kind='full', optimizer='adam8'),
    dict(name='full_sgd', kind='full', optimizer='sgd'),
    dict(name='last2_adamw', kind='partial', layers=2, optimizer='adamw'),
    dict(name='last8_adamw', kind='partial', layers=8, optimizer='adamw'),
    dict(name='lora_qv_r8', kind='lora', rank=8, targets='qv', optimizer='adamw'),
    dict(name='lora_qv_r32', kind='lora', rank=32, targets='qv', optimizer='adamw'),
    dict(name='lora_all_r16', kind='lora', rank=16, targets='all', optimizer='adamw'),
    dict(name='lora_all_r64', kind='lora', rank=64, targets='all', optimizer='adamw'),
    dict(name='dora_all_r16', kind='lora', rank=16, targets='all', dora=True, optimizer='adamw'),
    dict(name='qlora_nf4_r16', kind='lora', rank=16, targets='all', quant=4, optimizer='adamw'),
    dict(name='lora_int8_r16', kind='lora', rank=16, targets='all', quant=8, optimizer='adamw'),
]


def save_result(result):
    out = ROOT/'benchmarks'
    out.mkdir(exist_ok=True)
    (out/(result['name']+'.json')).write_text(json.dumps(result, indent=2))
    print('RESULT', json.dumps(result), flush=True)


def worker(config, sequence_length=1024, stress_mask=False):
    import numpy as np
    import torch
    from transformers import AutoModel, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    meta = json.loads((ROOT/'datasets'/'manifest.json').read_text())
    result = dict(**config, sequence_length=sequence_length, microbatch=1, accumulation=1,
                  warmup_steps=2, measured_steps=8, status='starting',
                  gradient_checkpointing='whole_layer, non-reentrant', gpu=torch.cuda.get_device_name(0))
    start = time.perf_counter()
    try:
        kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={'':0}, low_cpu_mem_usage=True)
        if config.get('quant'):
            if config['quant'] == 4:
                kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_skip_modules=['model.transformer.ff_out'])
            else:
                kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True,
                    llm_int8_skip_modules=['model.transformer.ff_out'])
        model = AutoModel.from_pretrained(meta['model_path'], **kwargs)
        model.config.use_cache = False
        base = model.model
        import importlib
        mod = importlib.import_module(base.__class__.__module__)
        base.set_activation_checkpointing(mod.ActivationCheckpointingStrategy.whole_layer)
        if config['kind'] == 'partial':
            model.requires_grad_(False)
            for layer in base.transformer.blocks[-config['layers']:]:
                layer.requires_grad_(True)
            base.transformer.ln_f.requires_grad_(True)
        elif config['kind'] == 'lora':
            if config.get('quant'):
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
            targets = ['q_proj','v_proj'] if config['targets']=='qv' else ['q_proj','k_proj','v_proj','attn_out','ff_proj','up_proj','ff_out']
            # Restrict targets to transformer blocks, never the vocabulary output projection.
            names = [name for name,module in model.named_modules()
                     if '.blocks.' in name and name.split('.')[-1] in targets]
            assert names, 'No adapter targets matched'
            model = get_peft_model(model, LoraConfig(r=config['rank'], lora_alpha=2*config['rank'],
                target_modules=names, lora_dropout=0.0, bias='none', use_dora=config.get('dora',False)))
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        result['trainable_parameters'] = sum(p.numel() for p in params)
        result['total_parameters_reported'] = sum(p.numel() for p in model.parameters())
        if config['optimizer']=='adam8':
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(params, lr=1e-5)
        elif config['optimizer']=='sgd':
            optimizer = torch.optim.SGD(params, lr=1e-5, momentum=0)
        else:
            optimizer = torch.optim.AdamW(params, lr=1e-5 if config['kind']=='full' else 1e-4, fused=True)
        torch.cuda.synchronize()
        result['load_setup_seconds'] = time.perf_counter()-start
        raw = np.load(ROOT/'datasets'/'benchmark_tokens.npy')
        blocks = torch.from_numpy(raw[:len(raw)//sequence_length*sequence_length].reshape(-1,sequence_length).copy()).long().cuda()
        assert len(blocks)>=10
        timings, losses, gradients = [], [], []
        torch.cuda.reset_peak_memory_stats()
        for i in range(result['warmup_steps']+result['measured_steps']):
            # Same corpus blocks, random seed and per-example mask schedule across configurations.
            ids = blocks[i:i+1]
            torch.manual_seed(1000+i)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            p = (0.999*torch.rand((1,1),device='cuda')+0.001).expand_as(ids)
            mask = torch.rand(ids.shape,device='cuda') < p
            assert mask.any()
            noisy = torch.where(mask,126336,ids)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits = model(input_ids=noisy,use_cache=False).logits
                loss = (torch.nn.functional.cross_entropy(logits[mask].float(),ids[mask],reduction='none')/p[mask]).sum()/ids.numel()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params,1.0)
            if not torch.isfinite(norm) or norm<=0:
                raise RuntimeError(f'Invalid gradient norm {norm}')
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-t0
            if i>=result['warmup_steps']:
                timings.append(elapsed)
                losses.append(float(loss.detach()))
                gradients.append(float(norm))
            print('STEP',config['name'],i,round(elapsed,3),'loss',float(loss.detach()),flush=True)
            del logits, loss
        if stress_mask:
            # Check maximum mask density too: this creates the largest CE tensor.
            optimizer.zero_grad(set_to_none=True)
            ids = blocks[10:11]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits = model(input_ids=torch.full_like(ids,126336),use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(logits.flatten(0,1).float(),ids.flatten())
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params,1.0)
            assert torch.isfinite(norm) and norm>0
            optimizer.step()
            torch.cuda.synchronize()
            result['fully_masked_stress_step'] = dict(status='ok',loss=float(loss.detach()),gradient_norm=float(norm))
            print('FULLY_MASKED_STRESS_OK',flush=True)
            del logits,loss
        epoch_blocks = math.ceil(meta['total_tokens_with_eos']/sequence_length)
        state_dtypes = sorted({str(v.dtype) for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)})
        result.update(status='ok', step_seconds=timings, mean_step_seconds=statistics.mean(timings),
            median_step_seconds=statistics.median(timings), minimum_step_seconds=min(timings), maximum_step_seconds=max(timings),
            std_step_seconds=statistics.stdev(timings), tokens_per_second=sequence_length/statistics.mean(timings),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, losses=losses, gradient_norms=gradients,
            optimizer_state_dtypes=state_dtypes,
            epoch_blocks=epoch_blocks,
            epoch_hours=epoch_blocks*statistics.mean(timings)/3600,
            epoch_hours_observed_fast=epoch_blocks*min(timings)/3600,
            epoch_hours_observed_slow=epoch_blocks*max(timings)/3600)
    except Exception as error:
        result.update(status='oom' if 'out of memory' in str(error).lower() else 'error',
                      error=str(error), traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter()-start)
    save_result(result)


def summarize():
    out = ROOT/'benchmarks'
    results = [json.loads(p.read_text()) for p in out.glob('*.json') if p.stem in {c['name'] for c in CONFIGS}]
    results.sort(key=lambda r:r.get('epoch_hours',float('inf')))
    columns = ['name','status','trainable_parameters','mean_step_seconds','tokens_per_second','peak_allocated_gib','epoch_hours','epoch_hours_observed_fast','epoch_hours_observed_slow','load_setup_seconds']
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(results)
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    meta=json.loads((ROOT/'datasets'/'manifest.json').read_text())
    lines=['# LLaDA BBC training timing benchmark','',
        f"Selected {meta['selected_articles']:,} of {meta['eligible_articles']:,} date-eligible articles; {meta['total_tokens_with_eos']:,} tokens including article EOS; {meta['blocks_1024']:,} packed blocks.",
        '','All runs: A100 80 GiB, 1024 tokens, batch 1, accumulation 1, BF16 compute, whole-layer activation checkpointing, 2 warmups + 8 measured complete optimizer steps. Same article content and diffusion objective. No training checkpoint saved.',
        '', '| Configuration | Status | Trainable params | Tokens/s | Peak allocated GiB | Hours / epoch | Observed step range, hours/epoch |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        if r['status']=='ok':
            lines.append(f"| {r['name']} | ok | {r['trainable_parameters']:,} | {r['tokens_per_second']:.0f} | {r['peak_allocated_gib']:.1f} | {r['epoch_hours']:.2f} | {r['epoch_hours_observed_fast']:.2f}–{r['epoch_hours_observed_slow']:.2f} |")
        else:
            lines.append(f"| {r['name']} | {r['status']} | | | | | |")
    lines += ['', 'Estimates cover training compute for ONE pass over the selected even-index half, not the full unfiltered source. Three epochs approximately triple training time. Loading, downloads, preprocessing, evaluation, checkpoint writes, and interruptions are excluded. Observed ranges are min/max of 8 steps, not statistical confidence intervals. Allow 15–30% operational headroom before adding checkpoint/evaluation time. Longer sequence lengths or different batches need fresh measurements.',
        '', 'Partial tuning trains only the final 2 or 8 blocks plus final norm; frozen embeddings and output head. LoRA/DoRA train adapters in the specified transformer-block projections. Full SGD uses no momentum. Full AdamW stores moments in BF16; this is a timing configuration, not equivalent to full FP32-master-weight AdamW. Full FP32-master-weight AdamW typically exceeds this GPU without offload/sharding. Quantized bases use BF16 compute and floating-point adapters.',
        '', 'Timing probes measure feasibility and speed; they do not rank model quality or determine learning rates. All configurations start from the same pinned base model in separate processes.',
        '', 'Sources: https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md ; https://huggingface.co/datasets/RealTimeData/bbc_news_alltime']
    (out/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--summarize',action='store_true');args=parser.parse_args()
    (ROOT/'benchmarks').mkdir(exist_ok=True)
    if args.worker:
        worker(next(c for c in CONFIGS if c['name']==args.worker));return
    if args.summarize:
        summarize();return
    versions={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','bitsandbytes','datasets','accelerate']}
    (ROOT/'benchmarks'/'environment.json').write_text(json.dumps(dict(python=sys.version,packages=versions,configs=CONFIGS),indent=2))
    for config in CONFIGS:
        result_path=ROOT/'benchmarks'/(config['name']+'.json')
        if result_path.exists() and json.loads(result_path.read_text())['status']=='ok':
            continue
        log=ROOT/'logs'/(config['name']+'.log')
        print('START',config['name'],flush=True)
        with log.open('w') as f:
            try:
                proc=subprocess.run([sys.executable,__file__,'--worker',config['name']],stdout=f,stderr=subprocess.STDOUT,timeout=1200)
                if not result_path.exists():
                    save_result(dict(**config,status='process_failed',returncode=proc.returncode))
            except subprocess.TimeoutExpired:
                save_result(dict(**config,status='timeout'))
        print('FINISHED',config['name'],flush=True)
        summarize()
    print('BENCHMARK_COMPLETE',flush=True)


if __name__=='__main__':
    main()
