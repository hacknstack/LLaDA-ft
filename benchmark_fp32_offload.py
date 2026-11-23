"""Probe full-parameter BF16 compute + CPU FP32 master weights/AdamW states.

Uses stock torch.optim.AdamW, not a custom update rule. Writes only measurements.
"""
from benchmark_llada import ROOT
import importlib
import json
import math
import resource
import statistics
import time
import traceback
import argparse


def main(accumulation=4, fp32_gradients=False):
    import numpy as np
    import psutil
    import torch
    from transformers import AutoModel
    torch.set_num_threads(12)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=True
    suffix='_fp32grad' if fp32_gradients else ''
    result=dict(name=f'full_fp32_master_cpu_adamw_4096_acc{accumulation}{suffix}',status='starting',
        sequence_length=4096,microbatch=1,accumulation=accumulation,warmup_updates=1,timed_updates=2,
        weights='GPU BF16 plus CPU FP32 master copy',gradients=f'GPU {"FP32" if fp32_gradients else "BF16"} accumulation across {accumulation} examples, then CPU FP32',
        optimizer_states='CPU FP32',optimizer='stock torch.optim.AdamW, foreach=False',
        learning_rate=1e-5,weight_decay=0.1,betas=[0.9,0.999],epsilon=1e-8,
        beta_epsilon_provenance='Explicit implementation assumptions; paper does not specify these',
        scheduler='constant only for timing probe',gradient_checkpointing='whole_layer, non-reentrant',
        cpu_threads=12,gpu=torch.cuda.get_device_name(0),initial_available_ram_gib=psutil.virtual_memory().available/2**30)
    start=time.perf_counter()
    try:
        meta=json.loads((ROOT/'datasets/manifest.json').read_text())
        model=AutoModel.from_pretrained(meta['model_path'],trust_remote_code=True,torch_dtype=torch.bfloat16,
            device_map={'':0},low_cpu_mem_usage=True)
        model.config.use_cache=False
        mod=importlib.import_module(model.model.__class__.__module__)
        model.model.set_activation_checkpointing(mod.ActivationCheckpointingStrategy.whole_layer)
        model.train()
        params=[p for p in model.parameters() if p.requires_grad]
        gradient_buffers={}
        handles=[]
        if fp32_gradients:
            def accumulate_gradient(parameter):
                key=id(parameter)
                with torch.no_grad():
                    if key not in gradient_buffers:
                        gradient_buffers[key]=torch.zeros_like(parameter,dtype=torch.float32)
                    gradient_buffers[key].add_(parameter.grad)
                    parameter.grad=None
            handles=[p.register_post_accumulate_grad_hook(accumulate_gradient) for p in params]
        result['trainable_parameters']=sum(p.numel() for p in params)
        # Keep an independent FP32 copy so small updates survive BF16 rounding.
        masters=[torch.nn.Parameter(p.detach().to(device='cpu',dtype=torch.float32),requires_grad=True) for p in params]
        optimizer=torch.optim.AdamW(masters,lr=1e-5,betas=(0.9,0.999),eps=1e-8,
            weight_decay=0.1,foreach=False,fused=False)
        raw=np.load(ROOT/'datasets/benchmark_tokens.npy')
        ids_all=torch.tensor(raw[:len(raw)//4096*4096].reshape(-1,4096).copy(),dtype=torch.long,device='cuda')
        torch.cuda.synchronize()
        result['load_setup_seconds']=time.perf_counter()-start
        records=[]
        torch.cuda.reset_peak_memory_stats()
        for update in range(3):
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            for buffer in gradient_buffers.values():
                buffer.zero_()
            torch.cuda.synchronize();t0=time.perf_counter()
            losses=[]
            for micro in range(accumulation):
                i=update*accumulation+micro
                torch.manual_seed(1000+i)
                ids=ids_all[i:i+1]
                prob=(0.999*torch.rand((1,1),device='cuda')+0.001).expand_as(ids)
                mask=torch.rand(ids.shape,device='cuda')<prob
                noisy=torch.where(mask,126336,ids)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(input_ids=noisy,use_cache=False).logits
                    loss=(torch.nn.functional.cross_entropy(logits[mask].float(),ids[mask],reduction='none')/prob[mask]).sum()/ids.numel()
                assert torch.isfinite(loss)
                (loss/accumulation).backward()
                losses.append(float(loss.detach()))
                del logits,loss
            torch.cuda.synchronize();forward_backward=time.perf_counter()-t0
            t1=time.perf_counter()
            for p,master in zip(params,masters):
                gradient=gradient_buffers[id(p)] if fp32_gradients else p.grad
                assert gradient is not None
                master.grad=gradient.detach().to(device='cpu',dtype=torch.float32)
                p.grad=None
            grad_copy=time.perf_counter()-t1
            t2=time.perf_counter()
            norm=torch.nn.utils.clip_grad_norm_(masters,1.0,foreach=False)
            assert torch.isfinite(norm) and norm>0
            # Inspect a real master parameter to verify an FP32 update occurred.
            old=masters[0].flatten()[:8192].detach().clone()
            optimizer.step()
            changed=bool((masters[0].flatten()[:8192].detach()!=old).any())
            assert changed,'FP32 master parameter did not change'
            cpu_step=time.perf_counter()-t2
            t3=time.perf_counter()
            with torch.no_grad():
                for p,master in zip(params,masters):
                    p.copy_(master)
            torch.cuda.synchronize()
            weight_copy=time.perf_counter()-t3
            elapsed=time.perf_counter()-t0
            record=dict(update=update,seconds=elapsed,forward_backward_seconds=forward_backward,
                gradient_copy_seconds=grad_copy,cpu_optimizer_seconds=cpu_step,weight_copy_seconds=weight_copy,
                losses=losses,gradient_norm=float(norm),master_parameter_changed=changed,
                process_rss_gib=psutil.Process().memory_info().rss/2**30,
                available_ram_gib=psutil.virtual_memory().available/2**30)
            records.append(record)
            print('UPDATE',json.dumps(record),flush=True)
        timings=[r['seconds'] for r in records[1:]]
        state_dtypes=sorted({str(v.dtype) for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)})
        assert state_dtypes==['torch.float32']
        block_count=math.ceil(meta['total_tokens_with_eos']/4096)
        result.update(status='ok',updates=records,optimizer_state_dtypes=state_dtypes,
            mean_update_seconds=statistics.mean(timings),tokens_per_second=4096*accumulation/statistics.mean(timings),
            epoch_blocks=block_count,epoch_hours=block_count/accumulation*statistics.mean(timings)/3600,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            estimated_cpu_persistent_state_gib=sum(m.numel() for m in masters)*4*3/2**30)
    except Exception as error:
        result.update(status='error',error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start)
    (ROOT/'benchmarks'/ (result['name']+'.json')).write_text(json.dumps(result,indent=2))
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--accumulation',type=int,default=4)
    parser.add_argument('--fp32-gradients',action='store_true');args=parser.parse_args()
    main(accumulation=args.accumulation,fp32_gradients=args.fp32_gradients)
