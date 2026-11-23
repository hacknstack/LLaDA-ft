"""Disposable full-parameter training probes. Never save trained probe weights."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time
import traceback

ROOT=Path('/workspace/bbc-exposure')


def main(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
    # Separate physical-core ranges near the assigned GPU, before any tensor allocation.
    affinity={0:list(range(0,20)),1:list(range(52,72)),2:list(range(72,92))}[args.gpu]
    if not args.unpinned:os.sched_setaffinity(0,affinity)
    import numpy as np
    import psutil
    import torch
    from transformers import AutoModel,AutoModelForCausalLM
    from ar_training_utils import CpuMasterAdamW,ar_loss
    if args.async_offload:
        from async_optimizer import AsyncCpuMasterAdamW
        CpuMasterAdamW=AsyncCpuMasterAdamW
    config=json.loads((ROOT/'inputs'/args.key/'training_config.json').read_text())
    config['gradient_clip_norm']=1.0
    if args.async_offload and args.microbatch>=4:
        earlier=ROOT/'audits'/('probe_async_mb2_'+args.key+'.json')
        if earlier.exists():
            trial=json.loads(earlier.read_text()).get('foreach_trial',{})
            config['foreach']=trial.get('recommended',False)
    if args.foreach:config['foreach']=True
    report=dict(model_key=args.key,gpu=args.gpu,tag=args.tag,status='starting',full_production_started=False,
           cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_threads=args.threads,physical_microbatch=args.microbatch,
           effective_batch_sequences=16,async_offload=args.async_offload,foreach=config.get('foreach',False),
           worst_case_mask_stress=args.worst_mask,updates=[],cpu_sweep=[])
    out=ROOT/'audits'/('probe_'+args.tag+'_'+args.key+'.json')
    def emit(event,**fields):
        report['last_event']=event
        out.write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(event=event,model=args.key,**fields)),flush=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=True
    load_start=time.perf_counter()
    if args.key=='llada8b':
        model=AutoModel.from_pretrained(config['model_path'],trust_remote_code=True,torch_dtype=torch.bfloat16,
             low_cpu_mem_usage=True,device_map={'':0})
        mod=importlib.import_module(model.model.__class__.__module__)
        model.model.set_activation_checkpointing(mod.ActivationCheckpointingStrategy.whole_layer)
    else:
        model=AutoModelForCausalLM.from_pretrained(config['model_path'],torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,device_map={'':0},attn_implementation='sdpa')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    model.train()
    params=list(model.parameters())
    assert all(p.requires_grad and p.dtype==torch.bfloat16 and p.is_cuda for p in params)
    report['parameters']=sum(p.numel() for p in params)
    data=np.load(ROOT/'inputs'/args.key/'train_tokens.npy',mmap_mode='r')
    opt=CpuMasterAdamW(params,config)
    report['load_seconds']=time.perf_counter()-load_start
    torch.cuda.reset_peak_memory_stats()
    emit('LOADED',parameters=report['parameters'])

    def loss_for(ids):
        if args.key=='llada8b':
            prob=(torch.ones((ids.shape[0],1),device='cuda') if args.worst_mask else
                  .001+.999*torch.rand((ids.shape[0],1),device='cuda')).expand_as(ids)
            mask=torch.ones_like(ids,dtype=torch.bool) if args.worst_mask else torch.rand(ids.shape,device='cuda')<prob
            noisy=torch.where(mask,126336,ids)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(input_ids=noisy,use_cache=False).logits
                if not bool(mask.any()):loss=logits[...,0].float().sum()*0.
                else:loss=(torch.nn.functional.cross_entropy(logits[mask].float(),ids[mask],reduction='none')/prob[mask]).sum()/ids.numel()
        else:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(input_ids=ids,use_cache=False).logits
                loss=ar_loss(logits,ids)
        return loss

    windows=[list(range(i*16,(i+1)*16)) for i in range(args.updates)]
    if args.tail:
        size=config['final_remainder_window']
        windows.append(list(range(config['blocks']-size,config['blocks'])))
    for update,blocks in enumerate(windows):
        opt.begin()
        torch.cuda.synchronize();started=time.perf_counter()
        losses=[]
        lengths=[]
        for offset in range(0,len(blocks),args.microbatch):
            selected=blocks[offset:offset+args.microbatch]
            arrays=[data[b*4096:min((b+1)*4096,len(data))].astype(np.int64) for b in selected]
            # The sole short final block is processed separately, without padding.
            groups=[]
            for array in arrays:
                if not groups or len(groups[-1][0])!=len(array):groups.append([])
                groups[-1].append(array)
            for group in groups:
                ids=torch.as_tensor(np.stack(group),device='cuda')
                lengths.extend([ids.shape[1]]*ids.shape[0])
                loss=loss_for(ids)
                assert torch.isfinite(loss)
                (loss*len(group)/len(blocks)).backward()
                losses.extend([float(loss.detach())]*len(group))
                del ids,loss
        torch.cuda.synchronize()
        fb=time.perf_counter()-started
        assert len(opt.buffers)==len(params) and all(p.grad is None for p in params)
        outcome=opt.step(config['peak_learning_rate'])
        item=dict(update=update+1,window=len(blocks),lengths=lengths,mean_loss=statistics.mean(losses),
           forward_backward_seconds=fb,total_seconds=time.perf_counter()-started,**outcome)
        report['updates'].append(item)
        emit('UPDATE_PASSED',**item)
    assert all(x.dtype==torch.float32 for x in opt.buffers.values())
    assert all(p.dtype==torch.float32 for p in opt.masters)
    assert {str(v.dtype) for s in opt.optimizer.state.values() for v in s.values() if torch.is_tensor(v)}=={'torch.float32'}
    if args.sweep:
        for threads in (8,12,16):
            torch.set_num_threads(threads)
            t=time.perf_counter()
            outcome=opt.step(config['peak_learning_rate'])
            row=dict(threads=threads,total_seconds=time.perf_counter()-t,**outcome)
            report['cpu_sweep'].append(row)
            emit('CPU_THREADS_CHECKED',**row)
        torch.set_num_threads(args.threads)
    if args.async_offload and args.microbatch==2:
        proof=ROOT/'audits/foreach_optimizer_validation.json'
        if proof.exists() and json.loads(proof.read_text())['passed']:
            for g in opt.optimizer.param_groups:g['foreach']=True
            t=time.perf_counter();trial=opt.step(config['peak_learning_rate'])
            reference=statistics.mean(u['cpu_optimizer_seconds'] for u in report['updates'][1:])
            report['foreach_trial']=dict(total_seconds=time.perf_counter()-t,**trial,
                  reference_cpu_seconds=reference,recommended=trial['cpu_optimizer_seconds']<reference*.85)
            for g in opt.optimizer.param_groups:g['foreach']=False
            emit('FOREACH_CPU_CHECKED',**report['foreach_trial'])
    full=[u['total_seconds'] for u in report['updates'][1:args.updates]]
    report.update(status='passed',warm_full_update_seconds=statistics.mean(full) if full else report['updates'][0]['total_seconds'],
            projected_training_hours=config['total_updates']*(statistics.mean(full) if full else report['updates'][0]['total_seconds'])/3600,
            peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            peak_host_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            all_parameter_gradients_verified=True,fp32_masters_gradients_moments_verified=True,
            actual_tail_tested=args.tail,serialization_tested=False)
    emit('PROBE_PASSED',hours=report['projected_training_hours'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--key',required=True);parser.add_argument('--gpu',required=True,type=int)
    parser.add_argument('--tag',required=True);parser.add_argument('--updates',type=int,default=3)
    parser.add_argument('--threads',type=int,default=12);parser.add_argument('--tail',action='store_true')
    parser.add_argument('--microbatch',type=int,default=1)
    parser.add_argument('--async-offload',action='store_true')
    parser.add_argument('--foreach',action='store_true')
    parser.add_argument('--worst-mask',action='store_true')
    parser.add_argument('--sweep',action='store_true');parser.add_argument('--unpinned',action='store_true')
    args=parser.parse_args()
    try:main(args)
    except Exception as e:
        (ROOT/'audits'/('ERROR_probe_'+args.tag+'_'+args.key+'.json')).write_text(json.dumps(dict(type=type(e).__name__,traceback=traceback.format_exc()),indent=2))
        print('PROBE_FAILED',args.key,type(e).__name__,flush=True)
        raise
