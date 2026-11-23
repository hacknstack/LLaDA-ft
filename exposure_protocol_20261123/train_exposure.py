"""Explicit-launch exposure training worker; final-only model serialization."""
import argparse
import collections
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time

ROOT=Path('/workspace/bbc-exposure')


def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.new')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def main(args):
    assert 0<=args.smoke_updates<=2
    config=json.loads((ROOT/'inputs'/args.key/'training_config.json').read_text())
    allocation=json.loads((ROOT/'protocol/allocation_plan.json').read_text())['models'][args.key]
    if not args.execute and not args.smoke_updates:
        print(json.dumps(dict(model=args.key,updates=config['total_updates'],allocation=allocation,
                              status='VALIDATION_ONLY_PASS_EXECUTE_TO_TRAIN'),indent=2));return
    readiness=json.loads((ROOT/'protocol/readiness.json').read_text())
    assert args.smoke_updates or (readiness['ready_to_launch'] and args.key in readiness['models'])
    run_name='smoke_'+str(int(time.time())) if args.smoke_updates else readiness['run_name']
    run=ROOT/('smoke_runs' if args.smoke_updates else 'runs')/run_name/args.key
    run.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
    affinity={0:list(range(0,20)),1:list(range(52,72)),2:list(range(72,92))}[args.gpu]
    os.sched_setaffinity(0,affinity)
    import numpy as np
    import torch
    from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer
    from ar_training_utils import ar_loss,learning_rate,validate_checkpoint,file_sha
    from async_optimizer import AsyncCpuMasterAdamW
    config.update(cpu_threads=allocation['cpu_threads'],microbatch=allocation['microbatch'],effective_batch_sequences=16,
                  accumulation=16//allocation['microbatch'],foreach=allocation.get('foreach',False))
    assert config['random_length_probability']==0 and config['sequence_length']==4096
    torch.set_num_threads(config['cpu_threads']);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=True
    for name,h in config['input_file_sha256'].items():assert file_sha(ROOT/'inputs'/args.key/name)==h
    data_dir=ROOT/'data/frozen_exposure'
    assert file_sha(data_dir/'articles.jsonl')==config['article_manifest_sha256']
    assert file_sha(data_dir/'occurrences.jsonl')==config['content_order_manifest_sha256']
    atomic(run/'config.json',config)
    state=dict(state='preparing',model=args.key,update=0,total_updates=config['total_updates'],started_unix=time.time())
    def emit(event,**fields):
        record=dict(time_unix=time.time(),event=event,**fields)
        with (run/'events.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        state.update(last_event=event,updated_unix=record['time_unix']);atomic(run/'status.json',state)
        print(json.dumps(record),flush=True)
    if args.key=='llada8b':
        model=AutoModel.from_pretrained(config['model_path'],trust_remote_code=True,torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,device_map={'':0})
        mod=importlib.import_module(model.model.__class__.__module__)
        model.model.set_activation_checkpointing(mod.ActivationCheckpointingStrategy.whole_layer)
    else:
        model=AutoModelForCausalLM.from_pretrained(config['model_path'],torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,
                    device_map={'':0},attn_implementation='sdpa')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False;model.train()
    params=list(model.parameters())
    assert all(p.requires_grad and p.dtype==torch.bfloat16 for p in params)
    tokenizer=AutoTokenizer.from_pretrained(config['model_path'],trust_remote_code=args.key=='llada8b')
    streams={name:np.load(ROOT/'inputs'/args.key/file,mmap_mode='r') for name,file in
             [('train','train_tokens.npy'),('control','control_tokens.npy')]}
    spans=[json.loads(l) for l in (ROOT/'inputs'/args.key/'occurrence_spans.jsonl').open()]
    occurrence_ends=np.array([r['end_token'] for r in spans],dtype=np.int64)
    occurrence_targets=np.zeros(len(spans),dtype=np.int64)
    occurrence_eos_targets=np.zeros(len(spans),dtype=np.int64)
    def loss_for(ids):
        if args.key=='llada8b':
            prob=(.001+.999*torch.rand((ids.shape[0],1),device='cuda')).expand_as(ids)
            mask=torch.rand(ids.shape,device='cuda')<prob
            noisy=torch.where(mask,126336,ids)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(input_ids=noisy,use_cache=False).logits
                loss=(torch.nn.functional.cross_entropy(logits[mask].float(),ids[mask],reduction='none')/prob[mask]).sum()/ids.numel() if bool(mask.any()) else logits[...,0].float().sum()*0.
            return loss,mask.detach().cpu().numpy()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(input_ids=ids,use_cache=False).logits
            targets=np.ones(tuple(ids.shape),dtype=bool);targets[:,0]=False
            return ar_loss(logits,ids),targets
    def diagnostics(tag):
        records=[];model.eval()
        try:
            with torch.random.fork_rng(devices=[0]),torch.no_grad():
                for group,data in streams.items():
                    full=len(data)//4096
                    for j,block in enumerate(np.linspace(0,full-1,min(8,full),dtype=int)):
                        ids=torch.as_tensor(data[block*4096:(block+1)*4096].astype(np.int64),device='cuda')[None,:]
                        for rep in range(4 if args.key=='llada8b' else 1):
                            torch.manual_seed(71001+100*j+rep)
                            loss,_=loss_for(ids)
                            assert torch.isfinite(loss)
                            records.append(dict(group=group,block=int(block),draw=rep,loss=float(loss)))
                            del loss
                        del ids
        finally:model.train()
        atomic(run/('diagnostics_'+tag+'.json'),dict(tag=tag,records=records,not_an_extraction_metric=True))
    diagnostics('baseline');emit('baseline_complete')
    opt=AsyncCpuMasterAdamW(params,config)
    total_blocks=config['blocks'];processed=0
    opportunities=0
    targets_path=run/'block_target_opportunities.jsonl'
    # Store only small target-coverage counters, never optimizer/model snapshots.
    opportunities_file=targets_path.open('w',buffering=1)
    milestones={math.ceil(config['total_updates']/3),math.ceil(2*config['total_updates']/3),config['total_updates']}
    state['state']='training';emit('training_started',physical_microbatch=config['microbatch'],effective_batch=16)
    offsets=list(range(0,total_blocks,16))
    if args.smoke_updates:offsets=([0,offsets[-1]][:args.smoke_updates])
    for update,offset in enumerate(offsets,start=1):
        indices=list(range(offset,min(offset+16,total_blocks)))
        opt.begin();torch.cuda.synchronize();started=time.perf_counter()
        window_tokens=0;weighted_loss=0.
        for position in range(0,len(indices),config['microbatch']):
            chosen=indices[position:position+config['microbatch']]
            groups=[]
            for block in chosen:
                array=streams['train'][block*4096:min((block+1)*4096,len(streams['train']))].astype(np.int64)
                if not groups or len(groups[-1][0][1])!=len(array):groups.append([])
                groups[-1].append((block,array))
            for group in groups:
                ids=torch.as_tensor(np.stack([x[1] for x in group]),device='cuda')
                loss,target_masks=loss_for(ids)
                assert torch.isfinite(loss)
                (loss*len(group)/len(indices)).backward()
                window_tokens+=ids.numel();weighted_loss+=float(loss.detach())*len(group)/len(indices)
                for (block,array),target_mask in zip(group,target_masks):
                    count=int(target_mask.sum())
                    opportunities_file.write(json.dumps(dict(block=block,input_tokens=len(array),target_opportunities=int(count)))+'\n')
                    opportunities+=count
                    positions=np.flatnonzero(target_mask)+block*4096
                    owners=np.searchsorted(occurrence_ends,positions,side='right')
                    np.add.at(occurrence_targets,owners,1)
                    eos=positions==occurrence_ends[owners]-1
                    np.add.at(occurrence_eos_targets,owners[eos],1)
                del ids,loss
        torch.cuda.synchronize();fb=time.perf_counter()-started
        outcome=opt.step(learning_rate(update,config['total_updates'],config))
        processed+=window_tokens
        state.update(update=update,processed_tokens=processed)
        emit('optimizer_update',window_examples=len(indices),mean_loss=weighted_loss,forward_backward_seconds=fb,
             update_seconds=time.perf_counter()-started,**outcome)
        if update in milestones:
            diagnostics('progress_'+str(update));emit('diagnostics_complete',update=update)
        if args.smoke_updates and update>=args.smoke_updates:break
    opportunities_file.close()
    if args.smoke_updates:
        assert state['update']==args.smoke_updates
        expected_tokens=sum(min(4096,len(streams['train'])-b*4096) for offset in offsets for b in range(offset,min(offset+16,total_blocks)))
        assert int(occurrence_targets.sum())==int(opportunities) and processed==expected_tokens
        state['state']='smoke_passed_no_model_saved'
        emit('production_path_smoke_passed',tokens=processed,target_opportunities=int(opportunities))
        atomic(run/'SMOKE_COMPLETE.json',dict(passed=True,model=args.key,full_training_started=False,
              optimizer_updates=args.smoke_updates,processed_tokens=processed,block_and_article_target_accounting_verified=True,
              actual_final_tail_exercised=args.smoke_updates==2,saved_model=False))
        return
    assert processed==config['actual_train_tokens'] and state['update']==config['total_updates']
    articles=[json.loads(l) for l in (data_dir/'articles.jsonl').open()]
    consumed=collections.Counter()
    last_end=0
    for line in (ROOT/'inputs'/args.key/'occurrence_spans.jsonl').open():
        r=json.loads(line);assert r['start_token']==last_end and r['end_token']<=processed
        consumed[r['article_id']]+=1;last_end=r['end_token']
    assert last_end==processed and all(consumed[r['article_id']]==r['exposures'] for r in articles)
    article_coverage=collections.defaultdict(lambda:dict(input_tokens=0,target_opportunities=0,eos_target_opportunities=0))
    for i,r in enumerate(spans):
        c=article_coverage[r['article_id']]
        c['input_tokens']+=r['end_token']-r['start_token']
        c['target_opportunities']+=int(occurrence_targets[i]);c['eos_target_opportunities']+=int(occurrence_eos_targets[i])
    atomic(run/'exposure_audit.json',dict(all_article_counts_verified=True,consumed=dict(consumed),input_tokens=processed,
                target_opportunities=int(opportunities),controls_training_presentations=0,article_coverage=dict(article_coverage)))
    for h in opt.handles:h.remove()
    opt.optimizer.zero_grad(set_to_none=True);opt.optimizer.state.clear();opt.buffers.clear();opt.masters.clear();opt.host_grads.clear()
    del opt;gc.collect();torch.cuda.empty_cache()
    final=run/'final';final.mkdir()
    codes={'torch.bfloat16':'BF16','torch.float32':'F32','torch.int64':'I64','torch.bool':'BOOL'}
    expected={k:(list(v.shape),codes[str(v.dtype)]) for k,v in model.state_dict().items()}
    emit('final_serialization_started')
    model.save_pretrained(final,safe_serialization=True,max_shard_size='2GB');tokenizer.save_pretrained(final)
    complete=validate_checkpoint(final,expected)
    complete.update(model_key=args.key,base_revision=config['model_revision'],updates=config['total_updates'],
             exact_exposures_verified=True,optimizer_saved=False,exact_resume_supported=False)
    atomic(final/'COMPLETE.json',complete)
    params.clear();del model;gc.collect();torch.cuda.empty_cache()
    state['state']='serialized_waiting_upload'
    emit('compute_complete_gpu_released',final_directory=str(final))
    atomic(run/'COMPUTE_COMPLETE.json',dict(final_directory=str(final),weight_bytes=complete['total_weight_bytes'],
                    trained=True,exposure_counts_verified=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--key',required=True);parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--smoke-updates',type=int,default=0)
    args=parser.parse_args()
    main(args)
