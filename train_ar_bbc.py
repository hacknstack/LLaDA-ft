"""Matched AR continuation, text diagnostics and one optional final model save."""
import os
os.environ.update(USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import psutil
import torch
from ar_training_utils import CpuMasterAdamW, ar_loss, learning_rate, validate_checkpoint, file_sha
from drive_checkpoint_upload import DriveCheckpointStore


def diagnostics(model, streams, config, epoch):
    records=[]
    model.eval()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            for group,tokens in streams.items():
                full=len(tokens)//config['sequence_length']
                positions=np.linspace(0,full-1,min(full,config['diagnostic_blocks_per_group']),dtype=int)
                for block in positions:
                    ids=torch.as_tensor(tokens[block*4096:(block+1)*4096].astype(np.int64),device='cuda')[None,:]
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(input_ids=ids,use_cache=False).logits
                        loss=ar_loss(logits,ids)
                    assert torch.isfinite(loss),'Nonfinite diagnostic loss'
                    records.append(dict(group=group,block=int(block),loss=float(loss)))
                    del ids,logits,loss
    finally:
        model.train()
    return dict(epoch=epoch,metric='deterministic native-token next-token CE; not a memorization measure',
                means={g:float(np.mean([r['loss'] for r in records if r['group']==g])) for g in streams},records=records)


def export_final(model,tokenizer,config,run_dir,environment,store,journal,emit):
    """Only large durable output: one final model, after releasing optimizer state."""
    payload=config['expected_parameters']*2
    quota=store.service.about().get(fields='storageQuota').execute(num_retries=5)['storageQuota']
    if quota.get('limit'):
        assert int(quota['limit'])-int(quota.get('usage',0))>=payload+2*2**30,'Insufficient final-save Drive quota'
    assert shutil.disk_usage('/content').free>=payload+2*2**30,'Insufficient local staging space'
    destination=Path('/content/ar-final-staging')/config['run_name']/'final'
    destination.mkdir(parents=True,exist_ok=False)
    emit('final_model_serialization_started')
    dtype_codes={'torch.bfloat16':'BF16','torch.float32':'F32','torch.int64':'I64','torch.bool':'BOOL'}
    expected={key:(list(value.shape),dtype_codes[str(value.dtype)]) for key,value in model.state_dict().items()}
    model.save_pretrained(destination,safe_serialization=True,max_shard_size='2GB')
    tokenizer.save_pretrained(destination)
    shutil.copy2(run_dir/'config.json',destination/'training_config.json')
    shutil.copy2(run_dir/'METHODS.md',destination/'METHODS.md')
    complete=validate_checkpoint(destination,expected)
    complete.update(run=config['run_name'],epoch=config['epochs'],update=config['expected_total_updates'],
                    model_id=config['model_id'],model_revision=config['model_revision'],
                    optimizer_saved=False,exact_training_resume_supported=False,
                    config_sha256=environment['config_sha256'],methods_sha256=environment['methods_sha256'])
    (destination/'COMPLETE.json').write_text(json.dumps(complete,indent=2))
    folder=store.ensure_path(['LLaDA-ft','ar-models',config['run_name'],'final'])
    previous={}
    def progress(name,done,total):
        if done-previous.get(name,0)>=256*2**20 or done==total:
            previous[name]=done
            emit('final_model_upload_progress',file=name,bytes=done,total=total)
    receipt=store.upload_directory(destination,folder,progress)
    receipt.update(epoch=config['epochs'],update=config['expected_total_updates'],run=config['run_name'])
    path=run_dir/'final_upload_receipt.json'
    path.write_text(json.dumps(receipt,indent=2))
    store.upload(path,journal)
    emit('final_model_verified',url=receipt['folder_url'],weight_bytes=complete['total_weight_bytes'])
    assert destination.resolve().is_relative_to(Path('/content/ar-final-staging').resolve())
    shutil.rmtree(destination)
    return receipt['folder_url']


def run(config_path):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    config=json.loads(Path(config_path).read_text())
    name=config['run_name']
    assert Path(name).name==name and name not in ('.','..')
    assert config['epochs']==3 and config['sequence_length']==4096 and config['accumulation']==16
    assert config['checkpoint_epochs']==[] and config['checkpoint_every_updates']==0
    assert not config['verify_full_checkpoint_before_training'] and not config['save_optimizer_state']
    root=Path('/content/drive/MyDrive/LLaDA-ft')
    local=Path('/content/ar-inputs')/config['model_key']
    run_dir=root/'runs'/name
    assert not run_dir.exists(),'Run already exists; refusing to overwrite/restart'
    store=DriveCheckpointStore()
    project=store.folder('root','LLaDA-ft',create=False)
    assert project
    existing_journal=store.folder(project,'run-api-records',create=False)
    if existing_journal:
        prior=store.folder(existing_journal,name,create=False)
        assert prior is None or not store.children(prior,'status_api.json'),'API journal already contains this run'
    run_dir.mkdir(parents=True)
    journal=store.ensure_path(['LLaDA-ft','run-api-records',name])
    state=dict(run=name,status='preparing',epoch=0,update=0,save_final_model=config['save_final_model'])
    started=time.perf_counter()
    def emit(event,**fields):
        record=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**fields)
        state.update(last_event=event,last_time_utc=record['time_utc'],elapsed_seconds=time.perf_counter()-started)
        with (run_dir/'events.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        (run_dir/'status.json').write_text(json.dumps(state,indent=2))
        store.write_state(journal,'status_api.json',state)
        print(json.dumps(record),flush=True)
    try:
        torch.set_num_threads(config['cpu_threads'])
        torch.manual_seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32=True
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        assert torch.cuda.get_device_properties(0).total_memory>=79*2**30
        assert psutil.virtual_memory().available>=140*2**30,'Need 140 GiB available host RAM'
        data=json.loads((local/'data_manifest.json').read_text())
        assert data['model_revision']==config['model_revision']
        assert data['train_tokens']==config['expected_tokens']
        for filename,details in data['token_files'].items():
            assert file_sha(local/filename)==details['sha256'] and (local/filename).stat().st_size==details['bytes']
        shutil.copy2(config_path,run_dir/'config.json')
        shutil.copy2('/content/AR_NOTEBOOK_METHOD.md',run_dir/'METHODS.md')
        shutil.copy2(local/'data_manifest.json',run_dir/'data_manifest.json')
        sources=['train_ar_bbc.py','prepare_ar_notebook_data.py','ar_training_utils.py','drive_checkpoint_upload.py']
        for source in sources:
            shutil.copy2(Path('/content')/source,run_dir/source)
        environment=dict(python=__import__('sys').version,
            packages={p:importlib.metadata.version(p) for p in ['torch','transformers','tokenizers','accelerate','safetensors','datasets']},
            gpu=torch.cuda.get_device_name(0),gpu_vram_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
            source_sha256={s:file_sha(Path('/content')/s) for s in sources},
            config_sha256=file_sha(config_path),methods_sha256=file_sha(run_dir/'METHODS.md'),
            model_revision=config['model_revision'],storage='text logs plus optional one final model only')
        (run_dir/'environment.json').write_text(json.dumps(environment,indent=2))
        for filename in ['config.json','METHODS.md','data_manifest.json','environment.json']+sources:
            store.upload(run_dir/filename,journal)
        model=AutoModelForCausalLM.from_pretrained(data['model_path'],torch_dtype=torch.bfloat16,
            device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa')
        model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        params=list(model.parameters())
        assert sum(p.numel() for p in params)==config['expected_parameters']
        assert all(p.requires_grad and p.dtype==torch.bfloat16 and p.is_cuda for p in params)
        tokenizer=AutoTokenizer.from_pretrained(data['model_path'])
        # Memory-map token arrays on local disk; move only one block onto the GPU.
        streams={g:np.load(local/f,mmap_mode='r') for g,f in [('train','train_tokens.npy'),('clean_odd_controls','control_tokens.npy')]}
        count=math.ceil(len(streams['train'])/4096)
        per_epoch=math.ceil(count/16)
        total=per_epoch*3
        assert (count,per_epoch,total)==(config['expected_blocks_per_epoch'],config['expected_updates_per_epoch'],config['expected_total_updates'])
        baseline=diagnostics(model,streams,config,0)
        p=run_dir/'diagnostics_epoch_000.json';p.write_text(json.dumps(baseline,indent=2));store.upload(p,journal)
        emit('baseline_complete',means=baseline['means'])
        opt=CpuMasterAdamW(params,config)
        state['status']='running'
        emit('training_started',total_updates=total,blocks_per_epoch=count)
        update=0
        for epoch in range(1,4):
            state['epoch']=epoch
            order=np.random.default_rng(config['seed']+epoch).permutation(count)
            crops=np.random.default_rng(config['seed']+10000+epoch)
            emit('epoch_started',epoch=epoch,block_order_sha256=hashlib.sha256(order.tobytes()).hexdigest())
            epoch_tokens=epoch_targets=epoch_crops=0
            for offset in range(0,count,16):
                indices=order[offset:offset+16]
                opt.begin()
                torch.cuda.synchronize();t0=time.perf_counter()
                losses=[];window_tokens=window_targets=window_crops=0
                for block in indices:
                    ids=torch.as_tensor(streams['train'][block*4096:min((block+1)*4096,len(streams['train']))].astype(np.int64),device='cuda')[None,:]
                    if crops.random()<config['random_length_probability']:
                        ids=ids[:,:int(crops.integers(1,ids.shape[1]+1))]
                        window_crops+=1
                    window_tokens+=ids.numel();window_targets+=max(0,ids.numel()-1)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(input_ids=ids,use_cache=False).logits
                        loss=ar_loss(logits,ids)
                    assert torch.isfinite(loss),'Nonfinite training loss'
                    (loss/len(indices)).backward()
                    losses.append(float(loss.detach()))
                    del ids,logits,loss
                torch.cuda.synchronize()
                fb=time.perf_counter()-t0
                update+=1
                lr=learning_rate(update,total,config)
                result=opt.step(lr)
                epoch_tokens+=window_tokens;epoch_targets+=window_targets;epoch_crops+=window_crops
                state.update(update=update,epoch_tokens=epoch_tokens,epoch_targets=epoch_targets)
                emit('optimizer_update',epoch=epoch,update=update,epoch_update=offset//16+1,
                    window_size=len(indices),mean_loss=float(np.mean(losses)),learning_rate=lr,
                    processed_tokens=window_tokens,prediction_targets=window_targets,random_crops=window_crops,
                    forward_backward_seconds=fb,update_seconds=time.perf_counter()-t0,
                    cpu_rss_gib=psutil.Process().memory_info().rss/2**30,
                    gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,**result)
            opt.optimizer.zero_grad(set_to_none=True)
            emit('epoch_training_complete',epoch=epoch,processed_tokens=epoch_tokens,prediction_targets=epoch_targets,random_crops=epoch_crops)
            diag=diagnostics(model,streams,config,epoch)
            p=run_dir/f'diagnostics_epoch_{epoch:03d}.json';p.write_text(json.dumps(diag,indent=2));store.upload(p,journal)
            emit('diagnostics_complete',epoch=epoch,means=diag['means'])
        assert update==total
        # Remove hooks and large CPU/GPU optimizer allocations before serialization.
        for handle in opt.handles:
            handle.remove()
        opt.optimizer.zero_grad(set_to_none=True)
        opt.optimizer.state.clear();opt.buffers.clear()
        del opt
        for p in params:
            p.grad=None
        gc.collect();torch.cuda.empty_cache()
        state['status']='saving_final' if config['save_final_model'] else 'complete'
        emit('optimization_complete',updates=update)
        if config['save_final_model']:
            state['final_model_url']=export_final(model,tokenizer,config,run_dir,environment,store,journal,emit)
            state['final_model_verified']=True
        state['status']='complete'
        emit('training_complete',updates=update,final_model_saved=config['save_final_model'])
    except BaseException as error:
        state.update(status='failed',error=str(error),traceback=traceback.format_exc())
        emit('training_failed',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
