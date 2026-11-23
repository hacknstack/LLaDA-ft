"""Protocol v1: full LLaDA BBC continuation with FP32 CPU AdamW state."""
from benchmark_llada import ROOT
import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch


def file_sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def learning_rate(step,total,config):
    assert 1<=step<=total
    warm=math.ceil(config['warmup_fraction']*total)
    decay=math.ceil(config['decay_fraction']*total)
    peak=config['peak_learning_rate']
    if step<=warm:
        return peak*step/warm
    if step<=total-decay:
        return peak
    return peak+(config['final_learning_rate']-peak)*(step-(total-decay))/decay


def diffusion_loss(logits,ids,mask,probability):
    if not bool(mask.any()):
        return logits[...,0].float().sum()*0.0
    return (torch.nn.functional.cross_entropy(logits[mask].float(),ids[mask],reduction='none')/probability[mask]).sum()/ids.numel()


class CpuMasterAdamW:
    def __init__(self,parameters,config):
        self.params=list(parameters)
        self.buffers={}
        self.masters=[torch.nn.Parameter(p.detach().to(device='cpu',dtype=torch.float32,copy=True)) for p in self.params]
        self.optimizer=torch.optim.AdamW(self.masters,lr=config['peak_learning_rate'],
            betas=tuple(config['betas']),eps=config['epsilon'],weight_decay=config['weight_decay'],foreach=False,fused=False)
        self.clip=config['gradient_clip_norm']
        self.handles=[p.register_post_accumulate_grad_hook(self._accumulate) for p in self.params]

    def _accumulate(self,p):
        with torch.no_grad():
            if id(p) not in self.buffers:
                self.buffers[id(p)]=torch.zeros_like(p,dtype=torch.float32)
            self.buffers[id(p)].add_(p.grad)
            p.grad=None

    def begin(self):
        self.optimizer.zero_grad(set_to_none=True)
        for p in self.params:
            p.grad=None
        for b in self.buffers.values():
            b.zero_()

    def step(self,lr):
        for group in self.optimizer.param_groups:
            group['lr']=lr
        t0=time.perf_counter()
        for p,m in zip(self.params,self.masters):
            assert id(p) in self.buffers,'Missing parameter gradient'
            m.grad=self.buffers[id(p)].detach().to(device='cpu',dtype=torch.float32,copy=True)
        grad_copy=time.perf_counter()-t0
        t1=time.perf_counter()
        norm=torch.nn.utils.clip_grad_norm_(self.masters,self.clip,foreach=False)
        if not torch.isfinite(norm) or norm<=0:
            raise RuntimeError(f'Invalid accumulated gradient norm: {norm}')
        old=self.masters[0].flatten()[:8192].detach().clone()
        self.optimizer.step()
        changed=bool((self.masters[0].flatten()[:8192].detach()!=old).any())
        if not changed:
            raise RuntimeError('FP32 master parameter probe did not change')
        cpu_step=time.perf_counter()-t1
        t2=time.perf_counter()
        with torch.no_grad():
            for p,m in zip(self.params,self.masters):
                p.copy_(m)
        if self.params[0].is_cuda:
            torch.cuda.synchronize()
        return dict(gradient_norm=float(norm),master_parameter_changed=changed,
            gradient_copy_seconds=grad_copy,cpu_optimizer_seconds=cpu_step,
            weight_copy_seconds=time.perf_counter()-t2)


def validate_checkpoint(path,expected):
    from safetensors import safe_open
    found={}
    locations={}
    shard_files=sorted(path.glob('*.safetensors'))
    if not shard_files:
        raise RuntimeError('No checkpoint shards')
    for p in shard_files:
        with safe_open(str(p),framework='pt',device='cpu') as f:
            for key in f.keys():
                if key in found:
                    raise RuntimeError(f'Duplicate checkpoint key: {key}')
                s=f.get_slice(key)
                found[key]=(list(s.get_shape()),s.get_dtype())
                locations[key]=p.name
    assert found==expected,'Checkpoint tensor headers differ from model state'
    index=path/'model.safetensors.index.json'
    if index.exists():
        weight_map=json.loads(index.read_text())['weight_map']
        assert set(weight_map)==set(found)
        assert weight_map==locations,'Checkpoint index points keys to incorrect shards'
        assert all(Path(name).name==name and (path/name).is_file() for name in weight_map.values())
    return dict(total_weight_bytes=sum(p.stat().st_size for p in shard_files),
        tensor_count=len(found),files={p.name:dict(bytes=p.stat().st_size,sha256=file_sha(p))
            for p in sorted(path.iterdir()) if p.is_file() and p.name!='COMPLETE.json'})


def diagnostics(model,streams,config,epoch):
    model.eval()
    records=[]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
        for group,tokens in streams.items():
            full=len(tokens)//config['sequence_length']
            positions=np.linspace(0,full-1,min(config['diagnostic_blocks_per_group'],full),dtype=int)
            for j,block in enumerate(positions):
                ids=tokens[block*4096:(block+1)*4096][None,:]
                for rep in range(config['diagnostic_mask_draws_per_block']):
                    torch.manual_seed(71001+100*j+rep)
                    prob=(0.001+0.999*torch.rand((1,1),device=ids.device)).expand_as(ids)
                    mask=torch.rand(ids.shape,device=ids.device)<prob
                    noisy=torch.where(mask,config['mask_token_id'],ids)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(input_ids=noisy,use_cache=False).logits
                        loss=diffusion_loss(logits,ids,mask,prob)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite diagnostic loss')
                    records.append(dict(group=group,block=int(block),draw=rep,loss=float(loss)))
                    del logits,loss
    model.train()
    return dict(epoch=epoch,metric='fixed-noise masked diffusion objective; not exact likelihood or extraction',
        means={g:float(np.mean([r['loss'] for r in records if r['group']==g])) for g in streams},records=records)


def run(config_path):
    import psutil
    from transformers import AutoModel,AutoTokenizer
    config_path=Path(config_path)
    config=json.loads(config_path.read_text())
    assert config['sequence_length']==4096 and config['microbatch']==1
    name=config['run_name']
    assert Path(name).name==name and name not in ('.','..')
    run_dir=ROOT/'runs'/name
    if run_dir.exists():
        raise RuntimeError(f'Run already exists; refusing to overwrite: {run_dir}')
    run_dir.mkdir(parents=True)
    start=time.perf_counter()
    state=dict(run=name,status='preparing',epoch=0,update=0)
    store=None
    api_run_folder=None

    def emit(kind,**fields):
        record=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=kind,**fields)
        with (run_dir/'events.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        state.update(last_event=kind,last_time_utc=record['time_utc'],elapsed_seconds=time.perf_counter()-start)
        (run_dir/'status.json').write_text(json.dumps(state,indent=2))
        if store is not None and api_run_folder is not None:
            store.write_state(api_run_folder,'status_api.json',state)
        print(json.dumps(record),flush=True)

    try:
        methods=ROOT/'docs'/config.get('methods_filename','LLADA_BBC_METHOD.md')
        assert methods.is_file(),'Methods document must be saved before training'
        meta=json.loads((ROOT/'datasets/manifest.json').read_text())
        model_path=config.get('model_path_override') or meta['model_path']
        data=json.loads((ROOT/'datasets/training_data_manifest.json').read_text())
        assert data['train_tokens']==3941989 and data['train_blocks_4096']==963
        assert data['model_revision']==meta['model_revision'] and data['dataset_revision']==meta['dataset_revision']
        for filename,details in data['token_files'].items():
            p=ROOT/'datasets'/filename
            assert p.stat().st_size==details['bytes'] and file_sha(p)==details['sha256']
        if psutil.virtual_memory().available<140*2**30:
            raise RuntimeError('Need at least 140 GiB available CPU RAM for this tested configuration')
        payload=8015581184*2
        if config.get('checkpoint_backend')=='drive_api':
            from drive_checkpoint_upload import DriveCheckpointStore
            store=DriveCheckpointStore()
            quota=store.service.about().get(fields='storageQuota').execute(num_retries=5)['storageQuota']
            # Peak: retained epochs + old rolling + new upload. Local staging
            # is on Colab disk and does not consume additional Drive quota.
            if quota.get('limit') and int(quota['limit'])-int(quota.get('usage',0))<(config['epochs']+1)*payload+2*2**30:
                raise RuntimeError('Insufficient actual Drive quota for epoch snapshots plus rolling upload staging')
            if shutil.disk_usage('/content').free<payload+4*2**30:
                raise RuntimeError('Insufficient Colab local disk for model serialization')
            # API-owned journal is separate from mount-written bookkeeping.
            api_run_folder=store.ensure_path(['LLaDA-ft','run-api-records',name])
        elif shutil.disk_usage(ROOT).free<config['epochs']*payload+5*2**30:
            raise RuntimeError('Insufficient reported Drive space for all epoch model snapshots')
        assert torch.cuda.is_available()
        torch.set_num_threads(config['cpu_threads'])
        torch.manual_seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32=True
        sources=[Path(__file__),Path('/content/prepare_training_data.py'),Path('/content/benchmark_llada.py')]
        if store is not None:
            sources.append(Path('/content/drive_checkpoint_upload.py'))
        for p in sources:
            shutil.copy2(p,run_dir/p.name)
        shutil.copy2(config_path,run_dir/'config.json')
        shutil.copy2(methods,run_dir/'METHODS.md')
        shutil.copy2(ROOT/'datasets/training_data_manifest.json',run_dir/'data_manifest.json')
        environment=dict(python=__import__('sys').version,
            packages={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','accelerate','datasets','safetensors')},
            gpu=torch.cuda.get_device_name(0),gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
            cuda_runtime=torch.version.cuda,
            source_sha256={p.name:file_sha(p) for p in sources},config_sha256=file_sha(config_path),methods_sha256=file_sha(methods),
            model_revision=meta['model_revision'],dataset_revision=meta['dataset_revision'],model_checkpoint_location=model_path)
        (run_dir/'environment.json').write_text(json.dumps(environment,indent=2))
        if store is not None:
            for filename in ['config.json','METHODS.md','environment.json','data_manifest.json']+[p.name for p in sources]:
                store.upload(run_dir/filename,api_run_folder)
        model=AutoModel.from_pretrained(model_path,trust_remote_code=True,torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,device_map={'':0})
        model.config.use_cache=False
        mod=importlib.import_module(model.model.__class__.__module__)
        model.model.set_activation_checkpointing(mod.ActivationCheckpointingStrategy.whole_layer)
        params=[p for p in model.parameters() if p.requires_grad]
        assert sum(p.numel() for p in params)==8015581184
        assert all(p.dtype==torch.bfloat16 and p.is_cuda for p in params)
        tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        streams={key:torch.tensor(np.load(ROOT/'datasets'/filename).astype(np.int64),device='cuda')
            for key,filename in [('train','train_tokens.npy'),('clean_odd_controls','control_tokens_clean.npy')]}
        count=math.ceil(len(streams['train'])/4096)
        per_epoch=math.ceil(count/config['accumulation'])
        total_updates=per_epoch*config['epochs']
        assert count==963 and per_epoch==61 and total_updates==183
        plan=dict(config=config,blocks_per_epoch=count,updates_per_epoch=per_epoch,total_updates=total_updates,
            warmup_updates=math.ceil(config['warmup_fraction']*total_updates),
            decay_updates=math.ceil(config['decay_fraction']*total_updates),model_tensor_bytes=payload)
        (run_dir/'plan.json').write_text(json.dumps(plan,indent=2))
        if store is not None:
            store.upload(run_dir/'plan.json',api_run_folder)
        emit('baseline_started')
        baseline=diagnostics(model,streams,config,0)
        (run_dir/'diagnostics_epoch_000.json').write_text(json.dumps(baseline,indent=2))
        if store is not None:
            store.upload(run_dir/'diagnostics_epoch_000.json',api_run_folder)
        emit('baseline_complete',means=baseline['means'])
        opt=CpuMasterAdamW(params,config)
        model.train()
        state['status']='checkpoint_preflight' if config.get('verify_full_checkpoint_before_training') else 'running'
        previous_periodic=None

        def export_checkpoint(tag,epoch,update,is_periodic):
            nonlocal previous_periodic
            state['active_checkpoint']=tag
            if store is not None:
                destination=Path('/content/llada-checkpoint-staging')/name/tag
            else:
                if shutil.disk_usage(ROOT).free<payload+2*2**30:
                    raise RuntimeError('Insufficient reported Drive space for next model checkpoint')
                destination=ROOT/'checkpoints'/name/tag
            destination.mkdir(parents=True,exist_ok=False)
            emit('checkpoint_started',epoch=epoch,update=update,path=str(destination),periodic=is_periodic)
            expected={key:(list(value.shape),{'torch.bfloat16':'BF16','torch.float32':'F32','torch.int64':'I64'}[str(value.dtype)])
                for key,value in model.state_dict().items()}
            model.save_pretrained(destination,safe_serialization=True,max_shard_size='2GB')
            tokenizer.save_pretrained(destination)
            for p in Path(model_path).glob('*.py'):
                shutil.copy2(p,destination/p.name)
            shutil.copy2(run_dir/'config.json',destination/'training_config.json')
            shutil.copy2(run_dir/'METHODS.md',destination/'METHODS.md')
            checkpoint=validate_checkpoint(destination,expected)
            checkpoint.update(epoch=epoch,update=update,run=name,optimizer_saved=False,
                exact_training_resume_supported=False,model_revision=meta['model_revision'],
                config_sha256=environment['config_sha256'],methods_sha256=environment['methods_sha256'])
            (destination/'COMPLETE.json').write_text(json.dumps(checkpoint,indent=2))
            if store is not None:
                folder=store.ensure_path(['LLaDA-ft','checkpoints',name,tag])
                progress_last={}
                def progress(filename,done,total):
                    if done-progress_last.get(filename,0)>=256*2**20 or done==total:
                        progress_last[filename]=done
                        state.update(upload_file=filename,upload_bytes=done,upload_total_bytes=total)
                        emit('checkpoint_upload_progress',checkpoint=tag,file=filename,bytes=done,total=total)
                receipt=store.upload_directory(destination,folder,progress)
                receipt.update(checkpoint=tag,epoch=epoch,update=update,periodic=is_periodic)
                receipt_path=run_dir/(tag+'_upload_receipt.json')
                receipt_path.write_text(json.dumps(receipt,indent=2))
                store.upload(receipt_path,api_run_folder)
                state['last_checkpoint']=dict(tag=tag,epoch=epoch,update=update,url=receipt['folder_url'],verified=True,model_only=True)
                emit('checkpoint_complete',epoch=epoch,update=update,tag=tag,url=receipt['folder_url'],weight_bytes=checkpoint['total_weight_bytes'])
                if previous_periodic is not None:
                    store.delete_own_periodic(previous_periodic)
                    emit('old_periodic_checkpoint_removed',folder_id=previous_periodic['folder_id'])
                previous_periodic=receipt if is_periodic else None
                # Remove only this verified checkpoint's temporary local staging directory.
                assert destination.resolve().is_relative_to(Path('/content/llada-checkpoint-staging').resolve())
                shutil.rmtree(destination)
            else:
                emit('checkpoint_complete',epoch=epoch,path=str(destination),weight_bytes=checkpoint['total_weight_bytes'])
            for key in ['active_checkpoint','upload_file','upload_bytes','upload_total_bytes']:
                state.pop(key,None)

        if config.get('verify_full_checkpoint_before_training'):
            export_checkpoint('base-transport-preflight',0,0,True)
        state['status']='running'
        emit('training_started',plan={k:v for k,v in plan.items() if k!='config'})
        global_update=0
        for epoch in range(1,config['epochs']+1):
            state['epoch']=epoch
            order_rng=np.random.default_rng(config['seed']+epoch)
            order=order_rng.permutation(count)
            crop_rng=np.random.default_rng(config['seed']+10000+epoch)
            emit('epoch_started',epoch=epoch,block_order_sha256=hashlib.sha256(order.tobytes()).hexdigest())
            epoch_tokens=epoch_crops=0
            for offset in range(0,count,config['accumulation']):
                indices=order[offset:offset+config['accumulation']]
                opt.begin()
                torch.cuda.synchronize();t0=time.perf_counter()
                losses=[]
                window_tokens=window_crops=0
                for block in indices:
                    ids=streams['train'][block*4096:min((block+1)*4096,len(streams['train']))][None,:]
                    if crop_rng.random()<config['random_length_probability']:
                        length=int(crop_rng.integers(config['random_length_min'],ids.shape[1]+1))
                        ids=ids[:,:length]
                        window_crops+=1
                    window_tokens+=ids.numel()
                    prob=(config['minimum_mask_probability']+(1-config['minimum_mask_probability'])*torch.rand((1,1),device='cuda')).expand_as(ids)
                    mask=torch.rand(ids.shape,device='cuda')<prob
                    noisy=torch.where(mask,config['mask_token_id'],ids)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(input_ids=noisy,use_cache=False).logits
                        loss=diffusion_loss(logits,ids,mask,prob)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite training loss at epoch {epoch}, block {block}')
                    (loss/len(indices)).backward()
                    losses.append(float(loss.detach()))
                    del logits,loss
                torch.cuda.synchronize()
                compute=time.perf_counter()-t0
                global_update+=1
                lr=learning_rate(global_update,total_updates,config)
                result=opt.step(lr)
                epoch_tokens+=window_tokens
                epoch_crops+=window_crops
                state.update(update=global_update,epoch_tokens=epoch_tokens)
                emit('optimizer_update',epoch=epoch,update=global_update,epoch_update=offset//config['accumulation']+1,
                    window_size=len(indices),mean_loss=float(np.mean(losses)),learning_rate=lr,
                    processed_tokens=window_tokens,random_crops=window_crops,
                    forward_backward_seconds=compute,update_seconds=time.perf_counter()-t0,
                    gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,
                    cpu_rss_gib=psutil.Process().memory_info().rss/2**30,**result)
                cadence=config.get('checkpoint_every_updates',0)
                epoch_remaining=per_epoch-(offset//config['accumulation']+1)
                if cadence and global_update%cadence==0 and epoch_remaining>5:
                    export_checkpoint(f'step-{global_update:06d}',epoch,global_update,True)
            opt.optimizer.zero_grad(set_to_none=True)
            emit('epoch_training_complete',epoch=epoch,processed_tokens=epoch_tokens,random_crops=epoch_crops)
            diag=diagnostics(model,streams,config,epoch)
            (run_dir/f'diagnostics_epoch_{epoch:03d}.json').write_text(json.dumps(diag,indent=2))
            if store is not None:
                store.upload(run_dir/f'diagnostics_epoch_{epoch:03d}.json',api_run_folder)
            emit('diagnostics_complete',epoch=epoch,means=diag['means'])
            export_checkpoint(f'epoch-{epoch:03d}',epoch,global_update,False)
        assert global_update==total_updates
        state['status']='complete'
        emit('training_complete',updates=global_update)
    except BaseException as error:
        state.update(status='failed',error=str(error),traceback=traceback.format_exc())
        emit('training_failed',error=str(error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    run(args.config)
