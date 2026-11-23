import torch
import time
import math
import hashlib
import json
from pathlib import Path

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

def ar_loss(logits, ids):
    import torch
    if ids.shape[1] < 2:
        return logits[..., 0].float().sum() * 0.0
    return torch.nn.functional.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                                             ids[:, 1:].reshape(-1), reduction='mean')
