import torch
import time

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
