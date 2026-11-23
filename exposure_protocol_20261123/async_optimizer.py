"""Same CPU FP32 AdamW arithmetic; pinned reusable buffers and async copies only."""
import time
import torch
from ar_training_utils import CpuMasterAdamW


class AsyncCpuMasterAdamW(CpuMasterAdamW):
    def __init__(self,parameters,config):
        super().__init__(parameters,config)
        for group in self.optimizer.param_groups:group['foreach']=config.get('foreach',False)
        # Keep the same Parameter objects referenced by the same AdamW instance.
        for m in self.masters:m.data=m.data.pin_memory()
        self.host_grads={}

    def step(self,lr):
        for group in self.optimizer.param_groups:group['lr']=lr
        started=time.perf_counter()
        for p,m in zip(self.params,self.masters):
            assert id(p) in self.buffers
            if id(p) not in self.host_grads:
                self.host_grads[id(p)]=torch.empty_like(m,pin_memory=True)
            grad=self.host_grads[id(p)]
            grad.copy_(self.buffers[id(p)],non_blocking=True)
            m.grad=grad
        torch.cuda.synchronize()
        grad_copy=time.perf_counter()-started
        started=time.perf_counter()
        norm=torch.nn.utils.clip_grad_norm_(self.masters,self.clip,foreach=False)
        if not torch.isfinite(norm) or norm<=0:raise RuntimeError('Invalid accumulated gradient norm')
        old=self.masters[0].flatten()[:8192].detach().clone()
        self.optimizer.step()
        changed=bool((self.masters[0].flatten()[:8192].detach()!=old).any())
        if not changed:raise RuntimeError('FP32 master parameter probe did not change')
        cpu_step=time.perf_counter()-started
        started=time.perf_counter()
        with torch.no_grad():
            for p,m in zip(self.params,self.masters):p.copy_(m,non_blocking=True)
        torch.cuda.synchronize()
        return dict(gradient_norm=float(norm),master_parameter_changed=changed,
                    gradient_copy_seconds=grad_copy,cpu_optimizer_seconds=cpu_step,
                    weight_copy_seconds=time.perf_counter()-started)
