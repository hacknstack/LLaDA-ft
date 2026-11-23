"""Independent multi-update comparison of transfer-only optimizer optimization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json
from pathlib import Path
import torch
from ar_training_utils import CpuMasterAdamW
from async_optimizer import AsyncCpuMasterAdamW

torch.set_num_threads(12)
torch.manual_seed(42)
config=dict(peak_learning_rate=1e-5,betas=[.9,.999],epsilon=1e-8,weight_decay=.1,gradient_clip_norm=1.)
a=torch.nn.Sequential(torch.nn.Linear(128,256),torch.nn.GELU(),torch.nn.Linear(256,64)).to('cuda',dtype=torch.bfloat16)
b=torch.nn.Sequential(torch.nn.Linear(128,256),torch.nn.GELU(),torch.nn.Linear(256,64)).to('cuda',dtype=torch.bfloat16)
b.load_state_dict(a.state_dict())
oa=CpuMasterAdamW(a.parameters(),config)
ob=AsyncCpuMasterAdamW(b.parameters(),config)
for step in range(4):
    oa.begin();ob.begin()
    for _ in range(16):
        x=torch.randn((3,128),device='cuda',dtype=torch.bfloat16)
        y=torch.randint(0,64,(3,),device='cuda')
        la=torch.nn.functional.cross_entropy(a(x).float(),y)
        lb=torch.nn.functional.cross_entropy(b(x).float(),y)
        assert torch.equal(la,lb)
        (la/16).backward();(lb/16).backward()
    ra=oa.step(1e-5);rb=ob.step(1e-5)
    assert ra['gradient_norm']==rb['gradient_norm']
    for pa,pb in zip(a.parameters(),b.parameters()):assert torch.equal(pa,pb)
    for ma,mb in zip(oa.masters,ob.masters):
        assert torch.equal(ma,mb)
        for key in oa.optimizer.state[ma]:
            va=oa.optimizer.state[ma][key];vb=ob.optimizer.state[mb][key]
            assert torch.equal(va,vb) if torch.is_tensor(va) else va==vb
report=dict(passed=True,updates=4,microbatches_per_update=16,model_weights_bit_identical=True,
            fp32_master_weights_bit_identical=True,optimizer_moments_bit_identical=True,gradient_norms_identical=True,
            scope='Small deterministic CUDA network, same reference optimizer; full-model resource probes required separately.')
Path('/workspace/bbc-exposure/audits/async_optimizer_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
