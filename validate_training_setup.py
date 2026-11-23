"""Focused CPU checks for expensive production training decisions."""
from pathlib import Path
import json
import tempfile
import torch
from safetensors.torch import save_file
from train_llada_bbc import CpuMasterAdamW, diffusion_loss, learning_rate, validate_checkpoint

config=json.loads(Path('/content/llada_training_config.json').read_text())
assert abs(learning_rate(1,183,config)-1e-6)<1e-15
assert learning_rate(10,183,config)==1e-5
assert learning_rate(164,183,config)==1e-5
assert abs(learning_rate(183,183,config)-1e-6)<1e-15
assert all(learning_rate(s,183,config)>learning_rate(s+1,183,config) for s in range(164,183))

torch.manual_seed(9)
model=torch.nn.Linear(5,3)
reference=torch.nn.Linear(5,3)
reference.load_state_dict(model.state_dict())
offload=CpuMasterAdamW(model.parameters(),config)
assert all(p.data_ptr()!=m.data_ptr() for p,m in zip(offload.params,offload.masters))
refopt=torch.optim.AdamW(reference.parameters(),lr=1e-5,betas=tuple(config['betas']),eps=config['epsilon'],weight_decay=.1,foreach=False,fused=False)
for n in (16,3):
    x=torch.randn(n,5);y=torch.randn(n,3)
    refopt.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(reference(x),y).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(),1.0,foreach=False)
    refopt.step()
    offload.begin()
    for i in range(n):
        (torch.nn.functional.mse_loss(model(x[i:i+1]),y[i:i+1])/n).backward()
    assert all(p.grad is None for p in model.parameters())
    offload.step(1e-5)
    for p,r in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(p,r,rtol=1e-6,atol=1e-7)
assert {v.dtype for s in offload.optimizer.state.values() for v in s.values() if torch.is_tensor(v)}=={torch.float32}

logits=torch.tensor([[[1.,2.,3.],[3.,2.,1.]]],requires_grad=True)
ids=torch.tensor([[2,1]])
prob=torch.full((1,2),.5)
mask=torch.tensor([[True,False]])
loss=diffusion_loss(logits,ids,mask,prob)
expected=-torch.log_softmax(logits,dim=-1)[0,0,2]/.5/2
torch.testing.assert_close(loss,expected)
zero=diffusion_loss(logits,ids,torch.zeros_like(mask),prob)
assert zero.item()==0
zero.backward()
assert torch.equal(logits.grad,torch.zeros_like(logits))

with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp)
    save_file({'a':torch.ones(2,3,dtype=torch.bfloat16)},str(out/'model.safetensors'))
    (out/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'model.safetensors'}}))
    report=validate_checkpoint(out,{'a':([2,3],'BF16')})
    assert report['tensor_count']==1
    try:
        validate_checkpoint(out,{'a':([3,2],'BF16')})
    except AssertionError:
        pass
    else:
        raise AssertionError('Shape corruption not detected')
print('TRAINING_SETUP_VALIDATED: schedule, full/remainder averaged updates, independent FP32 masters, loss alignment, zero-mask handling, checkpoint headers/index/hashes')
