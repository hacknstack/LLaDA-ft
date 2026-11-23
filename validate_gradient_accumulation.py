"""Check post-accumulate hooks against full-batch autograd across two updates."""
import torch

torch.manual_seed(7)
model=torch.nn.Linear(5,3,bias=False)
parameter=model.weight
buffers={}

def hook(p):
    with torch.no_grad():
        if id(p) not in buffers:
            buffers[id(p)]=torch.zeros_like(p,dtype=torch.float32)
        buffers[id(p)].add_(p.grad)
        p.grad=None

for _ in range(2):
    x=torch.randn(4,5)
    y=torch.randn(4,3)
    model.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(x),y).backward()
    expected=parameter.grad.clone()
    model.zero_grad(set_to_none=True)
    for b in buffers.values():
        b.zero_()
    handle=parameter.register_post_accumulate_grad_hook(hook)
    for a,b in zip(x.chunk(2),y.chunk(2)):
        (torch.nn.functional.mse_loss(model(a),b)/2).backward()
        assert parameter.grad is None
    handle.remove()
    torch.testing.assert_close(buffers[id(parameter)],expected,rtol=1e-6,atol=1e-7)
print('FP32_ACCUMULATION_MATCHES_FULL_BATCH_AUTOGRAD')
