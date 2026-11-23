"""Check sequence-mean weighting across physical batches and a short tail."""
import json
from pathlib import Path
import torch
from ar_training_utils import ar_loss

torch.manual_seed(42)
torch.set_num_threads(2)
records=[]
for objective in ['AR','diffusion']:
    logits=[torch.randn(11 if i<3 else 5,19,dtype=torch.float64,requires_grad=True) for i in range(4)]
    ids=[torch.randint(19,(len(x),)) for x in logits]
    probabilities=[torch.full((len(x),),p,dtype=torch.float64) for x,p in zip(logits,[.2,.4,.7,.9])]
    masks=[torch.rand(len(x))<p[0] for x,p in zip(logits,probabilities)]
    def calc(x,y,mask,prob):
        if objective=='AR':return ar_loss(x,y)
        if not bool(mask.any()):return x[...,0].sum()*0.
        return (torch.nn.functional.cross_entropy(x[mask],y[mask],reduction='none')/prob[mask]).sum()/y.numel()
    separate=sum(calc(x[None],y[None],m[None],p[None]) for x,y,m,p in zip(logits,ids,masks,probabilities))/4
    a=calc(torch.stack(logits[:3]),torch.stack(ids[:3]),torch.stack(masks[:3]),torch.stack(probabilities[:3]))
    b=calc(logits[3][None],ids[3][None],masks[3][None],probabilities[3][None])
    grouped=a*3/4+b/4
    torch.testing.assert_close(separate,grouped,rtol=1e-6,atol=1e-7)
    ga=torch.autograd.grad(separate,logits,retain_graph=True)
    gb=torch.autograd.grad(grouped,logits)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-7)
    records.append(dict(objective=objective,per_sequence_mean_preserved=True,short_tail_weight_preserved=True,gradient_check_passed=True))
out=dict(passed=True,records=records,scope='Loss aggregation and gradients; physical batching can still change finite-precision model kernels and RNG ordering.')
Path('/workspace/bbc-exposure/audits/batch_objective_validation.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
