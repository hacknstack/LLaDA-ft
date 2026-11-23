"""Score the same complete random reveal paths at several temperatures."""
import math
import torch
import probabilistic_extraction as pe


@torch.inference_mode()
def joint_random(model,sequence,masked_indexes,temperatures=(.5,2.),samples=500,seed=42,batch_size=64):
    device=next(model.parameters()).device
    positions=torch.tensor(sorted(i-1 for i in masked_indexes),device=device)
    targets=sequence[0,positions]
    rng=torch.Generator(device='cpu').manual_seed(seed)
    permutations=pe._uniform_path_permutations(samples,len(masked_indexes),rng,False,seed).to(device)
    projection=pe._random_remasking_projection_layer(model)
    assert projection is not None
    initial=sequence.clone();initial[:,positions]=126336
    logits=pe._random_remasking_forward_logits(model,initial,None,positions.unsqueeze(0),projection)
    initial_scores=[]
    for temperature in temperatures:
        value=pe._random_path_target_log_probs(logits.clone(),targets.unsqueeze(0),temperature,'full',1)[0]
        assert not (torch.isnan(value)|torch.isposinf(value)).any()
        initial_scores.append(value.clamp_max(0))
    all_logs=[[] for _ in temperatures]
    forward_calls=1
    for start in range(0,samples,batch_size):
        paths=permutations[start:start+batch_size]
        bsz=len(paths);x=initial.expand(bsz,-1).clone()
        first=paths[:,:1]
        weights=[scores[first].to(torch.float64).sum(-1) for scores in initial_scores]
        x.scatter_(1,positions[first],targets[first])
        for step in range(1,len(masked_indexes)):
            slots=paths[:,step:step+1]
            selected_positions=positions[slots]
            selected_targets=targets[slots]
            logits=pe._random_remasking_forward_logits(model,x,None,selected_positions,projection)
            forward_calls+=1
            for i,temperature in enumerate(temperatures):
                logs=pe._random_path_target_log_probs(logits.clone(),selected_targets,temperature,'full',1)
                assert not (torch.isnan(logs)|torch.isposinf(logs)).any()
                logs.clamp_max_(0)
                weights[i]+=logs.to(torch.float64).sum(-1)
            x.scatter_(1,selected_positions,selected_targets)
        for i,weight in enumerate(weights):all_logs[i].extend(weight.cpu().tolist())
    output=[]
    for temperature,logs in zip(temperatures,all_logs):
        values=torch.tensor(logs,dtype=torch.float64)
        logp=float(torch.logsumexp(values,0)-math.log(samples))
        output.append(dict(temperature=temperature,probability=math.exp(logp),log_probability=logp,
                           sample_log_probabilities=logs,num_samples=samples,steps=len(masked_indexes),
                           shared_model_forward_calls=forward_calls))
    return output
