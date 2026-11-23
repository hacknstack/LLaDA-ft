import json
import os
import time
from pathlib import Path

ROOT=Path('/workspace/temperature-study')


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=json.loads((ROOT/'mig_uuids.json').read_text())[4]
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from transformers import AutoModel
    import probabilistic_extraction as pe
    from joint_random_temperatures import joint_random
    torch.set_num_threads(2)
    model=AutoModel.from_pretrained(ROOT/'models/llada-base',trust_remote_code=True,local_files_only=True,
          torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':0}).eval()
    model.config.use_cache=False
    data=json.loads((ROOT/'pilot_inputs.json').read_text());row=data['selected']['validation'][0]
    sequence=torch.tensor([data['tokens'][row['document']+':'+row['window_number']]],device='cuda')
    masks=data['masks'][row['mask_id']-1]
    originals=[];times=[]
    with torch.inference_mode():
        for temperature in (.5,2.):
            torch.cuda.synchronize();start=time.perf_counter()
            originals.append(pe._path_sampling_random_probability_from_partially_masked(
                model,sequence,masks,50,None,126336,16,42,'full',1,temperature,batch_size=16,
                max_path_samples=None,max_path_steps=None,stratified_paths=False))
            torch.cuda.synchronize();times.append(time.perf_counter()-start)
        start=time.perf_counter()
        combined=joint_random(model,sequence,masks,samples=16,batch_size=16)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
    equal=all(a['sample_log_probabilities']==b['sample_log_probabilities']
              and a['log_probability']==b['log_probability'] for a,b in zip(originals,combined))
    assert equal,'Joint-temperature random scores changed'
    output=dict(validated=True,budget_only=True,bit_identical=True,separate_seconds=times,
                joint_seconds=elapsed,speedup=sum(times)/elapsed,samples_per_temperature=16,
                steps=50,shared_forward_calls=combined[0]['shared_model_forward_calls'])
    (ROOT/'results/joint_random_optimization.json').write_text(json.dumps(output,indent=2))
    print('JOINT_RANDOM_VALIDATED',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        (ROOT/'results/joint_random_optimization.json').write_text(json.dumps(dict(validated=False,error=str(e))))
        raise
