"""Small public AR benchmark after the diffusion pilot releases its MIG slices."""
import json
import argparse
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('/workspace/temperature-study')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--key',default='mistral-base')
    parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    if (ROOT/f'results/ar_pilot_{args.key}.json').exists():
        print('AR_PILOT_ALREADY_COMPLETE',args.key,flush=True)
        return
    os.environ['CUDA_VISIBLE_DEVICES']=json.loads((ROOT/'mig_uuids.json').read_text())[args.gpu]
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import probabilistic_extraction as pe
    torch.set_num_threads(3)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models'/args.key,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models'/args.key,local_files_only=True,
         torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':0}).eval()
    model.config.use_cache=False
    cohorts=json.loads((ROOT/'input_cohorts.json').read_text())
    windows=sorted({(r['document'],int(r['window_number'])) for r in cohorts['application']})
    chosen=[windows[round(i*(len(windows)-1)/79)] for i in range(80)]
    texts={name:(ROOT/'texts'/(name+'.txt')).read_text() for name,_ in chosen}
    rows=[]
    with torch.inference_mode():
        for document,index in chosen:
            text=texts[document];starts=[0]+[m.start() for m in re.finditer(r'\S+',text)][1:]
            tokens=tokenizer(text[starts[index]:starts[index]+800],add_special_tokens=False)['input_ids'][:100]
            if len(tokens)!=100:
                rows.append(dict(document=document,window_number=index,error='short_window',tokens=len(tokens)))
                continue
            full=torch.tensor([tokens],device='cuda')
            torch.cuda.synchronize();start=time.perf_counter()
            output=model(full,attention_mask=None)
            torch.cuda.synchronize();forward=time.perf_counter()-start
            class Reuse:
                def parameters(self):return model.parameters()
                def __call__(self,*args,**kwargs):return output
            for temperature in (.5,2.):
                start=time.perf_counter()
                result=pe._autoregressive_probability(Reuse(),full[:,:50],full[:,50:],None,
                                                     'full',1,temperature,False)
                torch.cuda.synchronize();score=time.perf_counter()-start
                rows.append(dict(document=document,window_number=index,temperature=temperature,
                                 forward_seconds=forward,score_seconds=score,
                                 probability=result['probability']))
    result=dict(model=args.key,rows=rows,
                peak_gib=torch.cuda.max_memory_allocated()/2**30,
                projected_gpu_seconds=sum(r.get('forward_seconds',0)/2+r.get('score_seconds',0)
                    for r in rows)*1899/80)
    (ROOT/f'results/ar_pilot_{args.key}.json').write_text(json.dumps(result,indent=2))
    print('AR_PILOT_COMPLETE',args.key,result['projected_gpu_seconds'],flush=True)


if __name__=='__main__':
    main()
