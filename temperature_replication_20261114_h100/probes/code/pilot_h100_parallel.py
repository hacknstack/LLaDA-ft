"""Bounded allocation-only probe; estimator code, RNG streams and task inputs stay fixed."""
import argparse
import hashlib
import json
import os
import threading
import time
import traceback
from pathlib import Path

ROOT=Path('/workspace/temperature-study')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--temperature',type=float,default=2.)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--contexts',type=int,default=1)
    parser.add_argument('--mode',choices=['within-case','broker'],required=True)
    parser.add_argument('--bulk',action='store_true')
    parser.add_argument('--scale',type=int,default=1)
    parser.add_argument('--cases',type=int,choices=[4,8],default=8)
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=json.loads((ROOT/'mig_uuids.json').read_text())[args.gpu]
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from transformers import AutoModel
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from allocation_broker_scalar import WithinCase,InferenceBroker,CrossCase
    import experimental_prefetch_extraction as pe
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    model=AutoModel.from_pretrained(ROOT/'models/llada-base',trust_remote_code=True,local_files_only=True,
         torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':0}).eval()
    model.config.use_cache=False
    data=json.loads((ROOT/'pilot_inputs.json').read_text())
    chosen=[data['selected']['validation'][i] for i in (0,3,6,9,12,15,18,23)]
    tasks=[]
    for i,row in enumerate(chosen):
        # Cases and draw counts are identical across all allocation configurations.
        method='path_sampling' if i%2==0 else 'monte-carlo'
        scheme='low-confidence' if (i//2)%2==0 else 'threshold'
        tasks.append(dict(index=i,document=row['document'],window_number=row['window_number'],
                          mask_id=row['mask_id'],method=method,scheme=scheme,temperature=args.temperature,
                          samples=(16 if method=='path_sampling' else 64)*args.scale,seed=20261112+i))
    if args.cases==4:tasks=[tasks[i] for i in (0,2,5,7)]
    brokers=[InferenceBroker(model,bulk=args.bulk) for _ in range(args.contexts)] if args.mode=='broker' else []
    broker=brokers[0] if brokers else None
    assignment_lock=threading.Lock(); assignment_index=[0]
    tls=threading.local()
    def evaluate(task):
        if not hasattr(tls,'stream'):
            tls.stream=torch.cuda.Stream()
            with assignment_lock:
                tls.broker=brokers[assignment_index[0]] if brokers else None
                assignment_index[0]+=1
        with torch.inference_mode(),torch.cuda.stream(tls.stream):
            wrapped=WithinCase(model) if broker is None else CrossCase(tls.broker)
            sequence=torch.tensor([data['tokens'][task['document']+':'+task['window_number']]],device='cuda')
            common=dict(model=wrapped,sequence_tokens=sequence,masked_indexes=data['masks'][task['mask_id']-1],
                 steps=50,attention_mask=None,mask_id=126336,num_samples=task['samples'],seed=task['seed'],temperature=args.temperature)
            extra={'confidence_threshold':.9} if task['scheme']=='threshold' else {}
            start=time.perf_counter()
            if task['method']=='path_sampling':
                fn=(pe._path_sampling_low_confidence_probability_fast_from_partially_masked
                    if task['scheme']=='low-confidence' else pe._path_sampling_threshold_probability_fast_from_partially_masked)
                result=fn(**common,**extra)
                probability=result['probability'];logp=result['log_probability'];weights=result['sample_log_probabilities']
            else:
                fn=(pe._monte_carlo_probability_temperature_fast_from_partially_masked
                    if task['scheme']=='low-confidence' else pe._monte_carlo_threshold_probability_fast_from_partially_masked)
                result=fn(**common,**extra,decoding_scheme='full',k=1,return_sample_logs=True)
                probability=result.estimate;logp=None;weights=result.sample_log_probabilities
            tls.stream.synchronize()
            signature=dict(probability_hex=float(probability).hex(),log_probability_hex=None if logp is None else float(logp).hex(),
                           sample_log_probabilities_hex=[float(v).hex() for v in weights])
            output=dict(task,seconds=time.perf_counter()-start,signature=signature)
            print('TASK_DONE',args.gpu,task['index'],round(output['seconds'],3),flush=True)
            return output
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    results=[]
    try:
        if args.mode=='within-case':
            for task in tasks:results.append(evaluate(task))
        else:
            with ThreadPoolExecutor(max_workers=args.contexts) as pool:
                pending=[pool.submit(evaluate,task) for task in tasks]
                for future in as_completed(pending):results.append(future.result())
    finally:
        for b in brokers:b.stop()
    torch.cuda.synchronize()
    output=dict(mode=args.mode,contexts=args.contexts,gpu=args.gpu,bulk=args.bulk,scale=args.scale,cases=args.cases,total_seconds=time.perf_counter()-start,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                tasks=sorted(results,key=lambda r:r['index']),broker=[b.stats() for b in brokers],
                budget_only=True,scientific_runtime_study=False,full_experiment_started=False)
    suffix=('_bulk' if args.bulk else '')+f'_T{args.temperature}'
    if args.scale!=1 or args.cases!=8:suffix+=f'_scale{args.scale}_cases{args.cases}'
    destination=ROOT/'h100_parallel_probes'/f'{args.mode}_contexts{args.contexts}_gpu{args.gpu}{suffix}.json'
    destination.write_text(json.dumps(output,indent=2))
    print('ALLOCATION_CONFIG_COMPLETE',args.gpu,output['total_seconds'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        parser=argparse.ArgumentParser();parser.add_argument('--gpu');parser.add_argument('--contexts');parser.add_argument('--mode');parser.add_argument('--bulk',action='store_true');parser.add_argument('--scale',type=int,default=1);parser.add_argument('--cases',type=int,default=8);a=parser.parse_args()
        root=ROOT/'h100_parallel_probes';root.mkdir(exist_ok=True)
        suffix='_bulk' if a.bulk else ''
        if a.scale!=1 or a.cases!=8:suffix+=f'_scale{a.scale}_cases{a.cases}'
        (root/f'ERROR_{a.mode}_contexts{a.contexts}_gpu{a.gpu}{suffix}.json').write_text(json.dumps(dict(error=traceback.format_exc())))
        raise
