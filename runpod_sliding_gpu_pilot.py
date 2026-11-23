"""Matched hardware pilot; unchanged estimator, public pinned Base, no full sweep."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HOME='/workspace/sliding-benchmark/hf',HF_HUB_DISABLE_XET='1',
                  TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
import torch
from huggingface_hub import HfApi,hf_hub_download,snapshot_download
from transformers import AutoTokenizer,AutoModel

ROOT=Path('/workspace/sliding-benchmark')
MODEL='GSAI-ML/LLaDA-8B-Base'
REV='0f2787f2d87eac5eed8a087d5ecd24277e6255b2'
DATA='RealTimeData/bbc_news_alltime'
DATA_REV='8dd1ecdc92ac43f9c04a3da3e945537dbb08179b'


def main():
    import pyarrow.parquet as pq
    from probabilistic_extraction import (
        _path_sampling_low_confidence_probability_fast_from_partially_masked as low,
        _path_sampling_threshold_probability_fast_from_partially_masked as threshold)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=True
    gpu=torch.cuda.get_device_name(0)
    report=dict(gpu=gpu,model_id=MODEL,model_revision=REV,model_is_finetuned=False,
        num_samples=10,sequence_tokens=100,model_dtype='bfloat16',estimator_dtype='float64',
        model_forward_batch_size=1,threshold=.9,cpu_threads=4,results=[],status='preparing',
        torch_version=torch.__version__,cuda=torch.version.cuda,
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ['probabilistic_extraction.py','get_log_likelihood.py','runpod_sliding_gpu_pilot.py']})
    def emit(event,**fields):
        (ROOT/'pilot_results.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(event=event,**fields)),flush=True)
    started=time.perf_counter()
    try:
        repo=HfApi().model_info(MODEL,revision=REV)
        names=[s.rfilename for s in repo.siblings if '/' not in s.rfilename]
        allowed=[n for n in names if n.endswith(('.safetensors','.json','.py','.model','.txt','.tiktoken'))]
        model_path=snapshot_download(MODEL,revision=REV,allow_patterns=allowed,max_workers=4)
        emit('BASE_MODEL_LOCAL',path=model_path)
        tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        files=HfApi().list_repo_files(DATA,repo_type='dataset',revision=DATA_REV)
        records=[]
        for month in ['2025-03','2025-04','2025-05','2025-06']:
            names=sorted(f for f in files if f.startswith(month+'/') and f.endswith('.parquet'))
            assert len(names)==1,(month,names)
            source=hf_hub_download(DATA,names[0],repo_type='dataset',revision=DATA_REV)
            for row in pq.read_table(source).to_pylist():
                date=str(row.get('published_date') or '')[:10]
                if date>='2025-03-01':records.append(row)
        assert len(records)==10643
        def normalized(text):return re.sub(r'\s+',' ',text).strip()
        train_hashes={hashlib.sha256(normalized(r.get('content') or '').encode()).hexdigest()
                      for i,r in enumerate(records) if i%2==0 and (r.get('content') or '').strip()}
        candidates={'train':[],'clean_odd_control':[]}
        for i,r in enumerate(records):
            text=r.get('content') or ''
            if not text.strip():continue
            if i%2 and hashlib.sha256(normalized(text).encode()).hexdigest() in train_hashes:continue
            group='train' if i%2==0 else 'clean_odd_control'
            # Identical window construction to the provided runner's first window.
            ids=tokenizer(text[:800],add_special_tokens=False)['input_ids'][:100]
            if len(ids)==100:candidates[group].append((i,ids))
        rng=random.Random(20261101)
        examples=[]
        for group in ['train','clean_odd_control']:
            idx,ids=rng.choice(candidates[group]);examples.append((group,idx,ids))
        report['examples']=[dict(group=g,filtered_index=i,token_ids=ids) for g,i,ids in examples]
        report['inputs_sha256']=hashlib.sha256(json.dumps(report['examples'],sort_keys=True).encode()).hexdigest()
        model=AutoModel.from_pretrained(model_path,trust_remote_code=True,torch_dtype=torch.bfloat16,
                                       low_cpu_mem_usage=True,device_map={'':0})
        model.eval();model.config.use_cache=False
        assert all(p.dtype==torch.bfloat16 for p in model.parameters())
        masks={'suffix50':list(range(51,101)),'alternating50':list(range(2,101,2))}
        with torch.no_grad():
            warm=torch.tensor([examples[0][2]],device='cuda')
            warm[:,50:]=126336
            for _ in range(3):model(warm)
            del warm
            torch.cuda.synchronize()
            report['status']='running'
            report['setup_seconds']=time.perf_counter()-started
            emit('PILOT_READY',gpu=gpu,input_hash=report['inputs_sha256'])
            # Two matched repetitions, alternating scheme order to reduce ordering bias.
            for repeat in range(2):
                for group,idx,ids in examples:
                    for label,mask in masks.items():
                        order=['low-confidence','threshold'] if repeat==0 else ['threshold','low-confidence']
                        for scheme in order:
                            sequence=torch.tensor([ids],device='cuda')
                            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                            t0=time.perf_counter()
                            kwargs=dict(model=model,sequence_tokens=sequence,masked_indexes=mask,steps=50,
                                attention_mask=None,mask_id=126336,num_samples=10,seed=12345+idx,
                                temperature=1.,verbose=False,use_state_cache=True,return_samples=False)
                            result=(low(**kwargs) if scheme=='low-confidence' else
                                    threshold(**kwargs,confidence_threshold=.9))
                            torch.cuda.synchronize();seconds=time.perf_counter()-t0
                            assert 0<=result['probability']<=1
                            row=dict(repeat=repeat,group=group,filtered_index=idx,mask=label,scheme=scheme,
                                seconds=seconds,probability=result['probability'],log_probability=result['log_probability'],
                                model_forward_calls=result.get('model_forward_calls'),
                                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                            report['results'].append(row)
                            emit('PILOT_CASE',**row)
                            del sequence,result;gc.collect()
        report['status']='complete'
        report['total_seconds']=time.perf_counter()-started
        report['summary']={scheme:dict(median_seconds=statistics.median(r['seconds'] for r in report['results'] if r['scheme']==scheme),
                    mean_seconds=statistics.mean(r['seconds'] for r in report['results'] if r['scheme']==scheme),
                    total_seconds=sum(r['seconds'] for r in report['results'] if r['scheme']==scheme))
                    for scheme in ['low-confidence','threshold']}
        emit('PILOT_COMPLETE',summary=report['summary'])
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        emit('PILOT_FAILED',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':main()
