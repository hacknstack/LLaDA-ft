"""Experimental budget optimization; require seeded estimator results to match."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('/workspace/temperature-study')


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=json.loads((ROOT/'mig_uuids.json').read_text())[4]
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from transformers import AutoModel
    import probabilistic_extraction as reference
    torch.set_num_threads(2)
    source=(ROOT/'code/probabilistic_extraction.py').read_text()
    marker='        log_a_rows = []\n'
    assert source.count(marker)==1
    source=source.replace(marker,"        if hasattr(model, 'prefetch'):\n            model.prefetch(x[alive], attention_mask)\n"+marker)
    for marker,rows_name in [
        ('            for state_key, trajectory_rows in rows_by_state.items():\n','trajectory_rows'),
        ('            for state_key, rows in rows_by_state.items():\n','rows'),
        ('            for key, rows in rows_by_state.items():\n','rows')]:
        assert source.count(marker)==1,(marker,source.count(marker))
        insert="            if hasattr(model, 'prefetch'):\n                model.prefetch(torch.stack([x[v[0]] for v in rows_by_state.values()]), attention_mask)\n"
        source=source.replace(marker,insert+marker)
    experimental=ROOT/'code/experimental_prefetch_extraction.py'
    experimental.write_text(source)
    spec=importlib.util.spec_from_file_location('experimental_prefetch_extraction',experimental)
    fast=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fast;spec.loader.exec_module(fast)
    model=AutoModel.from_pretrained(ROOT/'models/llada-base',trust_remote_code=True,local_files_only=True,
          torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':0}).eval()
    model.config.use_cache=False
    data=json.loads((ROOT/'pilot_inputs.json').read_text())

    class Proxy:
        def __init__(self,positions,values):self.positions,self.values=positions,values
        def __getitem__(self,index):
            batch,positions,vocab=index
            assert batch==0 and vocab==slice(None) and positions.tolist()==self.positions
            return self.values.to('cuda:0')

    class Batched(torch.nn.Module):
        def __init__(self,base):
            super().__init__();self.base=base;self.cache={};self.rows=self.calls=self.checks=0
            self.pending=[];self.inputs=None;self.offset=0

        def prefetch(self,inputs,attention_mask):
            assert attention_mask is None
            self.cache.clear()
            cpu=inputs.detach().cpu()
            unique={}
            for i,row in enumerate(cpu):unique.setdefault(row.numpy().tobytes(),i)
            self.pending=list(unique.items());self.inputs=inputs;self.offset=0

        def load_next(self):
            # Bounded lookahead: never retain thousands of full-vocabulary states.
            self.cache.clear()
            remaining=len(self.pending)-self.offset
            count=16 if remaining>=16 else (2 if remaining>=2 else 1)
            chosen=self.pending[self.offset:self.offset+count]
            indices=torch.tensor([i for _,i in chosen],device=self.inputs.device)
            with torch.autocast('cuda',enabled=False):
                outputs=self.base(self.inputs.index_select(0,indices),attention_mask=None)
            self.calls+=1;self.rows+=count
            for local,(key,i) in enumerate(chosen):
                positions=torch.nonzero(self.inputs[i]==126336,as_tuple=False).squeeze(-1)
                values=outputs.logits[local,positions,:].detach().cpu()
                if (self.rows-count+local)%97==0:
                    with torch.autocast('cuda',enabled=False):
                        direct=self.base(self.inputs[i:i+1],attention_mask=None).logits[0,positions,:].detach().cpu()
                    assert torch.equal(values,direct),'Batched native logits changed'
                    self.checks+=1
                self.cache[key]=(positions.tolist(),values)
            del outputs
            self.offset+=count

        def forward(self,ids,attention_mask=None):
            key=ids[0].detach().cpu().numpy().tobytes()
            while key not in self.cache and self.offset<len(self.pending):self.load_next()
            if key in self.cache:
                positions,values=self.cache.pop(key)
                return SimpleNamespace(logits=Proxy(positions,values))
            self.calls+=1;self.rows+=1
            return self.base(ids,attention_mask=attention_mask)

    row=data['selected']['validation'][0]
    sequence=torch.tensor([data['tokens'][row['document']+':'+row['window_number']]],device='cuda')
    common=dict(sequence_tokens=sequence,masked_indexes=data['masks'][row['mask_id']-1],steps=50,
                attention_mask=None,mask_id=126336,seed=42)
    records=[]
    with torch.inference_mode():
        for temp in (.5,2.):
            for scheme in ('low-confidence','threshold'):
                name=('_path_sampling_low_confidence_probability_fast_from_partially_masked'
                      if scheme=='low-confidence' else '_path_sampling_threshold_probability_fast_from_partially_masked')
                extra={'confidence_threshold':.9} if scheme=='threshold' else {}
                torch.cuda.synchronize();start=time.perf_counter()
                a=getattr(reference,name)(model=model,**common,**extra,temperature=temp,num_samples=10)
                torch.cuda.synchronize();baseline=time.perf_counter()-start
                wrapped=Batched(model)
                torch.cuda.synchronize();start=time.perf_counter()
                b=getattr(fast,name)(model=wrapped,**common,**extra,temperature=temp,num_samples=10)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                equal=a['sample_log_probabilities']==b['sample_log_probabilities'] and a['log_probability']==b['log_probability']
                records.append(dict(temperature=temp,scheme=scheme,samples=10,baseline_seconds=baseline,
                    optimized_seconds=elapsed,bit_identical=equal,model_calls=wrapped.calls,model_rows=wrapped.rows,
                    direct_logit_checks=wrapped.checks))
                assert equal,'Seeded estimator results changed under state batching'
        # A small correctness-cohort probe for budget planning, not the A100 runtime study.
        start=time.perf_counter()
        original=reference._path_sampling_low_confidence_probability_fast_from_partially_masked(
                model=model,**common,temperature=2.,num_samples=25)
        torch.cuda.synchronize();baseline=time.perf_counter()-start
        wrapped=Batched(model);start=time.perf_counter()
        b=fast._path_sampling_low_confidence_probability_fast_from_partially_masked(
                model=wrapped,**common,temperature=2.,num_samples=25)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        equal=b['sample_log_probabilities']==original['sample_log_probabilities'] and b['log_probability']==original['log_probability']
        records.append(dict(temperature=2.,scheme='low-confidence',samples=25,
               baseline_seconds=baseline,optimized_seconds=elapsed,bit_identical=equal,
               model_calls=wrapped.calls,model_rows=wrapped.rows,direct_logit_checks=wrapped.checks))
        assert equal,'25-sample mean changed'
        (ROOT/'results/prefetch_path_lazy_validation.json').write_text(json.dumps(dict(validated=True,budget_only=True,records=records),indent=2))
        mc_records=[]
        for scheme in ('low-confidence','threshold'):
            name=('_monte_carlo_probability_temperature_fast_from_partially_masked'
                  if scheme=='low-confidence' else '_monte_carlo_threshold_probability_fast_from_partially_masked')
            extra={'confidence_threshold':.9} if scheme=='threshold' else {}
            start=time.perf_counter()
            a=getattr(reference,name)(model=model,**common,**extra,temperature=2.,num_samples=125,
                                      decoding_scheme='full',k=1,return_sample_logs=True)
            torch.cuda.synchronize();baseline=time.perf_counter()-start
            wrapped=Batched(model);start=time.perf_counter()
            b=getattr(fast,name)(model=wrapped,**common,**extra,temperature=2.,num_samples=125,
                                 decoding_scheme='full',k=1,return_sample_logs=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            equal=a.sample_log_probabilities==b.sample_log_probabilities and a.hits==b.hits
            mc_records.append(dict(scheme=scheme,samples=125,baseline_seconds=baseline,
                 optimized_seconds=elapsed,bit_identical=equal,model_calls=wrapped.calls,
                 model_rows=wrapped.rows,direct_logit_checks=wrapped.checks))
            assert equal,'Seeded Monte Carlo outcomes changed'
        (ROOT/'results/prefetch_mc_optimization.json').write_text(json.dumps(dict(validated=True,budget_only=True,records=mc_records),indent=2))
    (ROOT/'results/prefetch_optimization.json').write_text(json.dumps(dict(validated=True,budget_only=True,records=records,MC_validated=True),indent=2))
    print('PREFETCH_OPTIMIZATION_VALIDATED',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        (ROOT/'results/prefetch_optimization.json').write_text(json.dumps(dict(validated=False,error=str(error))))
        raise
