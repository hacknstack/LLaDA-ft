import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
import json,math,sys,time,traceback
from pathlib import Path
from production_common import connect,sha

def work(run,gpu,key,generation,inbox,outbox):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    task=None
    try:
        import torch
        import sliding_window_extraction as ex
        config=json.loads((Path(run)/'config.json').read_text())
        assert sha(Path(__file__).parent/'probabilistic_extraction.py')==config['estimator_sha256']
        assert sha(Path(__file__).parent/'sliding_window_extraction.py')==config['wrapper_sha256']
        spec=next(s for s in config['models'] if s['key']==key)
        assert sha(Path(spec['local_path'])/'COMPLETE.json')==spec['complete_sha256']
        torch.set_num_threads(4)
        model=ex._load_model(spec['local_path'],spec['family'],'cuda:0');model.config.use_cache=False
        assert sum(p.numel() for p in model.parameters())==spec['parameters']
        assert all(p.is_cuda and p.dtype==torch.bfloat16 for p in model.parameters())
        sys.argv=['sliding_window_extraction.py','unused.txt','--model-name',spec['local_path'],'--model-family',spec['family'],
                  '--mode','path_sampling' if key=='llada' else 'exact','--decoding-scheme','full','--temperature','1','--num-samples','10']
        args=ex.parse_args();db=connect(run)
        outbox.put(dict(event='ready',gpu=gpu,model=key,generation=generation))
        while True:
            task=inbox.get()
            if task is None:break
            ids=json.loads(db.execute('SELECT token_ids FROM inputs WHERE input_id=? AND model=?',(task['input_id'],key)).fetchone()[0]);assert len(ids)==100
            args.seed=task['seed'];args.remasking=task['scheme'] if key=='llada' else 'low-confidence'
            args.masked_indexes=config['mask_definitions'][task['mask_id']-1][1] if key=='llada' else None
            args.confidence_threshold=.9
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
            probability,_=ex._compute_probability(model,ids[:50],ids[50:],args)
            torch.cuda.synchronize();seconds=time.perf_counter()-started;logp=args._last_log_probability
            assert math.isfinite(probability) and 0<=probability<=1+1e-10
            assert logp is not None and not math.isnan(logp) and logp<=1e-10
            assert math.isclose(probability,math.exp(logp),rel_tol=1e-8,abs_tol=5e-324)
            outbox.put(dict(event='result',gpu=gpu,model=key,generation=generation,task_id=task['task_id'],
                result=dict(probability=probability,log_probability=logp,seconds=seconds,
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)))
        outbox.put(dict(event='finished',gpu=gpu,model=key,generation=generation))
    except BaseException:
        outbox.put(dict(event='error',gpu=gpu,model=key,generation=generation,
            task_id=task['task_id'] if task else None,error=traceback.format_exc()))
        raise
