"""Confirm chosen settings concurrently, including vectorized CPU FP32 AdamW."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/workspace/bbc-exposure')
started=time.time()
while not (ROOT/'audits/microbatch_sweep_complete.json').exists():
    if time.time()-started>5400:raise RuntimeError('Microbatch sweep did not complete')
    time.sleep(3)
selected={}
for key in ['llada8b','llama3_8b','mistral7b_v01']:
    candidates=[]
    for p in (ROOT/'audits').glob('probe_async_mb*_'+key+'.json'):
        r=json.loads(p.read_text())
        if r.get('status')=='passed' and r['peak_gpu_reserved_gib']<75.5:candidates.append(r)
    assert candidates,key
    fastest=min(r['warm_full_update_seconds'] for r in candidates)
    # Prefer the smaller allocation when within 5% of the best measured time.
    chosen=min((r for r in candidates if r['warm_full_update_seconds']<=fastest*1.05),key=lambda r:r['physical_microbatch'])
    selected[key]=chosen['physical_microbatch']
jobs=[]
for gpu,(key,b) in enumerate(selected.items()):
    log=(ROOT/'audits'/('probe_final_'+key+'.log')).open('w')
    cmd=[str(ROOT/'venv/bin/python'),'-u',str(ROOT/'code/model_probe.py'),'--key',key,'--gpu',str(gpu),
         '--tag','final','--updates','3','--tail','--microbatch',str(b),'--async-offload','--foreach']
    if key=='llada8b':cmd.append('--worst-mask')
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    jobs.append((key,p,log))
    print('FINAL_CONCURRENCY_STARTED',key,b,gpu,p.pid,flush=True)
results={}
for key,p,log in jobs:
    p.wait(timeout=1800);log.close();results[key]=p.returncode
(ROOT/'audits/final_concurrency_complete.json').write_text(json.dumps(results,indent=2))
assert all(v==0 for v in results.values()),results
print('FINAL_CONCURRENCY_PASSED',flush=True)
