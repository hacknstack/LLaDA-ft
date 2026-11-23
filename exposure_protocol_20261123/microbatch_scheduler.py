"""Bounded allocation sweep, preserving full fine-tuning and 16 sequences/update."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/workspace/bbc-exposure')
KEYS=['llada8b','llama3_8b','mistral7b_v01','llama2_7b','olmo7b']


def main():
    started=time.time()
    progress=ROOT/'audits/benchmark_progress.json'
    pending=[(k,1) for k in KEYS]
    active={}
    history=[]
    while pending or active:
        try:initial=json.loads(progress.read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            time.sleep(2);continue
        # Begin on freed GPUs once all original probes have been dispatched.
        initial_busy={int(g) for g in initial['active']}
        eligible={k for k,v in initial['finished'].items() if v['exit_code']==0}
        failed={k for k,v in initial['finished'].items() if v['exit_code']!=0}
        pending=[(k,b) for k,b in pending if k not in failed]
        for gpu in range(3):
            if gpu in active or gpu in initial_busy or initial['pending'] or not pending:continue
            ready=[x for x in pending if x[0] in eligible]
            if not ready:continue
            key,b=ready[0];pending.remove((key,b))
            tag='async_mb'+str(b)
            log=(ROOT/'audits'/('probe_'+tag+'_'+key+'.log')).open('w')
            cmd=[str(ROOT/'venv/bin/python'),'-u',str(ROOT/'code/model_probe.py'),'--key',key,'--gpu',str(gpu),
                 '--tag',tag,'--updates','2','--tail','--microbatch',str(b),'--async-offload']
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            active[gpu]=dict(key=key,b=b,p=p,log=log,start=time.time())
            print('MICROBATCH_STARTED',key,b,gpu,p.pid,flush=True)
        for gpu,item in list(active.items()):
            p=item['p']
            if p.poll() is None and time.time()-item['start']>1800:
                p.terminate();p.wait(timeout=30)
            if p.poll() is not None:
                item['log'].close()
                row=dict(model=item['key'],microbatch=item['b'],exit_code=p.returncode,gpu=gpu)
                history.append(row)
                print('MICROBATCH_FINISHED',json.dumps(row),flush=True)
                if p.returncode==0 and item['b']<8:pending.append((item['key'],item['b']*2))
                del active[gpu]
        (ROOT/'audits/microbatch_progress.json').write_text(json.dumps(dict(pending=pending,active={g:dict(key=v['key'],microbatch=v['b']) for g,v in active.items()},history=history),indent=2))
        if time.time()-started>5400:
            for item in active.values():item['p'].terminate()
            raise RuntimeError('Allocation sweep deadline reached')
        time.sleep(2)
    (ROOT/'audits/microbatch_sweep_complete.json').write_text(json.dumps(history,indent=2))
    print('MICROBATCH_SWEEP_COMPLETE',flush=True)


if __name__=='__main__':main()
