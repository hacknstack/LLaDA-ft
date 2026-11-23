"""Keep three probe GPUs occupied as pinned model inputs become ready."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/workspace/bbc-exposure')
MODELS=['llada8b','llama3_8b','mistral7b_v01','llama2_7b','olmo7b']


def main():
    pending=MODELS.copy()
    active={}
    outcomes={}
    started=time.time()
    while pending or active:
        for gpu in range(3):
            if gpu in active:continue
            ready=[k for k in pending if (ROOT/'inputs'/k/'training_config.json').exists()]
            if not ready:continue
            key=ready[0];pending.remove(key)
            log=(ROOT/'audits'/('probe_initial_'+key+'.log')).open('w')
            cmd=[str(ROOT/'venv/bin/python'),'-u',str(ROOT/'code/model_probe.py'),'--key',key,'--gpu',str(gpu),
                 '--tag','initial','--updates','3','--tail','--sweep']
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            active[gpu]=dict(process=process,key=key,log=log,started=time.time())
            print('PROBE_STARTED',key,gpu,process.pid,flush=True)
        for gpu,item in list(active.items()):
            p=item['process']
            if p.poll() is None and time.time()-item['started']>1800:
                p.terminate();p.wait(timeout=30)
            if p.poll() is not None:
                item['log'].close()
                outcomes[item['key']]=dict(exit_code=p.returncode,gpu=gpu,elapsed=time.time()-item['started'])
                print('PROBE_FINISHED',item['key'],p.returncode,flush=True)
                del active[gpu]
        if time.time()-started>3600:
            for item in active.values():item['process'].terminate()
            raise RuntimeError('Bounded staging/probe deadline reached')
        (ROOT/'audits/benchmark_progress.json').write_text(json.dumps(dict(pending=pending,active={g:v['key'] for g,v in active.items()},finished=outcomes),indent=2))
        time.sleep(2)
    (ROOT/'audits/initial_benchmarks_complete.json').write_text(json.dumps(outcomes,indent=2))
    print('INITIAL_BENCHMARKS_COMPLETE',flush=True)


if __name__=='__main__':main()
