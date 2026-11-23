import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path('/workspace/temperature-study')
KEYS=['llama2-base','llama3-base','mistral-base','olmo-base']


def model_ready(key):
    path=ROOT/'models'/key
    index=path/'model.safetensors.index.json'
    if index.exists():
        try:return all((path/n).exists() for n in set(json.loads(index.read_text())['weight_map'].values()))
        except Exception:return False
    return (path/'model.safetensors').exists()


def main():
    start=time.time()
    while not (ROOT/'results/PILOT_COMPLETE.json').exists() or not all(model_ready(k) for k in KEYS):
        if time.time()-start>5400:raise TimeoutError('Post-pilot prerequisites timed out')
        time.sleep(2)
    time.sleep(5)
    processes=[]
    for gpu,key in enumerate(KEYS):
        log=open(ROOT/f'pilot_ar_{key}.log','w')
        p=subprocess.Popen([str(ROOT/'venv/bin/python'),'-u',str(ROOT/'code/pilot_ar.py'),
                            '--key',key,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,start_new_session=True,
                            env=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='1'))
        processes.append((key,p))
    for key,p in processes:
        print('AR_DONE',key,p.wait(timeout=600),flush=True)
    (ROOT/'results/POST_PILOTS_COMPLETE.json').write_text(json.dumps(dict(completed=time.time())))


if __name__=='__main__':main()
