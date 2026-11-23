"""Explicit opt-in launcher; compute workers and CPU-only uploads are separate."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/workspace/bbc-exposure')


def main(execute):
    plan=json.loads((ROOT/'protocol/allocation_plan.json').read_text())
    ready=json.loads((ROOT/'protocol/readiness.json').read_text())
    if not execute:
        print(json.dumps(dict(ready=ready,allocation=plan,status='NO_TRAINING_STARTED'),indent=2));return
    assert ready['ready_to_launch'],'Readiness checks have not all passed'
    run=ROOT/'runs'/ready['run_name'];run.mkdir(parents=True,exist_ok=False)
    queues={int(g):list(keys) for g,keys in plan['gpu_queues'].items()}
    active={};uploads={};finished=[]
    mirror_log=(run/'status_mirror.log').open('w')
    mirror=subprocess.Popen([str(ROOT/'venv/bin/python'),'-u',str(ROOT/'code/mirror_status.py')],
              stdout=mirror_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    try:
        while any(queues.values()) or active or uploads:
            for gpu,keys in queues.items():
                if gpu in active or not keys:continue
                key=keys.pop(0)
                log=(run/(key+'_training.log')).open('w')
                cmd=[str(ROOT/'venv/bin/python'),'-u',str(ROOT/'code/train_exposure.py'),'--key',key,'--gpu',str(gpu),'--execute']
                p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                active[gpu]=dict(key=key,p=p,log=log)
                print('TRAINING_WORKER_STARTED',key,gpu,p.pid,flush=True)
            for gpu,item in list(active.items()):
                code=item['p'].poll()
                if code is None:continue
                item['log'].close()
                assert code==0,'Training worker failed: '+item['key']
                key=item['key'];del active[gpu]
                log=(run/(key+'_upload.log')).open('w')
                p=subprocess.Popen([str(ROOT/'venv/bin/python'),'-u',str(ROOT/'code/upload_final.py'),'--key',key],
                         stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                uploads[key]=dict(p=p,log=log)
                print('GPU_RELEASED_UPLOAD_OVERLAPPED',key,gpu,flush=True)
            for key,item in list(uploads.items()):
                code=item['p'].poll()
                if code is None:continue
                item['log'].close();assert code==0,'Final upload failed: '+key
                finished.append(key);del uploads[key]
            state=dict(state='running',active_training={g:i['key'] for g,i in active.items()},uploading=list(uploads),
                 queued=queues,finished=finished,updated_unix=time.time())
            (run/'suite_status.json').write_text(json.dumps(state,indent=2))
            time.sleep(3)
        (run/'suite_status.json').write_text(json.dumps(dict(state='complete',finished=finished,updated_unix=time.time()),indent=2))
        (run/'SUITE_COMPLETE.json').write_text(json.dumps(dict(all_models_complete=True,models=finished,verified_final_models_on_drive=True),indent=2))
        print('ALL_FIVE_MODELS_FINISHED_AND_VERIFIED',flush=True)
    except BaseException:
        for item in active.values():item['p'].terminate()
        (run/'suite_status.json').write_text(json.dumps(dict(state='failed',finished=finished,updated_unix=time.time()),indent=2))
        raise
    finally:
        (run/'MIRROR_STOP').touch()
        try:mirror.wait(timeout=120)
        except subprocess.TimeoutExpired:mirror.terminate()
        mirror_log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    main(args.execute)
