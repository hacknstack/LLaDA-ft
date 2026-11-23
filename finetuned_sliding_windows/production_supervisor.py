import argparse,json,os,signal,subprocess,time,urllib.request,urllib.error
from pathlib import Path
from production_common import PYTHON,atomic,stamp,utc

CODE=Path(__file__).resolve().parent
KEYFILE=Path('/root/.config/research-runpod/api_key')

def stop_pod(run,config):
    for attempt in range(5):
        try:
            key=KEYFILE.read_text().strip() if KEYFILE.exists() else os.environ.get('RUNPOD_API_KEY','')
            request=urllib.request.Request('https://rest.runpod.io/v1/pods/'+config['pod_id']+'/stop',data=b'{}',method='POST',headers={'Authorization':'Bearer '+key,'User-Agent':'Mozilla/5.0 research-experiment/1.0','Accept':'application/json','Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=30) as response:status=response.status
            atomic(run/'shutdown_requested.json',dict(time=utc(),http_status=status));return True
        except Exception:
            try:
                p=subprocess.run(['runpodctl','stop','pod',config['pod_id']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=45,env=dict(os.environ,RUNPOD_API_KEY=key))
                if p.returncode==0:return True
            except Exception:pass
            time.sleep(10)
    atomic(run/'SHUTDOWN_FAILED.json',dict(time=utc(),reason='RunPod stop APIs rejected the available credential; manual pod stop required'))
    return False

def kill_group(child):
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait(timeout=10)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    config=json.loads((run/'config.json').read_text())
    atomic(run/'supervisor.json',dict(pid=os.getpid(),started_utc=utc(),automatic_shutdown=True))
    sync_log=(run/'sync.log').open('a');sync=subprocess.Popen([PYTHON,str(CODE/'production_sync.py'),'--run',str(run),'--loop'],stdout=sync_log,stderr=subprocess.STDOUT,start_new_session=True)
    outcome='failed';restarts=0;child=None
    try:
        while restarts<=5 and time.time()<config['compute_deadline']:
            log=(run/f'controller-{restarts:02d}.log').open('a')
            child=subprocess.Popen([PYTHON,str(CODE/'production_controller.py'),'--run',str(run)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            began=time.time();previous=-1;last_progress=began
            while child.poll() is None and time.time()<config['compute_deadline']:
                try:
                    status=json.loads((run/'status.json').read_text());done=status['total_done']
                    if done!=previous:previous=done;last_progress=time.time()
                except Exception:pass
                if time.time()-last_progress>900:kill_group(child);break
                if sync.poll() is not None:sync=subprocess.Popen([PYTHON,str(CODE/'production_sync.py'),'--run',str(run),'--loop'],stdout=sync_log,stderr=subprocess.STDOUT,start_new_session=True)
                time.sleep(10)
            if child.poll() is None:kill_group(child)
            code=child.returncode;kill_group(child);log.close()
            if code in [0,20]:outcome='complete' if code==0 else 'partial';break
            restarts+=1
            atomic(run/'restart_status.json',dict(time=utc(),restarts=restarts,last_exitcode=code))
            time.sleep(20)
        atomic(run/'execution_finished.json',dict(time=utc(),outcome=outcome,controller_restarts=restarts))
        status=json.loads((run/'status.json').read_text());status.update(status=outcome,updated_utc=utc());atomic(run/'status.json',status)
        (run/'SYNC_STOP').touch();kill_group(sync)
        for attempt in range(5):
            remaining=config['hard_deadline']-time.time()-120
            if remaining<=0:break
            try:
                result=subprocess.run([PYTHON,str(CODE/'production_sync.py'),'--run',str(run),'--final'],stdout=sync_log,stderr=subprocess.STDOUT,timeout=min(900,remaining))
                if result.returncode==0:break
            except subprocess.TimeoutExpired:pass
            time.sleep(30)
    finally:
        if child and child.poll() is None:kill_group(child)
        if sync.poll() is None:kill_group(sync)
        # A detached restart wrapper can die without leaving GPUs billing indefinitely.
        atomic(run/'shutdown_status.json',dict(time=utc(),attempting_stop=True,final_upload_verified=(run/'final_upload_verified.json').exists()))
        if outcome!='failed' or (run/'AUTOMATION_ARMED').exists() or time.time()>=config['hard_deadline']:
            stop_pod(run,config)
        else:
            atomic(run/'NEEDS_OPERATOR.json',dict(time=utc(),reason='Failure during the initial monitored launch; automatic recovery exhausted'))

if __name__=='__main__':main()
