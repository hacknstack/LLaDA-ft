"""Independent budget deadline, also supervising the detached controller wrapper."""
import argparse,json,os,subprocess,time
from pathlib import Path
from production_common import PYTHON,atomic,utc
from production_supervisor import stop_pod,kill_group

def alive(pid):
    try:os.kill(int(pid),0);return True
    except (ProcessLookupError,ValueError,TypeError):return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    config=json.loads((run/'config.json').read_text());restarts=0
    while True:
        now=time.time()
        atomic(run/'guard_status.json',dict(updated_utc=utc(),pid=os.getpid(),supervisor_restarts=restarts,hard_deadline_epoch=config['hard_deadline']))
        if now>=config['hard_deadline']:
            (run/'REQUEST_STOP_NOW').touch()
            if stop_pod(run,config):return
            time.sleep(30);continue
        if now>=config['compute_deadline']:(run/'REQUEST_STOP_NOW').touch()
        if (run/'AUTOMATION_ARMED').exists() and not (run/'shutdown_requested.json').exists():
            try:supervisor=json.loads((run/'supervisor.json').read_text())
            except Exception:supervisor={}
            if not alive(supervisor.get('pid')):
                # Remove only this run's orphaned controller/uploader process groups.
                groups=[]
                for entry in Path('/proc').iterdir():
                    if not entry.name.isdigit():continue
                    try:
                        cmd=(entry/'cmdline').read_bytes();pid=int(entry.name)
                        if str(run).encode() in cmd and any(s in cmd for s in [b'production_controller.py',b'production_sync.py']) and os.getpgid(pid)==pid:
                            os.killpg(pid,15);groups.append(pid)
                    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
                time.sleep(5)
                for pid in groups:
                    try:
                        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
                        if str(run).encode() in cmd:os.killpg(pid,9)
                    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
                if (run/'execution_finished.json').exists():
                    if not (run/'final_upload_verified.json').exists():
                        log=(run/'guard_final_upload.log').open('a')
                        try:subprocess.run([PYTHON,str(Path(__file__).parent/'production_sync.py'),'--run',str(run),'--final'],stdout=log,stderr=subprocess.STDOUT,timeout=max(10,min(900,config['hard_deadline']-time.time()-120)))
                        except subprocess.TimeoutExpired:pass
                    if stop_pod(run,config):return
                    time.sleep(30);continue
                if restarts>=3:
                    (run/'REQUEST_STOP_NOW').touch();stop_pod(run,config);return
                log=(run/'supervisor.log').open('a')
                child=subprocess.Popen([PYTHON,str(Path(__file__).parent/'production_supervisor.py'),'--run',str(run)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                restarts+=1;time.sleep(30)
        time.sleep(20)

if __name__=='__main__':main()
