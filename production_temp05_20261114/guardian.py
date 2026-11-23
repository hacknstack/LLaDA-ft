"""Independent Windows fallback: retrieve results and confirm the authorized pod stops."""
import configparser
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.parse
import urllib.request
from zipfile import ZipFile, ZIP_DEFLATED

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'config.json').read_text())
RUN=CONFIG['run_id']
REMOTE='/workspace/temperature-study/runs/'+RUN
DEST=Path.home()/'Downloads'/RUN
DEST.mkdir(exist_ok=True)
KEY=Path.home()/'.ssh/id_ed25519'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','18231','-i',str(KEY),'root@64.247.201.47']
SCP=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','18231','-i',str(KEY)]


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'),message,flush=True)


def api(action=None):
    key=(Path.home()/'Downloads/runpod-api-key.txt').read_text(encoding='utf-8-sig').strip()
    url='https://api.runpod.io/v2/pods/'+CONFIG['pod_id']+('/action' if action else '')
    req=urllib.request.Request(url,data=None if action is None else json.dumps({'action':action}).encode(),
          headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','User-Agent':'research-experiment/1.0'},method='GET' if action is None else 'POST')
    with urllib.request.urlopen(req,timeout=40) as r:
        raw=r.read()
        return json.loads(raw) if raw else {}


def remote_json(name):
    p=subprocess.run(SSH+['cat '+shlex.quote(REMOTE+'/'+name)],capture_output=True,timeout=40)
    return json.loads(p.stdout) if p.returncode==0 else None


def backup_local():
    # Only the small run directory: source/result text and journals, never model files.
    log('Securing local fallback copy')
    p=subprocess.run(SCP+['-r','root@64.247.201.47:'+REMOTE,str(DEST/'pod_backup')],capture_output=True,timeout=600)
    if p.returncode:
        log('Local SCP fallback unavailable')
        return False
    log('Local fallback copy completed')
    return True


def drive_token():
    c=configparser.ConfigParser()
    c.read(Path.home()/'.codex/rclone-tools/research-drive-auth/rclone.conf')
    s=c['researchdrive']
    body=urllib.parse.urlencode(dict(client_id=s['client_id'],client_secret=s['client_secret'],
        refresh_token=json.loads(s['token'])['refresh_token'],grant_type='refresh_token')).encode()
    with urllib.request.urlopen(urllib.request.Request('https://oauth2.googleapis.com/token',data=body),timeout=40) as r:
        return json.load(r)['access_token']


def fetch_drive(manifest):
    token=drive_token()
    receipt={}
    for name,meta in manifest['files'].items():
        if name=='config.json':
            continue
        file_id=meta['id']
        headers={'Authorization':'Bearer '+token}
        url='https://www.googleapis.com/drive/v3/files/'+file_id
        with urllib.request.urlopen(urllib.request.Request(url+'?fields=id,name,size,md5Checksum,webViewLink',headers=headers),timeout=60) as r:
            check=json.load(r)
        path=DEST/name
        with urllib.request.urlopen(urllib.request.Request(url+'?alt=media',headers=headers),timeout=120) as r,path.with_suffix(path.suffix+'.new').open('wb') as f:
            while chunk:=r.read(1024**2):
                f.write(chunk)
        tmp=path.with_suffix(path.suffix+'.new')
        assert hashlib.md5(tmp.read_bytes()).hexdigest()==check['md5Checksum'] and tmp.stat().st_size==int(check['size'])
        tmp.replace(path)
        receipt[name]=check
    (DEST/'drive_download_receipt.json').write_text(json.dumps(receipt,indent=2))
    return receipt


def main():
    manifest=json.loads((HERE/'drive_manifest.json').read_text())
    last_status=None
    last_seen=time.time()
    last_backup=0
    stop_requested=0
    while True:
        try:
            state=api()
            status=state.get('status')
            if status!='RUNNING':
                if status in ('EXITED','STOPPED','TERMINATED'):
                    log('Pod stopped: '+status)
                    (DEST/'pod_stopped.json').write_text(json.dumps(dict(pod_id=CONFIG['pod_id'],status=status,checked_unix=time.time()),indent=2))
                    try:
                        fetch_drive(manifest)
                        report=json.loads((DEST/'validation.json').read_text())
                        scientific_status=json.loads((DEST/'status.json').read_text())
                        complete=report['final'] and report['completed']==14400 and scientific_status['outcome']=='complete'
                        log('Final Drive retrieval complete; full experiment complete='+str(complete))
                    except Exception as e:
                        log('Drive retrieval needs retry: '+type(e).__name__)
                        time.sleep(30)
                        continue
                    with ZipFile(Path.home()/'Downloads'/(RUN+'.zip'),'w',ZIP_DEFLATED) as z:
                        for p in DEST.glob('*'):
                            if p.is_file():z.write(p,p.name)
                    return
                log('Pod status: '+str(status))
            progress=remote_json('progress.json')
            if progress:
                last_status=progress
                last_seen=progress.get('updated_unix',time.time())
                (DEST/'progress.json').write_text(json.dumps(progress,indent=2))
                log('Progress '+str(progress.get('completed'))+'/14400; state='+str(progress.get('state')))
            if time.time()-last_backup>600:
                current=remote_json('drive_manifest.json')
                if current:
                    manifest=current
                    (HERE/'drive_manifest.json').write_text(json.dumps(current,indent=2))
                    (DEST/'drive_manifest.json').write_text(json.dumps(current,indent=2))
                last_backup=time.time()
            terminal=last_status and not last_status.get('smoke') and last_status.get('state') in ('complete','failed')
            stale=time.time()-last_seen>CONFIG['stall_timeout_seconds']+120
            if terminal or stale:
                # If remote save failed, secure raw journals before stopping.
                durable=bool(last_status and last_status.get('drive_saved'))
                if not durable:
                    durable=backup_local()
                    if not durable:
                        try:
                            fetch_drive(manifest)
                            durable=True
                        except Exception:
                            pass
                if durable:
                    api('stop')
                    stop_requested=time.time()
                    log('Fallback stop requested; waiting for confirmation')
                else:
                    log('Cannot stop safely yet: retrying result backup')
            time.sleep(45)
        except Exception as e:
            # Do not print URL/header/request bodies, which can contain auth material.
            log('Guardian retry: '+type(e).__name__)
            time.sleep(30)


if __name__=='__main__':
    main()
