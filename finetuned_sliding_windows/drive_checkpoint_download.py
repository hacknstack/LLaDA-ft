"""Bounded Drive range downloads. OAuth stays in the pod's private rclone file."""
import configparser
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import ssl
import threading
import time
import urllib.error
import urllib.request

POOL=ThreadPoolExecutor(max_workers=24)
SSL_CONTEXT=ssl.create_default_context()
CONFIG='/root/.config/research-rclone/rclone.conf'
CHUNK=64*1024*1024
RATE_LOCK=threading.Lock()
NEXT_REQUEST=0.0
COOLDOWN=0.0


def pace():
    global NEXT_REQUEST
    with RATE_LOCK:
        now=time.monotonic();when=max(now,NEXT_REQUEST,COOLDOWN)
        NEXT_REQUEST=when+0.5
    time.sleep(max(0,when-time.monotonic()))


def read_range(file_id,start,end,descriptor):
    global COOLDOWN
    for attempt in range(30):
        try:
            pace()
            config=configparser.ConfigParser();config.read(CONFIG)
            token=json.loads(config['researchdrive']['token'])['access_token']
            request=urllib.request.Request('https://www.googleapis.com/drive/v3/files/'+file_id+'?alt=media',
                headers={'Authorization':'Bearer '+token,'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request,timeout=60,context=SSL_CONTEXT) as response:
                if response.status!=206:raise RuntimeError('Expected a partial Drive response')
                if not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    raise RuntimeError('Drive returned a different byte range')
                data=response.read(end-start+2)
            if len(data)!=end-start+1:raise RuntimeError('Incomplete Drive range')
            offset=0
            while offset<len(data):offset+=os.pwrite(descriptor,data[offset:],start+offset)
            return len(data)
        except Exception as error:
            if isinstance(error,urllib.error.HTTPError) and error.code in {403,429}:
                with RATE_LOCK:COOLDOWN=max(COOLDOWN,time.monotonic()+35)
            if attempt==29:raise RuntimeError(f'Drive range failed after retries at byte {start}') from None
            time.sleep(min(2**attempt,30))


def download_folder(remote,folder,remote_call):
    files=json.loads(remote_call(['lsjson',remote,'--files-only']))
    for item in files:
        name=item['Name'];assert Path(name).name==name
        path=folder/name
        if path.exists() and path.stat().st_size==item['Size']:continue
        temporary=path.with_suffix(path.suffix+'.ranges')
        descriptor=os.open(temporary,os.O_CREAT|os.O_RDWR,0o600)
        try:
            if os.fstat(descriptor).st_size!=item['Size']:os.ftruncate(descriptor,0)
            os.ftruncate(descriptor,item['Size'])
            def already_written(start,end):
                try:
                    return (os.lseek(descriptor,start,os.SEEK_DATA)==start
                            and os.lseek(descriptor,start,os.SEEK_HOLE)>=end+1)
                except OSError:return False
            futures=[POOL.submit(read_range,item['ID'],start,min(start+CHUNK,item['Size'])-1,descriptor)
                     for start in range(0,item['Size'],CHUNK)
                     if not already_written(start,min(start+CHUNK,item['Size'])-1)]
            for future in as_completed(futures):future.result()
            os.fsync(descriptor)
        finally:
            # Wait for outstanding writes before closing a shared descriptor.
            for future in locals().get('futures',[]):
                try:future.result()
                except Exception:pass
            os.close(descriptor)
        temporary.replace(path)
        print('DRIVE_CHECKPOINT_FILE_DOWNLOADED',folder.name,name,item['Size'],flush=True)
