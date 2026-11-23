"""Transfer renewable Drive auth over SSH without exposing tokens to stdout."""
import base64
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import uuid

private=Path(os.environ['USERPROFILE'])/'.codex/rclone-tools/research-drive-auth'
assert json.loads((private/'authorization_status.json').read_text())['status']=='authorized'
payload=base64.b64encode((private/'rclone.conf').read_bytes()).decode()
tag=uuid.uuid4().hex
ready='AUTH_READY_'+tag;done='AUTH_INSTALLED_'+tag;end='AUTH_END_'+tag
receiver=f'''import os,sys,base64
from pathlib import Path
os.umask(0o077)
root=Path('/root/.config/research-rclone')
root.mkdir(parents=True,exist_ok=True,mode=0o700)
root.chmod(0o700)
print({ready!r},flush=True)
parts=[]
for line in sys.stdin:
 if line.strip()=={end!r}:break
 parts.append(line.strip())
path=root/'rclone.conf'
temporary=root/'rclone.conf.new'
temporary.write_bytes(base64.b64decode(''.join(parts)))
temporary.chmod(0o600)
temporary.replace(path)
print({done!r},flush=True)
'''
args=['ssh','-tt','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20',
      '-i',str(Path(os.environ['USERPROFILE'])/'.ssh/id_ed25519'),
      'ynbsa87gmjxhoq-644117d1@ssh.runpod.io']
process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
messages=queue.Queue()
def reader():
    while True:
        piece=process.stdout.read(1)
        if not piece:messages.put(None);return
        messages.put(piece)
threading.Thread(target=reader,daemon=True).start()
def await_marker(marker):
    tail=b''
    while marker.encode() not in tail:
        piece=messages.get(timeout=60)
        assert piece is not None,'SSH closed before authorization transfer finished; output withheld'
        tail=(tail+piece)[-8192:]
encoded_receiver=base64.b64encode(receiver.encode()).decode()
entry="import base64;exec(base64.b64decode("+repr(encoded_receiver)+"))"
process.stdin.write(('stty -echo\npython -c '+shlex.quote(entry)+'\n').encode());process.stdin.flush()
await_marker(ready)
for offset in range(0,len(payload),120):
    process.stdin.write((payload[offset:offset+120]+'\n').encode())
process.stdin.write((end+'\n').encode());process.stdin.flush()
await_marker(done)
process.stdin.write(b'stty echo\nexit\n');process.stdin.flush()
try:
    process.wait(timeout=10)
except subprocess.TimeoutExpired:
    # The proxy may keep its interactive shell open after the confirmed write.
    # Close only this transfer connection; the pod and other SSH sessions remain.
    process.terminate()
    process.wait(timeout=5)
print('PRIVATE_DRIVE_CREDENTIALS_INSTALLED_ON_RUNPOD')
