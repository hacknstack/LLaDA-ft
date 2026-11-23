"""Local browser OAuth. Credentials stay in a private folder, never in stdout."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('--client-json',type=Path)
args=parser.parse_args()
client=None
if args.client_json:
    client=json.loads(args.client_json.read_text(encoding='utf-8'))['installed']
    assert client.get('client_id') and client.get('client_secret')

root=Path(os.environ['USERPROFILE'])/'.codex/rclone-tools'
install=json.loads((root/'install.json').read_text())
private=root/'research-drive-auth'
private.mkdir(exist_ok=True)
identity='*'+subprocess.check_output(['powershell','-NoProfile','-Command',
    '[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value'],text=True).strip()
subprocess.run(['icacls',str(private),'/inheritance:r','/grant:r',identity+':(OI)(CI)F'],
               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
authorize=[install['executable'],'authorize','drive']
if client:authorize.extend([client['client_id'],client['client_secret']])
with (private/'authorize_output.txt').open('w',encoding='utf-8') as output:
    process=subprocess.Popen(authorize,stdout=output,stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW)
(private/'authorization_status.json').write_text(json.dumps(dict(status='awaiting_browser',pid=process.pid)))
print('GOOGLE_DRIVE_BROWSER_APPROVAL_STARTED',flush=True)
deadline=time.monotonic()+1200
while process.poll() is None and time.monotonic()<deadline:
    time.sleep(2)
if process.poll() is None:
    (private/'authorization_status.json').write_text(json.dumps(dict(status='timed_out',pid=process.pid)))
    print('DRIVE_BROWSER_APPROVAL_TIMED_OUT',flush=True)
    raise SystemExit(1)
if process.returncode:
    (private/'authorization_status.json').write_text(json.dumps(dict(status='failed',exit_code=process.returncode)))
    print('DRIVE_BROWSER_APPROVAL_FAILED',process.returncode,flush=True)
    raise SystemExit(1)
content=(private/'authorize_output.txt').read_text(encoding='utf-8')
token=None
decoder=json.JSONDecoder()
for i,char in enumerate(content):
    if char!='{':continue
    try:value,_=decoder.raw_decode(content[i:])
    except ValueError:continue
    if isinstance(value,dict) and value.get('access_token'):
        token=value;break
assert token and token.get('refresh_token'),'OAuth did not return renewable credentials; output withheld'
configuration='[researchdrive]\ntype = drive\nscope = drive\n'
if client:configuration+='client_id = '+client['client_id']+'\nclient_secret = '+client['client_secret']+'\n'
configuration+='token = '+json.dumps(token,separators=(',',':'))+'\n'
temporary=private/'rclone.conf.new'
temporary.write_text(configuration,encoding='utf-8');temporary.replace(private/'rclone.conf')
(private/'authorization_status.json').write_text(json.dumps(dict(status='authorized',renewable=True)))
print('DRIVE_BROWSER_APPROVAL_COMPLETE_CREDENTIALS_SAVED_PRIVATELY',flush=True)
