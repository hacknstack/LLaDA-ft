"""Install a user-provided local RunPod key through an opaque SSH transfer."""
import base64,os,subprocess
from pathlib import Path

source=Path.home()/'Downloads/runpod-api-key.txt'
key=source.read_text(encoding='utf-8-sig').strip();assert len(key)>=20 and not any(c.isspace() for c in key)
folder=Path.home()/'.codex/research-runpod';folder.mkdir(exist_ok=True)
sid='*'+subprocess.check_output(['powershell','-NoProfile','-Command','[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value'],text=True).strip()
subprocess.run(['icacls',str(folder),'/inheritance:r','/grant:r',sid+':(OI)(CI)F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
(folder/'api_key').write_text(key,encoding='utf-8')
script=(Path(__file__).parent/'install_runpod_drive_credentials.py').read_text()
start=script.index('private=');end=script.index('\ntag=',start)
script=script[:start]+"payload=base64.b64encode((Path.home()/'.codex/research-runpod/api_key').read_bytes()).decode()\n"+script[end:]
script=script.replace('/root/.config/research-rclone','/root/.config/research-runpod').replace("root/'rclone.conf'","root/'api_key'").replace("root/'rclone.conf.new'","root/'api_key.new'")
script=script.replace('PRIVATE_DRIVE_CREDENTIALS_INSTALLED_ON_RUNPOD','PRIVATE_RUNPOD_SHUTDOWN_KEY_INSTALLED')
exec(compile(script,'opaque_runpod_key_transfer','exec'))
