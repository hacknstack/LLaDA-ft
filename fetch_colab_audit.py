"""Fetch the small read-only audit result; runtime credentials remain in memory."""
import json
import os
from pathlib import Path
import sys
import requests

base=Path(os.environ['TEMP'])/'codex-colab-cli-source'
sys.path.insert(0,str(base/'google-colab-cli-main/src'))
from colab_cli.common import state
state.config_path=str(base/'active-session.json')
assignments=state.client.list_assignments()
assert len(assignments)==1, 'Identify the intended runtime before fetching.'
info=assignments[0].runtime_proxy_info
response=requests.get(info.url.rstrip('/')+'/api/contents/content/llada_drive_audit.json',
    headers={'X-Colab-Runtime-Proxy-Token':info.token}, params={'type':'file','format':'text'}, timeout=45)
assert response.status_code==200, f'Audit fetch failed: HTTP {response.status_code}'
data=json.loads(response.json()['content'])
path=Path(__file__).resolve().parent/'drive_training_audit_20261026.json'
path.write_text(json.dumps(data,indent=2),encoding='utf-8')
print('AUDIT_SAVED',path.name)
for run in data['runs']:
    print(json.dumps(dict(run=run['run'],successful=run.get('successful'),
        diagnostics=[dict(epoch=d['epoch'],means=d['means']) for d in run.get('diagnostics',[])],
        final_checkpoint=run['checkpoints'][-1] if run['checkpoints'] else None)))
