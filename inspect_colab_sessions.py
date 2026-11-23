"""Read active notebook identities without printing runtime credentials."""
import json
import os
from pathlib import Path
import sys
import requests

base=Path(os.environ['TEMP'])/'codex-colab-cli-source'
sys.path.insert(0,str(base/'google-colab-cli-main/src'))
from colab_cli.common import state
from colab_cli.state import SessionState
state.config_path=str(base/'active-session.json')
for assignment in state.client.list_assignments():
    if len(sys.argv)==3 and sys.argv[1]=='--attach-endpoint':
        if assignment.endpoint==sys.argv[2]:
            state.store.add(SessionState(name='current-a100',token=assignment.runtime_proxy_info.token,
                url=assignment.runtime_proxy_info.url,endpoint=assignment.endpoint,
                accelerator=assignment.accelerator.value,machine_shape=assignment.machine_shape.name,variant='GPU'))
            print('ATTACHED_IDENTIFIED_RUNTIME',assignment.endpoint)
            break
        continue
    info=assignment.runtime_proxy_info
    row=dict(endpoint=assignment.endpoint,accelerator=assignment.accelerator.value,
             machine_shape=assignment.machine_shape.name)
    try:
        response=requests.get(info.url.rstrip('/')+'/api/sessions',
            headers={'X-Colab-Runtime-Proxy-Token':info.token},timeout=20)
        row['http_status']=response.status_code
        if response.ok:
            row['notebooks']=[dict(path=s.get('path'),name=s.get('name'),
                kernel_id=s.get('kernel',{}).get('id'),
                execution_state=s.get('kernel',{}).get('execution_state'),
                last_activity=s.get('kernel',{}).get('last_activity')) for s in response.json()]
    except Exception as error:
        row['error_type']=type(error).__name__
    print(json.dumps(row))
