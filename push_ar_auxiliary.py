"""Stage small authorized probe code into the already identified Colab runtime."""
import os
from pathlib import Path
import sys
import requests
base=Path(os.environ['TEMP'])/'codex-colab-cli-source'
sys.path.insert(0,str(base/'google-colab-cli-main/src'))
from colab_cli.common import state
state.config_path=str(base/'active-session.json')
session=state.store.get('current-a100')
assert session
for name in sys.argv[1:]:
    path=Path(name)
    assert path.name==name and path.stat().st_size<1024*1024
    response=requests.put(session.url.rstrip('/')+'/api/contents/content/'+name,
        headers={'X-Colab-Runtime-Proxy-Token':session.token},
        json=dict(type='file',format='text',content=path.read_text(encoding='utf-8')),timeout=30)
    assert response.status_code in (200,201), f'Stage failed: HTTP {response.status_code}'
    print('STAGED_SMALL_FILE',name)
