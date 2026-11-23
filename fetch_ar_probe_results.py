"""Read small probe reports from the identified Colab runtime."""
import json
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
output=Path(__file__).resolve().parent/'ar_probe_results'
output.mkdir(exist_ok=True)
names=sys.argv[1:] or ['mistral7b_v01_4096','olmo7b_4096','llama2_7b_4096','llama3_8b_4096']
for name in names:
    is_log=name.startswith('log:')
    remote_path=name[4:] if is_log else 'ar-feasibility/'+name+'.json'
    try:
        response=requests.get(session.url.rstrip('/')+'/api/contents/content/'+remote_path,
            headers={'X-Colab-Runtime-Proxy-Token':session.token},params={'type':'file','format':'text'},timeout=15)
    except requests.RequestException as error:
        print(name,'FETCH_ERROR',type(error).__name__)
        continue
    if response.status_code!=200:
        print(name,'HTTP',response.status_code)
        continue
    if is_log:
        print(response.json()['content'][-4500:].encode('ascii','backslashreplace').decode())
        continue
    data=json.loads(response.json()['content'])
    (output/(name+'.json')).write_text(json.dumps(data,indent=2))
    if 'runs' in data:
        print(json.dumps({'report':name,'runs':[dict(run=r['run'],verified=r.get('verified'),errors=r.get('errors'),folder_url=r.get('folder_url')) for r in data['runs']]}))
        continue
    print(json.dumps({k:v for k,v in data.items() if k not in ('native_config','traceback','model_path','updates')}))
    if 'updates' in data:
        print('PASSED_UPDATES',name,len(data['updates']))
