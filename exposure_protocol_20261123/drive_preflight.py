"""Prepare unique final/log destinations, quota checks and tiny verified upload."""
import datetime
import json
from pathlib import Path
from drive_access import store

ROOT=Path('/workspace/bbc-exposure')
protocol=json.loads((ROOT/'protocol/draft_protocol.json').read_text())
s=store()
quota=s.service.about().get(fields='storageQuota').execute(num_retries=5)['storageQuota']
parameters=dict(llada8b=8015581184,llama2_7b=6738415616,llama3_8b=8030261248,mistral7b_v01=7241732096,olmo7b=6888095744)
payload=sum(parameters.values())*2
free=None if not quota.get('limit') else int(quota['limit'])-int(quota.get('usage',0))
assert free is None or free>=payload+10*2**30,'Insufficient Drive quota for all five final models plus headroom'
old=ROOT/'protocol/output_locations.json'
if old.exists():locations=json.loads(old.read_text())
else:
    name='bbc_exposure_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    project='1cF3b2y9NAyEXZmC9FPZ9B8RDQYRP9SJz'
    check=s.service.files().get(fileId=project,fields='id,name,mimeType').execute(num_retries=5)
    assert check['mimeType']=='application/vnd.google-apps.folder'
    parent=s.folder(s.folder(project,'exposure-models'),name)
    meta=s.service.files().get(fileId=parent,fields='id,name,webViewLink').execute(num_retries=5)
    locations=dict(run_name=name,parent_id=parent,folder_url=meta['webViewLink'],models={})
    for key in protocol['models']:
        folder=s.folder(parent,key)
        locations['models'][key]=dict(logs=s.folder(folder,'logs'),final=s.folder(folder,'final'))
    old.write_text(json.dumps(locations,indent=2))
test=ROOT/'audits/drive_preflight_receipt_source.json'
test.write_text(json.dumps(dict(purpose='Small Drive transport preflight; no model weights',models=list(parameters),training_started=False)))
receipt=s.upload(test,locations['parent_id'])
report=dict(passed=True,quota=quota,free_bytes=free,final_weights_payload_bytes=payload,
   required_with_headroom_bytes=payload+10*2**30,run_name=locations['run_name'],folder_url=locations['folder_url'],
   verified_small_upload=receipt,full_model_transport_tested_this_session=False)
(ROOT/'audits/drive_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(passed=True,free_GB=None if free is None else free/1e9,weights_GB=payload/1e9,folder_url=locations['folder_url'])))
