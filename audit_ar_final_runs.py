"""Read-only audit of final-only Mistral and OLMo artifacts in Drive."""
import datetime
import json
from pathlib import Path
from drive_checkpoint_upload import DriveCheckpointStore

store=DriveCheckpointStore()
out=Path('/content/ar-feasibility');out.mkdir(exist_ok=True)
result=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=[])
project=store.folder('root','LLaDA-ft',create=False)
assert project
journals=store.folder(project,'run-api-records',create=False)
models_root=store.folder(project,'ar-models',create=False)
model_runs={f['name']:f for f in store.children(models_root)} if models_root else {}
for run_folder in store.children(journals):
    name=run_folder['name']
    if not name.startswith(('mistral7b_v01_bbc_final_','olmo7b_bbc_final_')):continue
    report=dict(run=name,errors=[],verified=False)
    try:
        records={f['name']:f for f in store.children(run_folder['id'])}
        def read(item):
            assert int(item.get('size',0))<10*2**20
            return json.loads(store.service.files().get_media(fileId=item['id']).execute(num_retries=5))
        status=read(records['status_api.json']);config=read(records['config.json'])
        report.update(status=status,config=config,
            diagnostics=[read(item) for name,item in sorted(records.items()) if name.startswith('diagnostics_epoch_')])
        if 'data_manifest.json' in records:report['data_manifest']=read(records['data_manifest.json'])
        assert status['status']=='complete','Run did not report completion'
        assert status['epoch']==3 and status['update']==config['expected_total_updates']
        if not config.get('save_final_model',True):
            report['text_only_run']=True
            assert not status.get('final_model_verified')
        else:
            receipt=read(records['final_upload_receipt.json'])
            assert receipt['verified'] and receipt['run']==name
            assert receipt['epoch']==3 and receipt['update']==config['expected_total_updates']
            assert name in model_runs,'Model run folder absent'
            folder=store.folder(model_runs[name]['id'],'final',create=False)
            assert folder==receipt['folder_id']
            cloud={f['name']:f for f in store.children(folder)}
            assert set(cloud)==set(receipt['files']),'Missing or extra final-model files'
            for filename,saved in receipt['files'].items():
                actual=cloud[filename]
                assert actual['id']==saved['id'] and int(actual['size'])==int(saved['size']),filename
                assert actual['md5Checksum']==saved['verified_md5']==saved['md5Checksum'],filename
                assert actual['sha256Checksum']==saved['local_sha256'],filename
            complete=read(cloud['COMPLETE.json'])
            assert complete['run']==name and complete['epoch']==3 and complete['update']==config['expected_total_updates']
            assert complete['model_revision']==config['model_revision']
            for filename,details in complete['files'].items():
                assert int(cloud[filename]['size'])==details['bytes'] and cloud[filename]['sha256Checksum']==details['sha256'],filename
            index=read(cloud['model.safetensors.index.json'])
            shards=set(index['weight_map'].values())
            assert shards=={n for n in cloud if n.endswith('.safetensors')}
            assert len(index['weight_map'])==complete['tensor_count']
            assert sum(int(cloud[n]['size']) for n in shards)==complete['total_weight_bytes']
            report.update(verified=True,folder_url=receipt['folder_url'],receipt=receipt,
                weight_bytes=complete['total_weight_bytes'],shards=len(shards),files=len(cloud))
    except Exception as error:
        report['errors'].append(dict(type=type(error).__name__,message=str(error) if isinstance(error,AssertionError) else 'API/read failure'))
    result['runs'].append(report)
    print(json.dumps(dict(event='AR_SAVED_RUN_AUDIT',run=name,verified=report['verified'],
        status=report.get('status'),folder_url=report.get('folder_url'),errors=report['errors'])),flush=True)
(out/'mistral_olmo_run_audit.json').write_text(json.dumps(result,indent=2))
print('AR_FINAL_RUN_AUDIT_FINISHED',flush=True)
