"""Read-only Drive audit from an authenticated fresh Colab runtime.

Reads small manifests/receipts and Drive server checksums; no weight downloads.
"""
import datetime
import json
from pathlib import Path
import google.auth
from googleapiclient.discovery import build

creds, _ = google.auth.default()
service = build('drive', 'v3', credentials=creds, cache_discovery=False)
FIELDS = 'id,name,mimeType,size,md5Checksum,sha256Checksum,parents'
result = dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), runs=[])


def children(parent):
    files = []
    token = None
    while True:
        response = service.files().list(q=f"'{parent}' in parents and trashed=false",
            fields=f'nextPageToken,files({FIELDS})', pageSize=1000, pageToken=token).execute(num_retries=3)
        files.extend(response.get('files', []))
        token = response.get('nextPageToken')
        if not token:
            return files


def folder(parent, name):
    matches = [f for f in children(parent) if f['name'] == name and f['mimeType'] == 'application/vnd.google-apps.folder']
    assert len(matches) == 1, f'Missing or ambiguous folder: {name}'
    return matches[0]['id']


def read(item):
    assert int(item.get('size', 0)) < 10 * 2**20, 'Refusing to download a large file during this audit'
    return json.loads(service.files().get_media(fileId=item['id']).execute(num_retries=3))


def verify_checkpoint(receipt, run_name, checkpoint_items, baseline):
    assert receipt.get('verified') is True
    folder_id = receipt['folder_id']
    assert folder_id in {f['id'] for f in checkpoint_items}, 'Checkpoint folder is not under this run'
    cloud = {f['name']: f for f in children(folder_id)}
    expected = receipt['files']
    assert set(cloud) == set(expected), 'Missing or unexpected checkpoint files'
    for name, saved in expected.items():
        actual = cloud[name]
        assert actual['id'] == saved['id'] and int(actual['size']) == int(saved['size']), name
        assert actual['md5Checksum'] == saved['verified_md5'] == saved['md5Checksum'], name
        assert actual.get('sha256Checksum'), 'Server SHA256 unavailable: ' + name
        assert actual['sha256Checksum'] == saved['local_sha256'], name
    complete = read(cloud['COMPLETE.json'])
    assert complete['run'] == run_name and complete['epoch'] == receipt['epoch']
    assert complete['update'] == receipt['update'] and complete['epoch'] > 0
    for name, details in complete['files'].items():
        assert int(cloud[name]['size']) == details['bytes'], name
        assert cloud[name]['sha256Checksum'] == details['sha256'], name
    index = read(cloud['model.safetensors.index.json'])
    shards = set(index['weight_map'].values())
    assert len(index['weight_map']) == complete['tensor_count']
    assert all(Path(name).name == name and name in cloud for name in shards)
    assert shards == {name for name in cloud if name.endswith('.safetensors')}
    weight_bytes = sum(int(cloud[name]['size']) for name in shards)
    assert weight_bytes == complete['total_weight_bytes']
    different_from_base = None
    if baseline:
        base_files = baseline['files']
        assert all(name in base_files for name in shards)
        different_from_base = any(cloud[name]['sha256Checksum'] != base_files[name]['local_sha256'] for name in shards)
        assert different_from_base, 'Trained checkpoint weights equal untrained Base export'
    saved_config = read(cloud['training_config.json'])
    assert saved_config['run_name'] == run_name
    return dict(tag=receipt['checkpoint'], epoch=complete['epoch'], update=complete['update'],
        folder_url=receipt['folder_url'], verified=True, files=len(cloud), shards=len(shards),
        tensor_count=complete['tensor_count'], weight_bytes=weight_bytes,
        different_from_base=different_from_base, model_only=not complete['optimizer_saved'])


project = folder('root', 'LLaDA-ft')
journal_root = folder(project, 'run-api-records')
checkpoint_root = folder(project, 'checkpoints')
checkpoint_runs = {f['name']: f for f in children(checkpoint_root)}
for run_folder in children(journal_root):
    if run_folder['mimeType'] != 'application/vnd.google-apps.folder':
        continue
    name = run_folder['name']
    row = dict(run=name, checkpoints=[], errors=[])
    try:
        records = {f['name']: f for f in children(run_folder['id'])}
        row['status'] = read(records['status_api.json']) if 'status_api.json' in records else None
        for filename, key in [('config.json', 'config'), ('environment.json', 'environment'),
                              ('data_manifest.json', 'data_manifest')]:
            if filename in records:
                data = read(records[filename])
                if key == 'data_manifest':
                    data.pop('duplicate_train_groups', None)
                row[key] = data
        row['diagnostics'] = [read(item) for filename, item in sorted(records.items())
                              if filename.startswith('diagnostics_epoch_') and filename.endswith('.json')]
        base_name = 'base-transport-preflight_upload_receipt.json'
        baseline = read(records[base_name]) if base_name in records else None
        cp_items = children(checkpoint_runs[name]['id']) if name in checkpoint_runs else []
        row['checkpoint_folder_names'] = sorted(item['name'] for item in cp_items)
        for filename, item in sorted(records.items()):
            if filename.startswith('epoch-') and filename.endswith('_upload_receipt.json'):
                try:
                    verified = verify_checkpoint(read(item), name, cp_items, baseline)
                    row['checkpoints'].append(verified)
                    print(json.dumps(dict(event='CHECKPOINT_VERIFIED', run=name, **verified)), flush=True)
                except Exception as error:
                    row['errors'].append(dict(file=filename, type=type(error).__name__, message=str(error) if isinstance(error, AssertionError) else 'API/read failure'))
        row['all_three_epochs_verified'] = sorted(c['epoch'] for c in row['checkpoints']) == [1, 2, 3]
        row['successful'] = bool(row['status'] and row['status'].get('status') == 'complete'
                                 and row['status'].get('update') == 183 and row['all_three_epochs_verified'] and not row['errors'])
    except Exception as error:
        row['errors'].append(dict(type=type(error).__name__, message=str(error) if isinstance(error, AssertionError) else 'API/read failure'))
    result['runs'].append(row)
    print(json.dumps(dict(event='RUN_AUDIT', run=name, status=row.get('status'),
        successful=row.get('successful', False), checkpoint_folder_names=row.get('checkpoint_folder_names'),
        errors=row['errors'])), flush=True)

Path('/content/llada_drive_audit.json').write_text(json.dumps(result, indent=2))
print('READ_ONLY_DRIVE_AUDIT_FINISHED', flush=True)
