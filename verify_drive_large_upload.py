"""Validate a real >2 GB weight shard using Drive resumable upload, then trash only the new test copy."""
from pathlib import Path
import json
import sys
import time
import shutil
from drive_checkpoint_upload import DriveCheckpointStore

root=Path('/content/drive/MyDrive/LLaDA-ft')
meta=json.loads((root/'datasets/manifest.json').read_text())
files=sorted(Path(meta['model_path']).glob('*.safetensors'))
original=next(p for p in files if p.exists() and p.stat().st_size>2_000_000_000)
assert str(original).startswith(str(root/'hf-cache'))
local_dir=Path('/content/drive_upload_probe')
local_dir.mkdir(exist_ok=True)
source=local_dir/original.name
print('COPYING_TEST_SHARD_TO_LOCAL_DISK',original.name,flush=True)
shutil.copyfile(original,source)
store=DriveCheckpointStore()
folder=store.ensure_path(['LLaDA-ft','storage-validation','resumable-upload-20261022'])
print('TEST_SOURCE',source.name,'BYTES',source.stat().st_size,flush=True)
start=time.perf_counter()
def progress(name,done,total):
    print('UPLOAD_PROGRESS',name,done,total,flush=True)
result=store.upload(source,folder,progress)
result.update(source_name=source.name,elapsed_seconds=time.perf_counter()-start,
    meaning='Base-model shard transport test only; this is not a fine-tuned checkpoint')
print('LARGE_UPLOAD_VERIFIED',json.dumps(result),flush=True)
if not result['reused']:
    # Delete only the test artifact created by this invocation; never the source.
    store.service.files().delete(fileId=result['id']).execute(num_retries=5)
    result['test_copy_deleted']=True
receipt=Path('/content/drive_upload_validation.json')
receipt.write_text(json.dumps(result,indent=2))
store.upload(receipt,folder)
print('VALIDATION_RECEIPT_SAVED',flush=True)
