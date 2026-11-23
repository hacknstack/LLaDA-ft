"""Read the independent API-owned run-status journal without relying on Drive FUSE."""
from drive_checkpoint_upload import DriveCheckpointStore
import sys

run=sys.argv[1] if len(sys.argv)>1 else 'bbc_fp32_3ep_v3_api'
store=DriveCheckpointStore()
parent='root'
for name in ['LLaDA-ft','run-api-records',run]:
    parent=store.folder(parent,name,create=False)
    if parent is None:
        raise SystemExit('API journal folder not found')
items=store.children(parent,'status_api.json')
assert len(items)==1
raw=store.service.files().get_media(fileId=items[0]['id']).execute(num_retries=5)
print(raw.decode('utf-8'))
