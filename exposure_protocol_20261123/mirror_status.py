"""CPU-only live text status mirror; never blocks a GPU training worker."""
import json
from pathlib import Path
import time
from drive_access import store

ROOT=Path('/workspace/bbc-exposure')
locations=json.loads((ROOT/'protocol/output_locations.json').read_text())
run=ROOT/'runs'/locations['run_name']
while True:
    try:
        s=store()
        for key,folders in locations['models'].items():
            path=run/key/'status.json'
            if path.exists():s.write_state(folders['logs'],'status_api.json',json.loads(path.read_text()))
        path=run/'suite_status.json'
        if path.exists():s.write_state(locations['parent_id'],'suite_status_api.json',json.loads(path.read_text()))
    except Exception as e:
        print('STATUS_MIRROR_RETRY',type(e).__name__,flush=True)
    if (run/'MIRROR_STOP').exists():break
    for _ in range(60):
        if (run/'MIRROR_STOP').exists():break
        time.sleep(5)
