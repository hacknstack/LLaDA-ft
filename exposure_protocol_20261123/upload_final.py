"""CPU-only final upload can overlap the next model's GPU training."""
import argparse
import json
from pathlib import Path
from drive_access import store

ROOT=Path('/workspace/bbc-exposure')
parser=argparse.ArgumentParser();parser.add_argument('--key',required=True);args=parser.parse_args()
locations=json.loads((ROOT/'protocol/output_locations.json').read_text())
run=ROOT/'runs'/locations['run_name']/args.key
assert (run/'COMPUTE_COMPLETE.json').exists()
s=store();dest=locations['models'][args.key]
receipt=s.upload_directory(run/'final',dest['final'],lambda name,done,total:print('UPLOAD',args.key,name,done,total,flush=True) if done==total else None)
(run/'final_upload_receipt.json').write_text(json.dumps(receipt,indent=2))
for p in run.iterdir():
    if p.is_file() and p.name!='status.json':s.upload(p,dest['logs'])
state=json.loads((run/'status.json').read_text());state.update(state='complete',final_folder_url=receipt['folder_url'])
(run/'status.json').write_text(json.dumps(state,indent=2));s.upload(run/'status.json',dest['logs'])
print('FINAL_MODEL_VERIFIED_ON_DRIVE',args.key,receipt['folder_url'],flush=True)
