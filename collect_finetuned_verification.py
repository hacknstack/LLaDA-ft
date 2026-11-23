"""Download only the small verification outputs, then build a review bundle."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import zipfile

parser=argparse.ArgumentParser()
parser.add_argument('run')
args=parser.parse_args()
assert args.run.startswith('verify_') and '/' not in args.run and '\\' not in args.run
workspace=Path(__file__).resolve().parent
tools=Path.home()/'.codex/rclone-tools'
install=json.loads((tools/'install.json').read_text())
config=tools/'research-drive-auth/rclone.conf'
remote='researchdrive:LLaDA-ft/sliding-window-verification'
destination=workspace/'finetuned_verification_results'/args.run
destination.mkdir(parents=True,exist_ok=True)
base=[install['executable'],'--config',str(config),'--log-level','ERROR','--checksum']
subprocess.run([*base,'copy',remote+'/'+args.run,str(destination),'--immutable',
                '--max-size','10M','--transfers','4'],check=True)
summary=json.loads((destination/'verification_summary.json').read_text())
assert summary['status']=='verified' and summary['models_verified']==5 and summary['windows_scored']==48
records=[]
for model in summary['models']:
    for run in model['runs']:
        # Absolute pod paths in the audit manifest are resolved by their suffix.
        suffix=Path(run['directory'].split('/'+args.run+'/',1)[1])
        with (destination/suffix/'windows.csv').open(newline='',encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                records.append(dict(model=model['model'],group=run['group'],scheme=run['scheme'],
                                    mask=run['mask'],**row))
assert len(records)==48 and not any(r['error'] for r in records)
with (destination/'all_windows.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
metadata=json.loads(subprocess.check_output([*base,'lsjson',remote,'--dirs-only'],text=True))
folder=next(item for item in metadata if item['Name']==args.run)
folder_url='https://drive.google.com/drive/folders/'+folder['ID']
(destination/'DRIVE_FOLDER.txt').write_text(folder_url+'\n')
subprocess.run([*base,'copyto',str(destination/'all_windows.csv'),remote+'/'+args.run+'/all_windows.csv','--immutable'],check=True)
if (destination/'VERIFICATION_REPORT.md').exists():
    subprocess.run([*base,'copyto',str(destination/'VERIFICATION_REPORT.md'),remote+'/'+args.run+'/VERIFICATION_REPORT.md','--immutable'],check=True)
archive=workspace/'finetuned_verification_results'/f'{args.run}.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
    for file in destination.rglob('*'):
        if file.is_file():bundle.write(file,args.run+'/'+file.relative_to(destination).as_posix())
    for file in (workspace/'finetuned_sliding_windows').glob('*'):
        if file.is_file() and file.suffix in {'.py','.md'}:bundle.write(file,'code/'+file.name)
    for name in ['build_finetuned_sliding_runner.py','collect_finetuned_verification.py']:
        bundle.write(workspace/name,'code/'+name)
downloads=Path.home()/'Downloads'/archive.name
downloads.write_bytes(archive.read_bytes())
print(json.dumps(dict(status='verified',rows=len(records),drive=folder_url,zip=str(archive),download=str(downloads)),indent=2))
