"""Upload only small experiment code, methods and measurements to Drive."""
import datetime
import json
from pathlib import Path
from drive_checkpoint_upload import DriveCheckpointStore

store=DriveCheckpointStore()
tag='feasibility-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
folder=store.ensure_path(['LLaDA-ft','ar-comparison',tag])
files=[Path('/content')/name for name in ['AR_COMPARISON_METHOD.md','ar_training_protocol_proposals.json',
        'ar_model_probe.py','ar_probe_optimizer.py','ar_probe_audits.py']]
files+=sorted(Path('/content/ar-feasibility').glob('*_4096.json'))
files.append(Path('/content/ar-feasibility/auxiliary_audits.json'))
for name in ['AR_FINAL_ONLY_NOTEBOOK_METHOD.md','AR_TRAINING_TESTS_20261027.md',
             'Llama2_7B_BBC_3epochs_Colab.ipynb','Llama3_8B_BBC_3epochs_Colab.ipynb',
             'ar_combined_auxiliary_audits.json']:
    path=Path('/content')/name
    if path.is_file():files.append(path)
for name in ['mistral_olmo_run_audit.json','mistral7b_v01_saved_model_smoke.json','olmo7b_saved_model_smoke.json']:
    path=Path('/content/ar-feasibility')/name
    if path.is_file():files.append(path)
receipt=dict(folder_url='https://drive.google.com/drive/folders/'+folder,files={})
for path in files:
    assert path.is_file() and path.stat().st_size<2**20, 'Only small evidence files are authorized here'
    receipt['files'][path.name]=store.upload(path,folder)
    print('EVIDENCE_UPLOADED',path.name,flush=True)
Path('/content/ar-feasibility/evidence_upload_receipt.json').write_text(json.dumps(receipt,indent=2))
print('AR_EVIDENCE_SAVED',receipt['folder_url'],flush=True)
