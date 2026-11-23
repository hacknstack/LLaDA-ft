"""Build the standalone Colab notebook from the reviewed training sources."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
cells = []


def cell(kind, text):
    item = dict(cell_type=kind, metadata={}, source=text.strip().splitlines(keepends=True))
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    cells.append(item)


def md(text):
    cell('markdown', text)


def code(text):
    cell('code', text)


def script(name, text=None):
    text = text if text is not None else (HERE / name).read_text(encoding='utf-8')
    ast.parse(text, filename=name)
    code(f'%%writefile /content/{name}\n' + text)


config = json.loads((HERE / 'llada_training_config_v3.json').read_text())
manifest = json.loads((HERE / 'benchmark_manifest.json').read_text())
manifest['model_path'] = '/content/llada-base-model'
manifest['packing'] = 'Content + EOS, 4096-token blocks; final short block unpadded'
for key in ('blocks_1024', 'sample_token_count'):
    manifest.pop(key, None)
expected_tokens = json.loads((HERE / 'training_data_manifest.json').read_text())['token_files']

md('''# LLaDA-8B-Base: BBC memorization training

Run these cells **in order in a fresh Google Colab GPU runtime**. Select an **A100 80 GB with high host RAM**; this tested setup requires at least **140 GiB available CPU RAM**. The notebook prints the actual hardware and stops if resources are insufficient.

This is the agreed **full-parameter continuation**: BF16 model computation, FP32 accumulated gradients, CPU FP32 master weights and AdamW moments; 4096-token blocks, microbatch 1, accumulation 16, **3 epochs / 183 optimizer updates**. The model and dataset revisions are frozen. It uses even indices after filtering publication dates to March 2025 onwards; the pinned dataset contains March–June 2025. There are 5,322 selected articles (5,315 nonempty), 3,941,989 tokens and 963 blocks.

All large downloads and serialization happen **on Colab's local disk**, never on your Windows computer. Final files upload to **My Drive / LLaDA-ft / checkpoints / RUN_NAME** via resumable Drive API uploads. Each model is approximately **16.03 GB / 14.93 GiB**; three epoch snapshots total about **48.1 GB**, plus temporary rolling snapshots. The quota check requires approximately 66.3 GB free: at most four models in Drive simultaneously, plus 2 GiB headroom. This is the budget for this run only; other jobs uploading into the same Drive can consume its headroom.

Expect roughly **3 h 40 min of training computation on the measured machine**, plus diagnostics and substantial upload time. A full untrained-base upload verifies transport before any training. It is explicitly labeled `base-transport-preflight`. Do not mistake it for a trained checkpoint. The training cell shows live progress and keeps the notebook kernel doing real work; Colab can still disconnect or terminate it.

Epoch snapshots and rolling snapshots are **model-only**. They support evaluation or a separately defined new training run, **not exact optimizer/RNG resumption**. A new notebook run starts from the pinned base. This notebook refuses to overwrite an existing run. Only an older rolling snapshot belonging to this run is removed after its replacement has been verified; epoch snapshots are retained.

The complete methods document, source code, configuration, data checksums, environment and upload receipts are saved with the run. Diagnostic losses are monitoring measurements, not a memorization/extraction test.
''')

md('## 1. Check the actual runtime before downloading anything')
code('''import os, sys, json, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone
import torch

assert Path('/content').is_dir(), 'Run this notebook in Google Colab.'
assert torch.cuda.is_available(), 'Select a GPU runtime first.'
gpu = torch.cuda.get_device_properties(0)
meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
available_gib = int(meminfo['MemAvailable'].split()[0]) * 1024 / 2**30
disk_gib = shutil.disk_usage('/content').free / 2**30
print(json.dumps({'gpu': gpu.name, 'vram_gib': gpu.total_memory / 2**30,
                  'available_host_ram_gib': available_gib, 'free_colab_disk_gib': disk_gib,
                  'torch': torch.__version__}, indent=2))
assert gpu.total_memory >= 79 * 2**30, 'This protocol requires an 80 GB GPU.'
assert torch.cuda.is_bf16_supported(), 'BF16 support is required.'
assert available_gib >= 140, 'Select a high-RAM runtime: need 140 GiB AVAILABLE host RAM.'
assert disk_gib >= 45, 'Need at least 45 GiB free local disk for base and checkpoint staging.'
''')

md('## 2. Authorize Google Drive\nApprove the authentication and mount prompts. Credentials stay in Colab; the notebook does not print them.')
code('''from google.colab import auth, drive
auth.authenticate_user()
drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/LLaDA-ft')
for name in ['datasets', 'docs', 'logs', 'runs']:
    (ROOT / name).mkdir(parents=True, exist_ok=True)
''')

md('## 3. Install the compatible training environment\nAn isolated environment inherits Colab’s existing PyTorch and CUDA installation. The constraints prevent replacing PyTorch. Other Python dependencies are installed only on Colab.')
code('''VENV = Path('/content/llada-bench-venv')
if not (VENV / 'bin/python').exists():
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '--without-pip', str(VENV)], check=True)
PYTHON = str(VENV / 'bin/python')
Path('/content/llada-constraints.txt').write_text('torch==' + str(torch.__version__) + '\\n')
requirements = ['transformers==4.46.3', 'peft==0.14.0', 'huggingface-hub==0.36.0',
                'tokenizers==0.20.3', 'datasets==4.8.5', 'accelerate==1.15.0',
                'safetensors==0.7.0', 'google-api-python-client', 'google-auth-httplib2', 'psutil']
subprocess.run([PYTHON, '-m', 'pip', 'install', '--disable-pip-version-check',
                '--constraint', '/content/llada-constraints.txt', *requirements], check=True)
subprocess.run([PYTHON, '-c', 'import torch, transformers, peft, datasets, accelerate, safetensors, psutil; '
                'import google.auth, googleapiclient.discovery; '
                'print("TRAINING_IMPORTS_OK", torch.__version__, transformers.__version__)'], check=True)
os.environ.update(HF_HOME='/content/hf-cache-local', HF_DATASETS_CACHE='/content/hf-cache-local/datasets',
                  HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false', USE_TF='0', USE_FLAX='0')
''')

md('## 4. Run identity and frozen configuration\nA unique name is generated once per notebook kernel. Keep the scientific settings unchanged to reproduce this protocol. An interrupted run cannot be exactly resumed; use a new name only when you intend to restart from Base.')
code('''if 'RUN_NAME' not in globals():
    RUN_NAME = 'bbc_fp32_colab_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
''' + 'CONFIG = ' + repr(config) + '''
CONFIG.update(run_name=RUN_NAME, methods_filename=RUN_NAME + '_METHODS.md')
CONFIG_PATH = Path('/content/llada_training_config.json')
CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2))
assert not (ROOT / 'runs' / RUN_NAME).exists(), 'Run already exists; do not overwrite or restart it.'
print('RUN_NAME:', RUN_NAME)
print(json.dumps(CONFIG, indent=2))
''')

md('## 5. Write the standalone implementation\nThese cells contain the actual reviewed training and upload implementation. They write small source files under `/content`; you can inspect or edit them before training. No files from the Windows workspace are needed.')
script('benchmark_llada.py', '''"""Shared paths and environment; no benchmark is launched by this module."""
import os
from pathlib import Path
ROOT = Path('/content/drive/MyDrive/LLaDA-ft')
os.environ.update(HF_HOME='/content/hf-cache-local', HF_DATASETS_CACHE='/content/hf-cache-local/datasets',
                  HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false', USE_TF='0', USE_FLAX='0')
''')
for name in ['drive_checkpoint_upload.py', 'train_llada_bbc.py', 'download_base_local.py',
             'validate_training_setup.py', 'validate_drive_checkpoint_upload.py']:
    script(name)
prep = (HERE / 'prepare_training_data.py').read_text()
prep = prep.replace("AutoTokenizer.from_pretrained(meta['model_path'],trust_remote_code=True)",
                    "AutoTokenizer.from_pretrained('/content/llada-base-model',trust_remote_code=True)")
script('prepare_training_data.py', prep)

md('## 6. Verify Drive API writing and actual storage quota\nThis performs a real small upload and reads it back, before downloading the model. Large checkpoint uploads later use 64 MiB resumable chunks, server checksum checks, and a `COMPLETE.json` marker uploaded last.')
script('notebook_drive_preflight.py', '''import json, sys
from pathlib import Path
from drive_checkpoint_upload import DriveCheckpointStore
config=json.loads(Path('/content/llada_training_config.json').read_text())
store=DriveCheckpointStore()
quota=store.service.about().get(fields='storageQuota').execute(num_retries=5)['storageQuota']
# Worst overlap: (epochs-1) retained epochs + old rolling + replacement upload.
# Local serialization staging is not another Drive copy.
required=(config['epochs']+1)*8015581184*2+2*2**30
if quota.get('limit'):
    free=int(quota['limit'])-int(quota.get('usage',0))
    print('DRIVE_FREE_GIB',round(free/2**30,2),'REQUIRED_GIB',round(required/2**30,2))
    assert free>=required, 'Insufficient Drive quota for epoch and rolling snapshots.'
folder=store.ensure_path(['LLaDA-ft','run-api-records',config['run_name']])
path=Path('/content/drive_write_probe.json')
payload=json.dumps({'run':config['run_name'],'write_test':True},sort_keys=True).encode()
path.write_bytes(payload)
receipt=store.upload(path,folder)
assert store.service.files().get_media(fileId=receipt['id']).execute(num_retries=5)==payload
print('DRIVE_API_WRITE_AND_READ_VERIFIED')
''')
code("subprocess.run([PYTHON, '-u', '/content/notebook_drive_preflight.py'], check=True)")

md('## 7. Download the pinned Base model to Colab local disk\nApproximately 16 GB. It deliberately bypasses the old Drive-mounted model cache. The model repository contains custom Python code, loaded from the fixed revision using `trust_remote_code=True`.')
code("subprocess.run([PYTHON, '-u', '/content/download_base_local.py'], check=True)")

md('## 8. Verify the frozen data, or rebuild it if missing\nReuse requires the original pinned revisions and both canonical token-file SHA-256 hashes. Otherwise rebuild the selection from the four pinned monthly configurations in source order, filter dates, select global even indices, then tokenize content + EOS. Training duplicates remain; exact train overlaps are excluded only from the odd-index diagnostic controls. This is the frozen March–June 2025 corpus, not newly added months.')
data_script = '''import datetime, hashlib, json, subprocess, sys
from pathlib import Path
from benchmark_llada import ROOT
EXPECTED = __MANIFEST__
TOKEN_FILES = __TOKENS__
folder=ROOT/'datasets'
folder.mkdir(parents=True,exist_ok=True)
meta_path=folder/'manifest.json'
if meta_path.exists():
    old=json.loads(meta_path.read_text())
    for key in ['dataset_id','dataset_revision','model_id','model_revision','configs']:
        assert old[key]==EXPECTED[key], 'Existing project has a different frozen corpus/model: '+key

def cache_valid():
    try:
        metadata=json.loads((folder/'training_data_manifest.json').read_text())
        for key in ['model_revision','dataset_revision']:
            assert metadata[key]==EXPECTED[key]
        assert metadata['train_tokens']==3941989 and metadata['train_blocks_4096']==963
        assert metadata['token_files']==TOKEN_FILES
        for name,details in TOKEN_FILES.items():
            path=folder/name
            assert path.stat().st_size==details['bytes']
            with path.open('rb') as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==details['sha256']
        return True
    except (OSError, ValueError, KeyError, AssertionError):
        return False

if cache_valid():
    if not meta_path.exists():
        meta_path.write_text(json.dumps(EXPECTED,indent=2))
    print('CANONICAL_TRAIN_AND_CONTROL_CACHE_VERIFIED',flush=True)
else:
    from datasets import load_dataset
    filtered=selected=empty=0
    target=folder/'bbc_even_after_2025_02.jsonl'
    temporary=target.with_suffix('.rebuild.jsonl')
    with temporary.open('w',encoding='utf-8') as out:
        for config in EXPECTED['configs']:
            ds=load_dataset(EXPECTED['dataset_id'],config,split='train',revision=EXPECTED['dataset_revision'])
            for source_i,row in enumerate(ds):
                date=str(row.get('published_date') or '')[:10]
                try:
                    date=datetime.date.fromisoformat(date).isoformat()
                except ValueError:
                    continue
                if date<'2025-03-01':
                    continue
                idx=filtered
                filtered+=1
                if idx%2:
                    continue
                text=row.get('content') or ''
                record=dict(filtered_index=idx,source_config=config,source_row=source_i,
                            published_date=date,title=row.get('title'),link=row.get('link'),content=text)
                out.write(json.dumps(record,ensure_ascii=False)+'\\n')
                selected+=1
                empty+=int(not text.strip())
            print('SELECTED_MONTH',config,flush=True)
    assert (filtered,selected,empty)==(10643,5322,7)
    temporary.replace(target)
    meta_path.write_text(json.dumps(EXPECTED,indent=2))
    subprocess.run([sys.executable,'-u','/content/prepare_training_data.py'],check=True)
    assert cache_valid(), 'Rebuilt tokens do not match the frozen reference; do not train.'
    print('REBUILT_CANONICAL_DATA_VERIFIED',flush=True)
'''.replace('__MANIFEST__', repr(manifest)).replace('__TOKENS__', repr(expected_tokens))
script('notebook_prepare_data.py', data_script)
code("subprocess.run([PYTHON, '-u', '/content/notebook_prepare_data.py'], check=True)")

md('## 9. Save the full methods document\nThis records similarities to LLaDA, resource-driven differences and unresolved original-training details. The added notebook note identifies storage/environment changes; the numerical training protocol is unchanged.')
methods = (HERE / 'LLADA_BBC_METHOD_v3.md').read_text(encoding='utf-8')
methods = methods.replace('**Run identifier:** `bbc_fp32_3ep_v3_api`', '**Run identifier:** recorded in the adjacent run configuration')
methods += '''\n\n## Standalone notebook execution note\n\nThe notebook generates a unique run name and embeds all implementation files. The shared path module is reduced to paths/environment only; Hugging Face caches and the pinned Base weights reside on Colab local disk. Data reuse requires canonical SHA-256 token-file checks; missing data is rebuilt from the pinned revisions and original global index rule. A genuine Drive API write/read check and resource/quota checks precede large downloads. The numerical training, diagnostics, checkpoint validation and upload implementation remain the protocol above. Python dependencies are pinned where tested; Colab supplies its installed PyTorch/CUDA version, which is constrained against replacement and recorded per run. Model-only snapshots do not restore FP32 optimizer state, data order or RNG state.\n'''
methods += '\nThe notebook quota calculation supersedes the earlier conservative storage budget: with three retained epochs, peak simultaneous Drive storage is two completed epochs plus one old rolling snapshot plus one replacement upload (four model payloads). Colab-local serialization does not consume Drive quota. The guard requires four payloads plus 2 GiB headroom, about 66.3 GB free at startup. This covers this run only; concurrent uploads by other jobs must be budgeted separately.\n'
code("#@title Save complete protocol document\nMETHODS_TEXT = " + repr(methods) + "\nmethods_path = ROOT / 'docs' / CONFIG['methods_filename']\nmethods_path.write_text(METHODS_TEXT, encoding='utf-8')\nprint('METHODS_SAVED:', methods_path)\n")

md('## 10. Run the focused checks\nCPU checks cover masked loss alignment, zero-mask handling, effective batch averaging including the short final batch, FP32 master/state behavior, learning-rate endpoints, checkpoint validation, upload failure gates and completion-marker ordering. No model training occurs here.')
code("for filename in ['validate_training_setup.py', 'validate_drive_checkpoint_upload.py']:\n    subprocess.run([PYTHON, '-u', '/content/' + filename], check=True)")

md('''## 11. Train — this cell runs for hours

Run once. Leave this blocking cell running. It first evaluates baseline diagnostics and uploads an **untrained** full-model transport test. Training starts only after that upload has passed. It then saves rolling snapshots every 20 global updates when more than five updates remain in the epoch, plus all three epoch snapshots. Each new successful snapshot replaces the preceding rolling snapshot; completed epoch snapshots remain.

Progress is printed live and logged under `My Drive/LLaDA-ft/logs/`. A checkpoint is usable only when the Drive API receipt is verified and its cloud folder contains `COMPLETE.json`. An upload failure stops the run. A Colab termination can still lose progress since the most recent completed upload. No automatic exact resume is claimed.
''')
code('''assert CONFIG_PATH.exists()
assert json.loads(CONFIG_PATH.read_text())['run_name'] == RUN_NAME
assert not (ROOT / 'runs' / RUN_NAME).exists(), 'This run already started; inspect its status instead of restarting.'
LOG_PATH = ROOT / 'logs' / (RUN_NAME + '.log')
print('Starting', RUN_NAME, '\\nLog:', LOG_PATH, flush=True)
with LOG_PATH.open('a', encoding='utf-8', buffering=1) as log:
    process = subprocess.Popen([PYTHON, '-u', '/content/train_llada_bbc.py', '--config', str(CONFIG_PATH)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            log.write(line)
            print(line, end='', flush=True)
        returncode = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
if returncode:
    raise RuntimeError(f'Training exited with code {returncode}. Inspect the log and API status; do not assume completion.')
print('Training process finished. Run the final cloud verification cell below.', flush=True)
''')

md('## 12. Verify all three epoch checkpoints in Drive\nThis checks the API journal and every epoch receipt against current cloud file sizes/checksums, including the completion marker. A success message is printed only after all three checks pass. This cell runs after training; the same notebook kernel cannot run another cell while the training cell is busy.')
script('notebook_verify_final.py', '''import json
from pathlib import Path
from drive_checkpoint_upload import DriveCheckpointStore
config=json.loads(Path('/content/llada_training_config.json').read_text())
store=DriveCheckpointStore()
parent='root'
for part in ['LLaDA-ft','run-api-records',config['run_name']]:
    parent=store.folder(parent,part,create=False)
    assert parent, 'Run API journal missing.'
def read_json(name):
    files=store.children(parent,name)
    assert len(files)==1, 'Missing or ambiguous journal file: '+name
    return json.loads(store.service.files().get_media(fileId=files[0]['id']).execute(num_retries=5))
status=read_json('status_api.json')
print('RUN_STATUS',json.dumps(status,indent=2))
assert status['status']=='complete', 'Run did not report successful completion.'
for epoch in [1,2,3]:
    receipt=read_json(f'epoch-{epoch:03d}_upload_receipt.json')
    assert receipt['verified'] and receipt['epoch']==epoch and 'COMPLETE.json' in receipt['files']
    for name,expected in receipt['files'].items():
        actual=store.service.files().get(fileId=expected['id'],fields='id,name,size,md5Checksum,sha256Checksum').execute(num_retries=5)
        assert actual['name']==name and actual['size']==expected['size']
        assert actual['md5Checksum']==expected['md5Checksum']
        if actual.get('sha256Checksum'):
            assert actual['sha256Checksum']==expected['local_sha256']
    marker=receipt['files']['COMPLETE.json']
    complete=json.loads(store.service.files().get_media(fileId=marker['id']).execute(num_retries=5))
    assert complete['epoch']==epoch and complete['run']==config['run_name']
    print(f'EPOCH_{epoch}_CLOUD_VERIFIED',receipt['folder_url'],flush=True)
print('ALL_THREE_TRAINED_EPOCH_MODELS_VERIFIED_IN_DRIVE')
''')
code("subprocess.run([PYTHON, '-u', '/content/notebook_verify_final.py'], check=True)")

notebook = dict(cells=cells, metadata=dict(
    kernelspec=dict(display_name='Python 3', language='python', name='python3'),
    language_info=dict(name='python', version='3'), accelerator='GPU',
    colab=dict(name='LLaDA_BBC_3epochs_Colab.ipynb', provenance=[], gpuType='A100')),
    nbformat=4, nbformat_minor=5)
for index, item in enumerate(cells):
    source = ''.join(item['source'])
    item['id'] = hashlib.sha256(f'{index}:{source}'.encode()).hexdigest()[:12]
    if item['cell_type'] == 'code':
        python = source.split('\n', 1)[1] if source.startswith('%%writefile ') else source
        ast.parse(python, filename=f'cell_{index}')
output = HERE / 'LLaDA_BBC_3epochs_Colab.ipynb'
output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(f'WROTE {output.name}: {len(cells)} cells, {output.stat().st_size:,} bytes; all Python cells parse')
