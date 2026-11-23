"""Generate selected standalone final-only Colab notebooks from reviewed sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
reference=(HERE/'train_llada_bbc.py').read_text(encoding='utf-8')
tree=ast.parse(reference)
helpers='import torch\nimport time\nimport math\nimport hashlib\nimport json\nfrom pathlib import Path\n\n'
for name in ['file_sha','learning_rate','validate_checkpoint','CpuMasterAdamW']:
    node=next(n for n in tree.body if getattr(n,'name',None)==name)
    helpers+=ast.get_source_segment(reference,node)+'\n\n'
probe=(HERE/'ar_model_probe.py').read_text(encoding='utf-8')
loss_node=next(n for n in ast.parse(probe).body if getattr(n,'name',None)=='ar_loss')
helpers+=ast.get_source_segment(probe,loss_node)+'\n'
(HERE/'ar_training_utils.py').write_text(helpers,encoding='utf-8')

override='''# Autoregressive final-only notebook protocol

This notebook retains the matched three-epoch, 4096-token, full-parameter AR continuation protocol below, with these storage and execution changes:

- No periodic model saves, epoch model saves, untrained-base export or full-size upload preflight. No optimizer/master/moment snapshots are written.
- By default, save exactly one final BF16 model after all three epochs, using local Colab staging, validated safetensors and checksummed resumable Drive API upload with COMPLETE.json last. Approximate final weights: Llama 2 13.48 GB; Llama 3 16.06 GB; Mistral 14.48 GB; OLMo 13.78 GB. Epoch-1/2 weight comparisons and recovery after a runtime termination are unavailable in this final-only design.
- `SAVE_FINAL_MODEL=False` instead produces only small text logs, configuration, source code, methods, hashes and diagnostics. It does not produce any durable trained model; all trained weights are lost when the runtime ends.
- Preserve BF16 computation, FP32 gradient accumulation and CPU FP32 master AdamW state. Do not introduce LoRA, quantization or altered optimizer precision to reduce memory. The tested runtime still needs an A100 80 GB and high host RAM.
- Keep non-reentrant activation checkpointing: this recomputes activations to reduce RAM and does not save files. Stream one input block onto the GPU; memory-map local token files; release optimizer hooks, master weights, moments and gradient buffers before final serialization. These changes do not change intended training arithmetic.
- Base weights, Hugging Face/dataset caches, token arrays and serialization staging reside on Colab local disk. No large Windows writes or Drive model cache. Only the optional final model and small text artifacts go to Drive.
- Preserve baseline/per-epoch text diagnostics. They do not save model weights. Training runs as a blocking notebook cell with live progress; there is no synthetic keepalive, automatic restart or exact optimizer resume.
- The production schedule uses the model-specific update count recorded in the adjacent configuration and measured-result table. Warmup and decay are ceil(5%) and ceil(10%) of that count. The 1% random-length policy includes a one-token crop with zero AR target loss, averaged as a real occurrence.
- Original OLMo remains at unscaled 4096 computation with its native 2048 pretraining-context configuration unchanged. This explicitly extrapolates beyond its original context; a successful feasibility probe does not validate its effect on memorization quality.

The appended comparison document describes the reference and earlier feasibility protocol. This section overrides its intermediate checkpoint/retention statements for these notebooks. Each notebook trains only its named model; it does not create a Colab runtime.

---

'''
method=override+(HERE/'AR_COMPARISON_METHOD.md').read_text(encoding='utf-8')
(HERE/'AR_FINAL_ONLY_NOTEBOOK_METHOD.md').write_text(method,encoding='utf-8')
proposals=json.loads((HERE/'ar_training_protocol_proposals.json').read_text())

catalog=[('mistral7b_v01','Mistral 7B v0.1','Mistral7B_BBC_3epochs_Colab.ipynb'),
         ('olmo7b','OLMo 7B','OLMo7B_BBC_3epochs_Colab.ipynb'),
         ('llama2_7b','Llama 2 7B','Llama2_7B_BBC_3epochs_Colab.ipynb'),
         ('llama3_8b','Llama 3 8B','Llama3_8B_BBC_3epochs_Colab.ipynb')]
parser=argparse.ArgumentParser()
parser.add_argument('--models',nargs='+',choices=[row[0] for row in catalog],default=['mistral7b_v01','olmo7b'])
selected=parser.parse_args().models
auxiliary=json.loads((HERE/'ar_probe_results/auxiliary_audits.json').read_text())
for key,title,filename in catalog:
    if key not in selected:continue
    probe_result=json.loads((HERE/'ar_probe_results'/f'{key}_4096.json').read_text())
    assert probe_result['status']=='ok','Require a passed full-parameter probe before producing a training notebook'
    config=dict(next(c for c in proposals if c['model_id']==probe_result['model_id']))
    config.pop('run_name_template',None)
    config.update(model_key=key,checkpoint_epochs=[],checkpoint_every_updates=0,
        verify_full_checkpoint_before_training=False,save_final_model=True,
        save_optimizer_state=False,checkpoint_format='One final BF16 safetensors model only; verified Drive API upload',
        expected_parameters=probe_result['trainable_parameters'],
        expected_control_tokens=auxiliary['native_control_tokens'][key],
        requires_hf_login=key.startswith('llama'),
        methods_filename='AR_FINAL_ONLY_NOTEBOOK_METHOD.md',
        storage_plan='No model outputs before final; one optional final model plus small text; no automatic deletion',
        status='READY_TO_RUN_NOT_STARTED')
    (HERE/f'{key}_final_only_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    cells=[]
    def add(kind,text):
        item=dict(cell_type=kind,metadata={},source=text.strip().splitlines(keepends=True))
        if kind=='code':item.update(execution_count=None,outputs=[])
        item['id']=hashlib.sha256((str(len(cells))+text).encode()).hexdigest()[:12]
        cells.append(item)
    def md(text):add('markdown',text)
    def code(text):add('code',text)
    def script(name):
        source=(HERE/name).read_text(encoding='utf-8')
        ast.parse(source,filename=name)
        code(f'%%writefile /content/{name}\n'+source)

    size=config['expected_parameters']*2/1e9
    context='\n**OLMo context:** 4096 extends beyond the original 2048-token pretraining context. The native configuration and RoPE remain unchanged; this matches the tested probe and is documented as a comparison limitation.\n' if key=='olmo7b' else ''
    md(f'''# {title}: BBC continuation on Colab A100

Run the cells **in order in a fresh A100 80 GB, high-RAM Colab session**. This is full-parameter training, not LoRA: BF16 computation, FP32 accumulated gradients and CPU FP32 master AdamW, 4096-token blocks, accumulation 16, three epochs. It requires **140 GiB available host RAM**, despite lower GPU memory use.

Uses exactly the LLaDA experiment's even-indexed BBC articles from March-June 2025 at the frozen dataset revision. Both training articles and clean odd controls are verified against the canonical LLaDA token-file hashes before native retokenization. {config['expected_tokens']:,} native training tokens, {config['expected_blocks_per_epoch']:,} blocks/epoch, **{config['expected_total_updates']} total updates**.

**Storage:** no periodic saves, no epoch model saves, no full-model upload test, no optimizer snapshots. Default output is **one final model, approximately {size:.2f} GB**, plus small text logs/configuration/diagnostics/methods. Large caches and staging stay on the Colab VM. Set `SAVE_FINAL_MODEL=False` in section 4 for text-only output and no durable trained weights. Nothing large is written to Windows.

The measured compute-only projection is about {probe_result['three_epoch_compute_hours']:.1f} hours; initialization, diagnostics and the single final upload add time. Colab can still terminate a runtime. This final-only notebook cannot recover trained weights if the session ends before that upload completes. Run only one training notebook per runtime.
{context}
Activation checkpointing is retained to reduce RAM: it recomputes activations and writes no checkpoint files.
''')
    md('## 1. Check hardware and local disk')
    code('''import os,sys,json,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone
import torch
assert Path('/content').is_dir(), 'Run this in Google Colab.'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'Select a BF16-capable GPU.'
gpu=torch.cuda.get_device_properties(0)
memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
available=int(memory['MemAvailable'].split()[0])*1024/2**30
free_disk=shutil.disk_usage('/content').free/2**30
print(json.dumps(dict(gpu=gpu.name,vram_gib=gpu.total_memory/2**30,available_host_gib=available,free_colab_disk_gib=free_disk),indent=2))
assert gpu.total_memory>=79*2**30,'Select A100 80 GB.'
assert available>=140,'Need at least 140 GiB AVAILABLE host RAM.'
assert free_disk>=55,'Need 55 GiB free local disk for source weights and optional final staging.'
os.environ.update(HF_HUB_CACHE='/content/ar-ft-cache/hub',HF_DATASETS_CACHE='/content/ar-data-cache',
                  HF_HUB_DISABLE_XET='1',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
''')
    auth_note='Also log in to the Hugging Face account approved for this Llama repository. Enter the token only in the Colab login widget.' if config['requires_hf_login'] else 'No Hugging Face login is required for this public base model.'
    md('## 2. Authorize and mount Drive\nApprove both prompts. '+auth_note)
    code('''from google.colab import auth,drive
auth.authenticate_user()
drive.mount('/content/drive')
ROOT=Path('/content/drive/MyDrive/LLaDA-ft')
for folder in ['runs','logs']:
    (ROOT/folder).mkdir(parents=True,exist_ok=True)
''')
    if config['requires_hf_login']:
        code('''from huggingface_hub import get_token,login
if not get_token():
    login()
else:
    print('Using the saved Hugging Face login; model access is checked below.')
''')
    md('## 3. Install the compatible environment\nUses Colab’s installed PyTorch/CUDA; constrains PyTorch against replacement.')
    code('''VENV=Path('/content/ar-ft-venv')
if not (VENV/'bin/python').exists():
    subprocess.run([sys.executable,'-m','venv','--system-site-packages','--without-pip',str(VENV)],check=True)
PYTHON=str(VENV/'bin/python')
Path('/content/ar-torch-constraint.txt').write_text('torch=='+str(torch.__version__)+'\\n')
packages=['transformers==4.46.3','huggingface-hub==0.36.0','tokenizers==0.20.3',
          'accelerate==1.15.0','datasets==4.8.5','safetensors==0.8.0','sentencepiece==0.2.2',
          'psutil','google-api-python-client','google-auth-httplib2']
subprocess.run([PYTHON,'-m','pip','install','--disable-pip-version-check','--constraint','/content/ar-torch-constraint.txt',*packages],check=True)
subprocess.run([PYTHON,'-c','import torch,transformers,accelerate,datasets,psutil,safetensors; import google.auth,googleapiclient; print("AR_IMPORTS_OK",torch.__version__,transformers.__version__)'],check=True)
''')
    md('## 4. Storage choice and frozen training configuration\n`SAVE_FINAL_MODEL=True` keeps one final trained model. Set it to `False` for text-only output. Preserve the numerical settings for this comparison. An existing run name is never overwritten.')
    code('SAVE_FINAL_MODEL = True\n\n'+f"if 'RUN_NAME' not in globals():\n    RUN_NAME = '{key}_bbc_final_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')\n"+
         'CONFIG = '+repr(config)+'''
CONFIG.update(run_name=RUN_NAME,save_final_model=SAVE_FINAL_MODEL)
assert not (ROOT/'runs'/RUN_NAME).exists(),'Run already started; do not restart under this name.'
CONFIG_PATH=Path('/content/ar_notebook_config.json')
CONFIG_PATH.write_text(json.dumps(CONFIG,indent=2))
print(json.dumps(CONFIG,indent=2))
''')
    md('## 5. Write the standalone source files\nAll implementation is embedded here; no Windows files or previous runtime are required.')
    for name in ['ar_training_utils.py','drive_checkpoint_upload.py','prepare_ar_notebook_data.py','train_ar_bbc.py','validate_ar_final_only.py']:
        script(name)
    code('#@title Write the full comparison and final-only methods document\nMETHODS = '+repr(method)+
         "\nPath('/content/AR_NOTEBOOK_METHOD.md').write_text(METHODS,encoding='utf-8')\nprint('METHODS_READY')")
    md('## 6. Small Drive write/read test and final-model quota check\nOnly a tiny text probe is uploaded. No base-model export or large upload test.')
    code('''preflight = r"""
import json
from pathlib import Path
from drive_checkpoint_upload import DriveCheckpointStore
c=json.loads(Path('/content/ar_notebook_config.json').read_text())
if c.get('requires_hf_login'):
    from huggingface_hub import hf_hub_download
    hf_hub_download(c['model_id'],'config.json',revision=c['model_revision'])
    print('GATED_MODEL_ACCESS_VERIFIED')
s=DriveCheckpointStore()
quota=s.service.about().get(fields='storageQuota').execute(num_retries=5)['storageQuota']
need=c['expected_parameters']*2+2*2**30 if c['save_final_model'] else 2**20
if quota.get('limit'):
    free=int(quota['limit'])-int(quota.get('usage',0))
    print('FREE_DRIVE_GB',round(free/1e9,2),'REQUIRED_GB',round(need/1e9,2))
    assert free>=need,'Insufficient Drive quota for the requested final output'
folder=s.ensure_path(['LLaDA-ft','ar-notebook-access-tests',c['run_name']])
p=Path('/content/ar_drive_text_probe.json')
p.write_text(json.dumps(dict(run=c['run_name'],small_write_test=True),sort_keys=True))
r=s.upload(p,folder)
assert s.service.files().get_media(fileId=r['id']).execute(num_retries=5)==p.read_bytes()
print('SMALL_DRIVE_WRITE_AND_READ_VERIFIED')
"""
subprocess.run([PYTHON,'-u','-c',preflight],check=True)
''')
    md('### Focused CPU checks\nChecks loss alignment, one-token zero loss, schedule endpoints, final-model integrity and interrupted-upload behavior using a tiny temporary model. No GPU training or Drive model upload occurs.')
    code("subprocess.run([PYTHON,'-u','/content/validate_ar_final_only.py'],check=True)")
    md('## 7. Reproduce the exact data and download the pinned base locally\nReuses the saved frozen article JSONL files if present. Otherwise rebuilds from the pinned dataset. Verifies canonical LLaDA training/control hashes, then native token counts/hashes. No token arrays or base weights are written to Drive.')
    code("subprocess.run([PYTHON,'-u','/content/prepare_ar_notebook_data.py',str(CONFIG_PATH)],check=True)")
    md('## 8. Train once\nThis blocking cell prints live updates and writes small text logs to Drive. There are no intermediate model saves. With final saving enabled, it releases optimizer memory, serializes locally and uploads one final model after epoch 3. A final upload failure leaves local staging available while this VM survives. Do not restart training to retry an upload.')
    code('''assert json.loads(CONFIG_PATH.read_text())['run_name']==RUN_NAME
assert not (ROOT/'runs'/RUN_NAME).exists(),'Run already exists; inspect logs instead of restarting.'
LOG_PATH=ROOT/'logs'/(RUN_NAME+'.log')
print('RUN',RUN_NAME,'LOG',LOG_PATH,flush=True)
with LOG_PATH.open('a',encoding='utf-8',buffering=1) as log:
    process=subprocess.Popen([PYTHON,'-u','/content/train_ar_bbc.py','--config',str(CONFIG_PATH)],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in process.stdout:
            log.write(line)
            print(line,end='',flush=True)
        returncode=process.wait()
    except BaseException:
        process.terminate()
        try: process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill();process.wait()
        raise
assert returncode==0,f'Training failed with exit code {returncode}; inspect the text log.'
print('Training process finished. Verify the result below.')
''')
    md('## 9. Verify the cloud result\nChecks completion through the Drive API. With final saving enabled it verifies every saved file against server checksums and reads the completion marker. In text-only mode it explicitly reports that no model was saved.')
    code('''verification = r"""
import json
from pathlib import Path
from drive_checkpoint_upload import DriveCheckpointStore
c=json.loads(Path('/content/ar_notebook_config.json').read_text())
s=DriveCheckpointStore()
parent='root'
for name in ['LLaDA-ft','run-api-records',c['run_name']]:
    parent=s.folder(parent,name,create=False)
    assert parent,'Run API journal missing'
def read(name):
    items=s.children(parent,name)
    assert len(items)==1,'Missing/ambiguous journal file: '+name
    return json.loads(s.service.files().get_media(fileId=items[0]['id']).execute(num_retries=5))
status=read('status_api.json')
assert status['status']=='complete' and status['update']==c['expected_total_updates'],'Training did not complete'
if not c['save_final_model']:
    assert not status.get('final_model_verified')
    print('TEXT_ONLY_RUN_COMPLETE: NO TRAINED MODEL WAS SAVED')
else:
    receipt=read('final_upload_receipt.json')
    assert receipt['verified'] and receipt['run']==c['run_name']
    assert receipt['epoch']==3 and receipt['update']==c['expected_total_updates']
    actual={item['id']:item for item in s.children(receipt['folder_id'])}
    assert set(actual)=={item['id'] for item in receipt['files'].values()}
    for name,saved in receipt['files'].items():
        cloud=actual[saved['id']]
        assert cloud['name']==name and int(cloud['size'])==int(saved['size'])
        assert cloud['md5Checksum']==saved['verified_md5']
        if cloud.get('sha256Checksum'):
            assert cloud['sha256Checksum']==saved['local_sha256']
    marker=receipt['files']['COMPLETE.json']
    complete=json.loads(s.service.files().get_media(fileId=marker['id']).execute(num_retries=5))
    assert complete['run']==c['run_name'] and complete['epoch']==3
    assert complete['update']==c['expected_total_updates']
    assert complete['model_revision']==c['model_revision']
    print('FINAL_MODEL_CLOUD_VERIFIED',receipt['folder_url'])
print('TEXT_LOGS: My Drive/LLaDA-ft/runs/'+c['run_name'])
"""
subprocess.run([PYTHON,'-u','-c',verification],check=True)
''')
    for i,entry in enumerate(cells):
        if entry['cell_type']=='code':
            source=''.join(entry['source'])
            if source.startswith('%%writefile '):source=source.split('\n',1)[1]
            ast.parse(source,filename=f'{filename}:cell-{i}')
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),accelerator='GPU',colab=dict(name=filename,provenance=[],gpuType='A100')),
        nbformat=4,nbformat_minor=5)
    output=HERE/filename
    output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('CREATED',filename,len(cells),'cells',output.stat().st_size,'bytes')
