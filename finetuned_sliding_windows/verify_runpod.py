"""Small end-to-end extraction check on five verified fine-tuned models."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path('/workspace/finetuned-extraction')
CODE=ROOT/'code'
RCLONE=ROOT/'tools/rclone-v1.75.2-linux-amd64/rclone'
CONFIG=Path('/root/.config/research-rclone/rclone.conf')
REMOTE='researchdrive:LLaDA-ft'
ARTICLE_SHA='8f22ff46d5dfc7d85a2c58d598c72252b754a335514e3bd02ca485a18b87e91e'
SPECS=[
 dict(key='llada',family='llada',source='checkpoints/bbc_fp32_colab_20261026_084128_323830/epoch-003',updates=183,
      revision='0f2787f2d87eac5eed8a087d5ecd24277e6255b2',parameters=8015581184),
 dict(key='llama2_7b',family='llama2',source='ar-models/llama2_7b_bbc_final_20261027_222522_712251/final',updates=207,
      revision='01c7f73d771dfac7d292323805ebc428287df4f9',parameters=6738415616),
 dict(key='llama3_8b',family='llama',source='ar-models/llama3_8b_bbc_final_20261027_222543_039805/final',updates=177,
      revision='8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920',parameters=8030261248),
 dict(key='mistral7b_v01',family='mistral',source='ar-models/mistral7b_v01_bbc_final_20261026_212546_149177/final',updates=198,
      revision='27d67f1b5f57dc0953326b2601d68371d40ea8da',parameters=7241732096),
 dict(key='olmo7b',family='olmo',source='ar-models/olmo7b_bbc_final_20261026_212627_105207/final',updates=180,
      revision='11fb3186a2e4f681edea621fa8b4345147a9db6a',parameters=6888095744)]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def remote_call(args,capture=True):
    p=subprocess.run([str(RCLONE),'--config',str(CONFIG),'--log-level','ERROR',*args],
                     stdout=subprocess.PIPE if capture else None,stderr=subprocess.PIPE,text=True)
    if p.returncode:raise RuntimeError('Drive operation failed: '+p.stderr[-1500:])
    return p.stdout


def verify_model(spec):
    folder=ROOT/'models'/spec['key'];folder.mkdir(parents=True,exist_ok=True)
    from drive_checkpoint_download import download_folder
    download_folder(REMOTE+'/'+spec['source'],folder,remote_call)
    marker=json.loads((folder/'COMPLETE.json').read_text())
    assert marker['epoch']==3 and marker['update']==spec['updates']
    assert marker['model_revision']==spec['revision']
    for name,details in marker['files'].items():
        assert Path(name).name==name
        path=folder/name
        assert path.stat().st_size==details['bytes'] and sha(path)==details['sha256'],name
    index=json.loads((folder/'model.safetensors.index.json').read_text())['weight_map']
    assert len(index)==marker['tensor_count']
    assert set(index.values())=={p.name for p in folder.glob('*.safetensors')}
    spec.update(local_path=str(folder),training_run=marker['run'],complete_sha256=sha(folder/'COMPLETE.json'),
                weight_bytes=marker['total_weight_bytes'])
    print(json.dumps(dict(event='FINETUNED_MODEL_DOWNLOADED_AND_VERIFIED',model=spec['key'],run=marker['run'])),flush=True)
    return spec


def prepare(run):
    run.mkdir(parents=True,exist_ok=False)
    assets=ROOT/'assets';assets.mkdir(exist_ok=True)
    for filename in ['bbc_even_after_2025_02.jsonl','bbc_odd_controls.jsonl']:
        remote_call(['copyto',REMOTE+'/datasets/'+filename,str(assets/filename),'--immutable'])
    selected=assets/'bbc_even_after_2025_02.jsonl'
    assert sha(selected)==ARTICLE_SHA
    train=[json.loads(s) for s in selected.open()]
    odd=[json.loads(s) for s in (assets/'bbc_odd_controls.jsonl').open()]
    assert [r['filtered_index'] for r in train]==list(range(0,10643,2))
    def normhash(text):return hashlib.sha256(re.sub(r'\s+',' ',text).strip().encode()).hexdigest()
    training_hashes={normhash(r['content']) for r in train if r['content'].strip()}
    clean=[r for r in odd if r['content'].strip() and normhash(r['content']) not in training_hashes]
    assert len(clean)==3345
    examples=[]
    for group,idx,rows in [('train',1734,train),('clean_odd_control',1695,clean)]:
        row=next(r for r in rows if r['filtered_index']==idx)
        path=run/f'{group}_{idx}.txt';path.write_text(row['content'],encoding='utf-8')
        examples.append(dict(group=group,filtered_index=idx,source_config=row['source_config'],
            source_row=row['source_row'],published_date=row['published_date'],input_path=str(path),
            content_sha256=hashlib.sha256(row['content'].encode()).hexdigest(),windows=[0,1,2]))
    (run/'selection_manifest.json').write_text(json.dumps(dict(dataset_revision='8dd1ecdc92ac43f9c04a3da3e945537dbb08179b',
        frozen_article_sha256=ARTICLE_SHA,examples=examples,negative_policy='Odd articles excluding normalized exact training overlaps'),indent=2))
    access=run/'drive_write_test.json';access.write_text(json.dumps(dict(run=run.name,small_verification=True)))
    destination=REMOTE+'/sliding-window-verification/'+run.name
    remote_call(['copyto',str(access),destination+'/drive_write_test.json','--immutable'])
    assert json.loads(remote_call(['cat',destination+'/drive_write_test.json']))==json.loads(access.read_text())
    print('DRIVE_PRIVATE_READ_AND_RESULT_WRITE_VERIFIED',flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:models=list(pool.map(verify_model,[dict(x) for x in SPECS]))
    (run/'models_manifest.json').write_text(json.dumps(models,indent=2))
    return models


def worker(run,key):
    import torch
    import sliding_window_extraction as extraction
    torch.set_num_threads(4)
    models=json.loads((run/'models_manifest.json').read_text())
    spec=next(s for s in models if s['key']==key)
    selected=json.loads((run/'selection_manifest.json').read_text())['examples']
    model=extraction._load_model(spec['local_path'],spec['family'],'cuda:0')
    model.config.use_cache=False
    assert sum(p.numel() for p in model.parameters())==spec['parameters']
    assert all(p.is_cuda and p.dtype==torch.bfloat16 for p in model.parameters())
    tokenizer=extraction._load_tokenizer(spec['local_path'])
    runs=[]
    for example in selected:
        cases=[('low-confidence','suffix50',list(range(51,101))),('threshold','suffix50',list(range(51,101))),
               ('low-confidence','alternating50',list(range(2,101,2))),('threshold','alternating50',list(range(2,101,2)))] if key=='llada' else [('exact','suffix50',None)]
        for scheme,mask_name,mask in cases:
            output=run/'results'/key/example['group']/scheme/mask_name
            args=['sliding_window_extraction.py',example['input_path'],'--model-family',spec['family'],
                  '--model-name',spec['local_path'],'--device','cuda:0','--output-dir',str(output),
                  '--mode','path_sampling' if key=='llada' else 'exact','--decoding-scheme','full',
                  '--temperature','1.0','--tau','0.001','--num-samples','10','--seed','12345',
                  '--windows','0','1','2','--stride-words','1']
            if key=='llada':args+=['--remasking',scheme,'--confidence-threshold','0.9','--masked_indexes',*map(str,mask)]
            sys.argv=args
            started=time.perf_counter()
            result=extraction.main()
            rows=list(csv.DictReader((result/'windows.csv').open()))
            assert len(rows)==3 and [int(r['window_index']) for r in rows]==[0,1,2]
            assert not any(r['error'] for r in rows),rows
            for row in rows:
                p=float(row['p_z']);logp=float(row['log_p_z'])
                assert math.isfinite(p) and 0<=p<=1+1e-10
                assert not math.isnan(logp) and logp<=1e-10
                if p>0:assert abs(math.log(p)-logp)<1e-8
            summary_path=result/'summary.json'
            summary=json.loads(summary_path.read_text())
            assert summary['checkpoint']['run']==spec['training_run']
            assert summary['num_windows_scored']==3
            summary['data_provenance']=example
            summary['verified_finetuned_model']=spec
            summary['wall_seconds']=time.perf_counter()-started
            summary_path.write_text(json.dumps(summary,indent=2))
            assert not list(result.glob('*detailed*')) and not list(result.glob('*verbose*'))
            runs.append(dict(model=key,group=example['group'],scheme=scheme,mask=mask_name,
                             directory=str(result),rows=len(rows),seconds=summary['wall_seconds']))
    # Independent AR teacher-forcing check of the suffix alignment/convention.
    if key!='llada':
        text=Path(selected[0]['input_path']).read_text()
        ids=tokenizer(text[:800],add_special_tokens=False)['input_ids'][:100]
        tokens=torch.tensor([ids],device='cuda')
        with torch.no_grad():logits=model(tokens,use_cache=False).logits[0]
        step_probs=[float(torch.softmax(logits[49+t],dim=-1)[ids[50+t]]) for t in range(50)]
        manual=-math.inf if any(p==0 for p in step_probs) else sum(math.log(p) for p in step_probs)
        first=list(csv.DictReader((Path(runs[0]['directory'])/'windows.csv').open()))[0]
        score=float(first['log_p_z'])
        assert score==manual or abs(score-manual)<1e-8,'AR suffix alignment differs from teacher forcing'
    report=dict(model=key,status='verified',gpu=torch.cuda.get_device_name(0),runs=runs,
                rows=sum(r['rows'] for r in runs),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                ar_teacher_forcing_alignment_verified=key!='llada',
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_version=torch.__version__,
                model_dtype='bfloat16',llada_estimator_dtype='float64' if key=='llada' else None)
    (run/(key+'_verification.json')).write_text(json.dumps(report,indent=2))
    print('FINETUNED_SLIDING_WORKER_VERIFIED',json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--run')
    args=parser.parse_args()
    if args.worker:worker(Path(args.run),args.worker);return
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('verify_%Y%m%dT%H%M%SZ')
    run=ROOT/'verification'/stamp
    models=prepare(run)
    children=[]
    for gpu,spec in enumerate(models):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
        log=(run/(spec['key']+'.log')).open('w')
        child=subprocess.Popen([sys.executable,__file__,'--worker',spec['key'],'--run',str(run)],
            stdout=log,stderr=subprocess.STDOUT,env=env)
        children.append((spec,child,log))
    failures=[]
    for spec,child,log in children:
        code=child.wait();log.close()
        if code:failures.append(dict(model=spec['key'],exit_code=code))
    reports=[json.loads(p.read_text()) for p in run.glob('*_verification.json')]
    result=dict(run=stamp,status='verified' if not failures and len(reports)==5 else 'failed',
        failures=failures,models_verified=len(reports),windows_scored=sum(r['rows'] for r in reports),
        full_sweep_started=False,normal_results_only=True,
        estimator_source_sha256=sha(CODE/'probabilistic_extraction.py'),
        models=reports)
    (run/'verification_summary.json').write_text(json.dumps(result,indent=2))
    remote_call(['copy',str(run),REMOTE+'/sliding-window-verification/'+stamp,'--immutable','--transfers','3'])
    print('FINETUNED_VERIFICATION_RESULT',json.dumps(result),flush=True)
    print('DRIVE_RESULTS_PATH','LLaDA-ft/sliding-window-verification/'+stamp,flush=True)
    if failures:raise RuntimeError('Some model verification workers failed; output files preserved')


if __name__=='__main__':main()
