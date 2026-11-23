"""Download verified final artifacts onto Colab and smoke-test native loading."""
import os
os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_CACHE='/content/ar-ft-cache/hub',HF_HUB_DISABLE_XET='1')
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

OUT=Path('/content/ar-feasibility')
STAGE=Path('/content/ar-saved-model-smoke')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare():
    from drive_checkpoint_upload import DriveCheckpointStore
    from googleapiclient.http import MediaIoBaseDownload
    store=DriveCheckpointStore()
    (OUT/'saved_ar_models_download_status.json').write_text(json.dumps(dict(status='running')))
    audit=json.loads((OUT/'mistral_olmo_run_audit.json').read_text())
    from concurrent.futures import ThreadPoolExecutor
    jobs=[]
    for run in audit['runs']:
        if not run['verified']:continue
        folder=STAGE/run['run'];folder.mkdir(parents=True,exist_ok=True)
        for name,item in run['receipt']['files'].items():
            assert Path(name).name==name
            path=folder/name
            if path.exists() and path.stat().st_size==int(item['size']) and sha(path)==item['local_sha256']:
                continue
            jobs.append((run['run'],name,item,path))
    def download_one(job):
        run_name,name,item,path=job
        # Separate httplib2 transports: the discovery client is not thread-safe.
        client=DriveCheckpointStore()
        temporary=path.with_name(path.name+'.partial')
        with temporary.open('wb') as f:
            download=MediaIoBaseDownload(f,client.service.files().get_media(fileId=item['id']),chunksize=64*2**20)
            done=False
            while not done:_,done=download.next_chunk(num_retries=8)
        assert temporary.stat().st_size==int(item['size']) and sha(temporary)==item['local_sha256'],name
        temporary.replace(path)
        print('SAVED_MODEL_FILE_DOWNLOADED_AND_VERIFIED',run_name,name,flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download_one,jobs))
    (OUT/'saved_ar_models_download_status.json').write_text(json.dumps(dict(status='complete')))
    print('SAVED_AR_MODELS_PREPARED',flush=True)


def read_range(url,start,end):
    """Never buffer an entire base-model shard if an HTTP server ignores Range."""
    import requests
    from huggingface_hub import get_token
    headers={'Range':f'bytes={start}-{end}','Cache-Control':'no-cache'}
    token=get_token()
    if token:headers['Authorization']='Bearer '+token
    with requests.get(url,headers=headers,stream=True,timeout=45) as response:
        assert response.status_code==206,'Base server did not honor HTTP Range'
        assert response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'),'Unexpected range response'
        maximum=end-start+1
        content=bytearray()
        for piece in response.iter_content(chunk_size=8192):
            content.extend(piece)
            assert len(content)<=maximum,'Oversized range response'
        assert len(content)==maximum
        return content


def parameter_comparison(model,config):
    import torch
    from huggingface_hub import hf_hub_download,hf_hub_url
    index=json.loads(Path(hf_hub_download(config['model_id'],'model.safetensors.index.json',revision=config['model_revision'])).read_text())
    checks=[]
    for key in ['model.layers.0.self_attn.q_proj.weight','model.layers.0.mlp.down_proj.weight']:
        try:
            shard=index['weight_map'][key]
            url=hf_hub_url(config['model_id'],shard,revision=config['model_revision'])
            size=struct.unpack('<Q',read_range(url,0,7))[0]
            assert 0<size<4*2**20
            header=json.loads(read_range(url,8,7+size))
            entry=header[key]
            element_size={'BF16':2,'F16':2,'F32':4}[entry['dtype']]
            begin=8+size+entry['data_offsets'][0]
            count=512
            content=read_range(url,begin,begin+count*element_size-1)
            source=torch.frombuffer(content,dtype={'BF16':torch.bfloat16,'F16':torch.float16,'F32':torch.float32}[entry['dtype']])
            # Match the actual BF16-load -> FP32-master initialization convention.
            baseline=source.to(torch.bfloat16).float()
            trained=model.get_parameter(key).detach().flatten()[:count].float().cpu()
            checks.append(dict(parameter=key,values=count,changed_values=int(torch.count_nonzero(trained!=baseline)),
                               maximum_abs_difference=float((trained-baseline).abs().max())))
        except Exception as error:
            checks.append(dict(parameter=key,comparison_unavailable=type(error).__name__))
    return checks


def one(run_name):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(12)
    folder=STAGE/run_name
    config=json.loads((folder/'training_config.json').read_text())
    started=time.perf_counter()
    result=dict(run=run_name,model_id=config['model_id'],success=False)
    try:
        model=AutoModelForCausalLM.from_pretrained(folder,torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,device_map={'':0},attn_implementation='sdpa')
        model.eval();model.config.use_cache=False
        assert sum(p.numel() for p in model.parameters())==config['expected_parameters']
        tokenizer=AutoTokenizer.from_pretrained(folder)
        prompt='The government announced new measures on Monday.'
        ids=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)['input_ids'].to('cuda')
        with torch.no_grad():
            output=model(input_ids=ids,labels=ids,use_cache=False)
            assert torch.isfinite(output.loss) and torch.isfinite(output.logits).all()
            loss=float(output.loss)
            del output
            generated=model.generate(input_ids=ids,max_new_tokens=16,do_sample=False,
                use_cache=False,pad_token_id=tokenizer.eos_token_id)
        try:
            comparisons=parameter_comparison(model,config)
        except Exception as error:
            comparisons=[dict(comparison_unavailable=type(error).__name__)]
        result.update(success=True,finite_forward_and_loss=True,short_prompt_ce=loss,
            generated_new_tokens=generated.shape[1]-ids.shape[1],
            generated_text=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True),
            parameter_checks=comparisons,elapsed_seconds=time.perf_counter()-started)
        result['trained_values_differ_from_bf16_base']=any(c.get('changed_values',0)>0 for c in result['parameter_checks'])
    except Exception as error:
        result.update(error_type=type(error).__name__,error=str(error))
    (OUT/(config['model_key']+'_saved_model_smoke.json')).write_text(json.dumps(result,indent=2))
    print('SAVED_AR_MODEL_SMOKE',json.dumps(result),flush=True)


def main(mode,run_name=None):
    if mode=='prepare':
        try:prepare()
        except Exception as error:
            (OUT/'saved_ar_models_download_status.json').write_text(json.dumps(dict(status='failed',error_type=type(error).__name__)))
            raise
        return
    if mode=='one':one(run_name);return
    deadline=time.monotonic()+1200
    while True:
        path=OUT/'saved_ar_models_download_status.json'
        status=json.loads(path.read_text()) if path.exists() else dict(status='not_started')
        assert status['status']!='failed','Saved-model download failed; inspect its log'
        if status['status']=='complete':break
        assert time.monotonic()<deadline,'Saved-model download did not finish in time'
        time.sleep(10)
    audit=json.loads((OUT/'mistral_olmo_run_audit.json').read_text())
    for run in audit['runs']:
        if run['verified']:
            subprocess.run([sys.executable,__file__,'one','--run',run['run']],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','one','test']);p.add_argument('--run')
    args=p.parse_args();main(args.mode,args.run)
