"""Reproduce the frozen articles/controls and download only this AR base locally."""
import os
os.environ.update(HF_HUB_CACHE='/content/ar-ft-cache/hub',HF_DATASETS_CACHE='/content/ar-data-cache',
                  HF_HUB_DISABLE_XET='1',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
import gc
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import numpy as np
from huggingface_hub import HfApi,snapshot_download
from transformers import AutoTokenizer

ROOT=Path('/content/drive/MyDrive/LLaDA-ft')
DATA_REV='8dd1ecdc92ac43f9c04a3da3e945537dbb08179b'
LLADA_REV='0f2787f2d87eac5eed8a087d5ecd24277e6255b2'
ARTICLE_SHA='8f22ff46d5dfc7d85a2c58d598c72252b754a335514e3bd02ca485a18b87e91e'


def digest(text):
    return hashlib.sha256(re.sub(r'\s+',' ',text).strip().encode()).hexdigest()


def encode(tokenizer,texts):
    batches=tokenizer(texts,add_special_tokens=False,return_attention_mask=False,return_token_type_ids=False)['input_ids']
    return np.asarray([token for ids in batches for token in ids+[tokenizer.eos_token_id]],dtype=np.int32)


def array_sha(array):
    data=io.BytesIO();np.save(data,array)
    return hashlib.sha256(data.getvalue()).hexdigest()


def source_rows():
    selected_path=ROOT/'datasets/bbc_even_after_2025_02.jsonl'
    odd_path=ROOT/'datasets/bbc_odd_controls.jsonl'
    if selected_path.exists() and odd_path.exists():
        assert hashlib.sha256(selected_path.read_bytes()).hexdigest()==ARTICLE_SHA,'Frozen article file differs; refusing to train'
        selected=[json.loads(line) for line in selected_path.open()]
        odd=[json.loads(line) for line in odd_path.open()]
    else:
        import datetime
        from datasets import load_dataset
        selected=[];odd=[];filtered=0
        for month in ['2025-03','2025-04','2025-05','2025-06']:
            dataset=load_dataset('RealTimeData/bbc_news_alltime',month,split='train',revision=DATA_REV)
            for source_i,row in enumerate(dataset):
                date=str(row.get('published_date') or '')[:10]
                try:
                    date=datetime.date.fromisoformat(date).isoformat()
                except ValueError:
                    continue
                if date<'2025-03-01':
                    continue
                idx=filtered;filtered+=1
                record=dict(filtered_index=idx,source_config=month,source_row=source_i,
                    published_date=date,title=row.get('title'),link=row.get('link'),content=row.get('content') or '')
                (odd if idx%2 else selected).append(record)
            print('PINNED_MONTH_READY',month,flush=True)
        assert filtered==10643
        canonical=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected).encode()
        assert hashlib.sha256(canonical).hexdigest()==ARTICLE_SHA,'Rebuilt article bytes differ from frozen source'
    assert [row['filtered_index'] for row in selected]==list(range(0,10643,2))
    assert [row['filtered_index'] for row in odd]==list(range(1,10643,2))
    assert all(row['published_date']>='2025-03-01' for row in selected+odd)
    train=[row['content'] for row in selected if row['content'].strip()]
    assert len(train)==5315
    hashes={digest(text) for text in train}
    assert len(hashes)==4236
    clean=[row['content'] for row in odd if row['content'].strip() and digest(row['content']) not in hashes]
    assert len(clean)==3345
    return train,clean


def main(config_path):
    config=json.loads(Path(config_path).read_text())
    output=Path('/content/ar-inputs')/config['model_key'];output.mkdir(parents=True,exist_ok=True)
    train,controls=source_rows()
    reference=AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base',revision=LLADA_REV,trust_remote_code=True)
    reference_tokens=encode(reference,train)
    assert len(reference_tokens)==3941989 and array_sha(reference_tokens)=='f5e29db9e7b1459bf030cac1567f77f6595da59acdfaf010ab977b21fa0254df'
    del reference_tokens
    reference_tokens=encode(reference,controls)
    assert len(reference_tokens)==2617971 and array_sha(reference_tokens)=='3a6968777b4ae68e4ed1f89b3309ab4e1133a43a2a659bf7fe561dc70416e27d'
    del reference_tokens,reference
    gc.collect()
    print('EXACT_LLADA_ARTICLES_AND_CONTROLS_VERIFIED',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(config['model_id'],revision=config['model_revision'],use_fast=True)
    assert tokenizer.eos_token_id==config['native_eos_token_id']
    files={}
    for filename,texts,count in [('train_tokens.npy',train,config['expected_tokens']),
                                ('control_tokens.npy',controls,config['expected_control_tokens'])]:
        tokens=encode(tokenizer,texts)
        assert len(tokens)==count
        path=output/filename;np.save(path,tokens)
        files[filename]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if filename=='train_tokens.npy':
            assert files[filename]['sha256']==config['expected_native_token_file_sha256']
        del tokens
    del train,controls
    gc.collect()
    info=HfApi().model_info(config['model_id'],revision=config['model_revision'])
    assert info.sha==config['model_revision']
    names=[item.rfilename for item in info.siblings if '/' not in item.rfilename]
    weights=[name for name in names if name.endswith('.safetensors')]
    assert weights,'Require native safetensors weights'
    allowed=weights+[name for name in names if name.endswith(('.json','.model','.txt','.tiktoken'))]
    model_path=snapshot_download(config['model_id'],revision=config['model_revision'],allow_patterns=allowed,max_workers=4)
    index_path=Path(model_path)/'model.safetensors.index.json'
    if index_path.exists():
        weight_map=json.loads(index_path.read_text())['weight_map']
        assert set(weight_map.values())==set(weights)
        assert all(Path(name).name==name and (Path(model_path)/name).is_file() for name in weight_map.values())
    manifest=dict(model_id=config['model_id'],model_revision=config['model_revision'],model_path=model_path,
        dataset_id='RealTimeData/bbc_news_alltime',dataset_revision=DATA_REV,
        selected_articles=5322,nonempty_train_articles=5315,clean_control_articles=3345,
        empty_train_articles=7,unique_normalized_train_contents=4236,duplicate_train_rows=1079,
        frozen_article_file_sha256=ARTICLE_SHA,train_tokens=config['expected_tokens'],
        clean_control_tokens=config['expected_control_tokens'],token_files=files,
        tokenization='Native tokenizer, no BOS/chat/title; content plus one EOS per nonempty article')
    (output/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('AR_LOCAL_MODEL_AND_DATA_READY',json.dumps(manifest),flush=True)


if __name__=='__main__':
    main(sys.argv[1])
