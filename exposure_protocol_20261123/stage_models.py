"""Download pinned base artifacts and compile native-token exposure manifests."""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

os.environ.update(TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
ROOT=Path('/workspace/bbc-exposure')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(key):
    import numpy as np
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    protocol=json.loads((ROOT/'protocol/draft_protocol.json').read_text())
    config=protocol['models'][key]
    dest=ROOT/'inputs'/key
    dest.mkdir(parents=True,exist_ok=True)
    config_path=dest/'training_config.json'
    if config_path.exists():
        saved=json.loads(config_path.read_text())
        assert all(sha(dest/name)==h for name,h in saved['input_file_sha256'].items())
        print('INPUTS_ALREADY_VERIFIED',key,flush=True)
        return
    article_dir=ROOT/'data/frozen_exposure'
    manifest=json.loads((article_dir/'manifest.json').read_text())
    assert all(sha(article_dir/name)==h for name,h in manifest['file_sha256'].items())
    print('MODEL_DOWNLOAD_START',key,flush=True)
    model_path=snapshot_download(config['model_id'],revision=config['model_revision'],cache_dir=ROOT/'cache/models',
             allow_patterns=['*.safetensors','*.json','*.py','*.model','*.txt','*.tiktoken'],max_workers=4)
    print('MODEL_DOWNLOAD_COMPLETE',key,flush=True)
    tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=(key=='llada8b'),use_fast=True)
    assert tokenizer.eos_token_id is not None
    rows=[json.loads(l) for l in (article_dir/'articles.jsonl').open()]
    occurrences=[json.loads(l) for l in (article_dir/'occurrences.jsonl').open()]
    encoded=tokenizer([r['content'] for r in rows],add_special_tokens=False,return_attention_mask=False,return_token_type_ids=False)['input_ids']
    tokens={r['article_id']:np.asarray(ids+[tokenizer.eos_token_id],dtype=np.int32) for r,ids in zip(rows,encoded)}
    total=sum(len(tokens[r['article_id']]) for r in occurrences)
    stream=np.lib.format.open_memmap(dest/'train_tokens.npy',mode='w+',dtype=np.int32,shape=(total,))
    offset=0
    consumed=collections.Counter()
    with (dest/'occurrence_spans.jsonl').open('w') as f:
        for r in occurrences:
            ids=tokens[r['article_id']]
            end=offset+len(ids)
            stream[offset:end]=ids
            f.write(json.dumps(dict(r,start_token=offset,end_token=end,content_tokens=len(ids)-1,eos_tokens=1,
                 first_block=offset//4096,last_block=(end-1)//4096,first_update=offset//65536+1,last_update=(end-1)//65536+1))+'\n')
            offset=end
            consumed[r['article_id']]+=1
    stream.flush()
    assert offset==total and all(consumed[r['article_id']]==r['exposures'] for r in rows)
    control_ids=np.concatenate([tokens[r['article_id']] for r in rows if r['role']=='control'])
    np.save(dest/'control_tokens.npy',control_ids)
    # Select deterministic nonempty training blocks for parity with earlier diagnostics.
    blocks=math.ceil(total/4096)
    updates=math.ceil(blocks/16)
    config.update(model_key=key,model_path=model_path,seed=42,num_epochs=None,random_length_probability=0.,
          actual_train_tokens=total,actual_control_tokens=len(control_ids),blocks=blocks,tail_tokens=total%4096 or 4096,
          total_updates=updates,warmup_updates=math.ceil(updates*.05),decay_updates=math.ceil(updates*.1),
          native_eos_token_id=tokenizer.eos_token_id,gradient_checkpointing=True,
          content_order_manifest_sha256=sha(article_dir/'occurrences.jsonl'),article_manifest_sha256=sha(article_dir/'articles.jsonl'),
          input_file_sha256={name:sha(dest/name) for name in ['train_tokens.npy','control_tokens.npy','occurrence_spans.jsonl']},
          scheduled_articles=len(rows),scheduled_occurrences=len(occurrences),all_article_exposure_counts_verified=True,
          final_remainder_window=blocks%16 or 16,status='INPUTS_READY_PRODUCTION_NOT_STARTED')
    config_path.write_text(json.dumps(config,indent=2))
    print('INPUTS_READY',key,'tokens',total,'updates',updates,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('key');args=parser.parse_args()
    try:main(args.key)
    except Exception as e:
        out=ROOT/'audits'/('stage_'+args.key+'_error.json')
        out.write_text(json.dumps(dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()),indent=2))
        print('STAGING_FAILED',args.key,type(e).__name__,flush=True)
        raise
