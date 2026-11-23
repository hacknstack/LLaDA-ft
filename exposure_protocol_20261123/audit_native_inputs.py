"""Independently re-encode originals and verify every expanded occurrence."""
import collections
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

os.environ.update(TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
ROOT=Path('/workspace/bbc-exposure')
KEYS=['llada8b','llama2_7b','llama3_8b','mistral7b_v01','olmo7b']


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    from transformers import AutoTokenizer
    article_dir=ROOT/'data/frozen_exposure'
    articles=[json.loads(l) for l in (article_dir/'articles.jsonl').open()]
    result={}
    for key in KEYS:
        path=ROOT/'inputs'/key
        while not (path/'training_config.json').exists():time.sleep(2)
        config=json.loads((path/'training_config.json').read_text())
        assert all(sha(path/name)==h for name,h in config['input_file_sha256'].items())
        tokenizer=AutoTokenizer.from_pretrained(config['model_path'],trust_remote_code=key=='llada8b')
        ids=tokenizer([r['content'] for r in articles],add_special_tokens=False,return_attention_mask=False,return_token_type_ids=False)['input_ids']
        native={r['article_id']:np.asarray(x+[tokenizer.eos_token_id],dtype=np.int32) for r,x in zip(articles,ids)}
        stream=np.load(path/'train_tokens.npy',mmap_mode='r')
        counts=collections.Counter();starting_updates=collections.defaultdict(collections.Counter)
        offset=0
        for line in (path/'occurrence_spans.jsonl').open():
            r=json.loads(line);name=r['article_id'];start=r['start_token'];end=r['end_token']
            assert start==offset and np.array_equal(stream[start:end],native[name])
            assert stream[end-1]==tokenizer.eos_token_id
            counts[name]+=1;starting_updates[name][start//65536]+=1;offset=end
        assert offset==len(stream) and all(counts[r['article_id']]==r['exposures'] for r in articles)
        control=np.load(path/'control_tokens.npy')
        assert np.array_equal(control,np.concatenate([native[r['article_id']] for r in articles if r['role']=='control']))
        groups=[]
        for e in [0,1,2,4,8,16,32,64,128,256]:
            rows=[r for r in articles if r['role']!='background' and r['exposures']==e]
            groups.append(dict(exposures=e,articles=len(rows),mean_native_content_tokens=float(np.mean([len(native[r['article_id']])-1 for r in rows])),
                min_distinct_starting_updates=min(len(starting_updates[r['article_id']]) for r in rows),
                max_distinct_starting_updates=max(len(starting_updates[r['article_id']]) for r in rows),
                max_occurrences_starting_in_same_update=max((max(starting_updates[r['article_id']].values(),default=0) for r in rows),default=0)))
        item=dict(passed=True,total_tokens=len(stream),occurrences=sum(counts.values()),all_source_token_spans_verified=True,
            every_exposure_count_verified=True,controls_excluded_from_training=True,groups=groups,
            caution='Presentation counts are not counts of separate optimizer updates; copies may start within the same update and articles can span boundaries.')
        (path/'independent_input_audit.json').write_text(json.dumps(item,indent=2))
        result[key]=item
        print('INDEPENDENT_NATIVE_INPUT_AUDIT_PASSED',key,flush=True)
    (ROOT/'audits/all_native_input_audits.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
