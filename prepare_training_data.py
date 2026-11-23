"""Materialize the frozen even-index corpus and clean odd-index diagnostics on Drive."""
from benchmark_llada import ROOT
import hashlib
import json
from pathlib import Path
import re


def digest_text(text):
    return hashlib.sha256(re.sub(r'\s+',' ',text).strip().encode('utf-8')).hexdigest()


def main():
    import numpy as np
    from transformers import AutoTokenizer
    from datasets import load_dataset
    meta=json.loads((ROOT/'datasets/manifest.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(meta['model_path'],trust_remote_code=True)
    selected=[json.loads(line) for line in (ROOT/'datasets/bbc_even_after_2025_02.jsonl').open()]
    assert len(selected)==meta['selected_articles']
    train_tokens=[]
    hashes={}
    empty=0
    for row in selected:
        assert row['filtered_index']%2==0 and row['published_date']>='2025-03-01'
        text=row['content']
        if not text.strip():
            empty+=1
            continue
        hashes.setdefault(digest_text(text),[]).append(row['filtered_index'])
        train_tokens.extend(tokenizer.encode(text,add_special_tokens=False)+[tokenizer.eos_token_id])
    assert len(train_tokens)==meta['total_tokens_with_eos']
    assert empty==meta['selected_empty_articles']
    train_path=ROOT/'datasets/train_tokens.npy'
    np.save(train_path,np.asarray(train_tokens,dtype=np.int32))
    print('TRAIN_READY',len(train_tokens),flush=True)
    control_path=ROOT/'datasets/bbc_odd_controls.jsonl'
    control_tokens=[]
    filtered=odd=odd_empty=overlaps=0
    with control_path.open('w') as out:
        for config in meta['configs']:
            ds=load_dataset(meta['dataset_id'],config,split='train',revision=meta['dataset_revision'])
            for source_i,row in enumerate(ds):
                date=str(row.get('published_date') or '')[:10]
                # The frozen metadata contains ISO dates; reject malformed dates explicitly.
                import datetime
                try:
                    date=datetime.date.fromisoformat(date).isoformat()
                except ValueError:
                    continue
                if date<'2025-03-01':
                    continue
                idx=filtered
                filtered+=1
                if idx%2==0:
                    continue
                odd+=1
                text=row.get('content') or ''
                empty_text=not text.strip()
                overlap=not empty_text and digest_text(text) in hashes
                odd_empty+=int(empty_text)
                overlaps+=int(overlap)
                out.write(json.dumps(dict(filtered_index=idx,source_config=config,source_row=source_i,
                    published_date=date,title=row.get('title'),link=row.get('link'),content=text,
                    empty=empty_text,exact_normalized_overlap_with_train=overlap),ensure_ascii=False)+'\n')
                if not empty_text and not overlap:
                    control_tokens.extend(tokenizer.encode(text,add_special_tokens=False)+[tokenizer.eos_token_id])
            print('CONTROL_MONTH_READY',config,flush=True)
    assert filtered==meta['eligible_articles']
    assert odd==filtered//2
    controls_path=ROOT/'datasets/control_tokens_clean.npy'
    np.save(controls_path,np.asarray(control_tokens,dtype=np.int32))
    assert len(control_tokens)>4096*8
    data=dict(dataset_revision=meta['dataset_revision'],model_revision=meta['model_revision'],
        selected_articles=len(selected),nonempty_train_articles=len(selected)-empty,
        train_tokens=len(train_tokens),train_blocks_4096=(len(train_tokens)+4095)//4096,
        train_final_block_tokens=len(train_tokens)%4096 or 4096,
        unique_normalized_train_contents=len(hashes),duplicate_nonempty_train_rows=len(selected)-empty-len(hashes),
        duplicate_train_groups={k:v for k,v in hashes.items() if len(v)>1},
        odd_control_articles=odd,empty_odd_controls=odd_empty,
        odd_controls_exact_overlap=overlaps,clean_nonempty_odd_controls=odd-odd_empty-overlaps,
        clean_control_tokens=len(control_tokens),
        overlap_definition='SHA256 of content after Unicode whitespace collapse and trimming; case and punctuation preserved',
        training_selection='Immutable even filtered_index; overlaps and within-train duplicates are retained in training',
        near_duplicate_audit='Not performed; exact overlap filtering alone does not establish independence',
        token_files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (train_path,controls_path)})
    (ROOT/'datasets/training_data_manifest.json').write_text(json.dumps(data,indent=2))
    print('TRAINING_DATA_READY',json.dumps({k:v for k,v in data.items() if k!='duplicate_train_groups'}),flush=True)


if __name__=='__main__':
    main()
