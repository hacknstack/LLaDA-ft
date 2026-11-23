"""Exact corpus/window inventory on pod storage; no article copies to Windows."""
import os
os.environ.update(USE_TF='0',USE_FLAX='0',RAYON_NUM_THREADS='4',TOKENIZERS_PARALLELISM='true')
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,re,time
from pathlib import Path

ROOT=Path('/workspace/finetuned-extraction')
OUT=ROOT/'planning_20261101';OUT.mkdir(exist_ok=True)
KEYS=['llada','llama2_7b','llama3_8b','mistral7b_v01','olmo7b']

def digest(text):return hashlib.sha256(re.sub(r'\s+',' ',text).strip().encode()).hexdigest()

def corpus():
    raw=[json.loads(s) for s in (ROOT/'assets/bbc_even_after_2025_02.jsonl').open()]
    odd=[json.loads(s) for s in (ROOT/'assets/bbc_odd_controls.jsonl').open()]
    assert hashlib.sha256((ROOT/'assets/bbc_even_after_2025_02.jsonl').read_bytes()).hexdigest()=='8f22ff46d5dfc7d85a2c58d598c72252b754a335514e3bd02ca485a18b87e91e'
    train_hashes={digest(r['content']) for r in raw if r['content'].strip()}
    groups={};counts={}
    for name,rows in [('train',raw),('control',odd)]:
        selected={};nonempty=0;overlaps=0
        for row in rows:
            if not row['content'].strip():continue
            nonempty+=1;h=digest(row['content'])
            if name=='control' and h in train_hashes:overlaps+=1;continue
            if h in selected:selected[h]['original_filtered_indices'].append(row['filtered_index']);continue
            selected[h]=dict(row,group=name,normalized_sha256=h,
                original_filtered_indices=[row['filtered_index']],words=len(re.findall(r'\S+',row['content'])))
        counts[name]=dict(raw_rows=len(rows),nonempty_rows=nonempty,train_overlap_rows_removed=overlaps,unique_documents=len(selected))
        groups[name]=list(selected.values())
    return [*groups['train'],*groups['control']],counts

def count_windows(tokenizer,text):
    starts=[0]+[m.start() for m in re.finditer(r'\S+',text)][1:]
    valid=0
    for offset in range(0,len(starts),256):
        items=starts[offset:offset+256]
        encoded=tokenizer.encode_batch([text[p:p+800] for p in items],add_special_tokens=False)
        for e in encoded:
            if len(e.ids)<100:return valid
            valid+=1
    return valid

def one_model(key):
    from tokenizers import Tokenizer
    docs,_=corpus();tok=Tokenizer.from_file(str(ROOT/'models'/key/'tokenizer.json'))
    rows=[];started=time.monotonic()
    for i,d in enumerate(docs):
        rows.append(dict(group=d['group'],filtered_index=d['filtered_index'],windows=count_windows(tok,d['content'])))
        if (i+1)%500==0:print('INVENTORY_PROGRESS',key,i+1,flush=True)
    result=dict(model=key,rows=rows,seconds=time.monotonic()-started)
    (OUT/('window_counts_'+key+'.json')).write_text(json.dumps(result))
    print('INVENTORY_MODEL_COMPLETE',key,len(rows),round(result['seconds'],1),flush=True)
    return result

def main():
    docs,counts=corpus()
    manifest=[{k:v for k,v in d.items() if k!='content'} for d in docs]
    (OUT/'corpus_manifest.json').write_text(json.dumps(dict(counts=counts,documents=manifest),indent=2))
    print('CORPUS_COUNTS',json.dumps(counts),flush=True)
    with ProcessPoolExecutor(max_workers=5) as pool:models=list(pool.map(one_model,KEYS))
    assert all([(r['group'],r['filtered_index']) for r in m['rows']]==[(d['group'],d['filtered_index']) for d in docs] for m in models)
    common=[]
    for i,d in enumerate(manifest):
        common.append(dict(d,windows_by_model={m['model']:m['rows'][i]['windows'] for m in models},
                           common_windows=min(m['rows'][i]['windows'] for m in models)))
    result=dict(status='complete',counts=counts,documents=common,
                window_policy='Exact original 800-character chunks, word stride 1, stop at first chunk with fewer than 100 native tokens; use the intersection across five tokenizers')
    (OUT/'corpus_inventory.json').write_text(json.dumps(result,indent=2))
    print('INVENTORY_COMPLETE',json.dumps({g:dict(documents=sum(d['group']==g and d['common_windows']>0 for d in common),windows=sum(d['common_windows'] for d in common if d['group']==g)) for g in ['train','control']}),flush=True)

if __name__=='__main__':main()
