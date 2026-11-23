"""CPU-only frozen BBC corpus reconstruction and explicit disjointness audit."""
import collections
import csv
import datetime
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from scipy.sparse import csr_matrix

ROOT=Path('/workspace/bbc-exposure')
REV='8dd1ecdc92ac43f9c04a3da3e945537dbb08179b'
OLD_SHA='8f22ff46d5dfc7d85a2c58d598c72252b754a335514e3bd02ca485a18b87e91e'


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False))


def normalized(text):
    return re.sub(r'\s+',' ',text).strip()


def words(text):
    return re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold())


def load_source():
    rows=[]
    for month in ['2025-03','2025-04','2025-05','2025-06']:
        path=hf_hub_download('RealTimeData/bbc_news_alltime',month+'/train-00000-of-00001.parquet',
               repo_type='dataset',revision=REV,cache_dir=ROOT/'cache')
        for source_i,r in enumerate(pq.read_table(path).to_pylist()):
            date=str(r.get('published_date') or '')[:10]
            try:date=datetime.date.fromisoformat(date).isoformat()
            except ValueError:continue
            if date<'2025-03-01':continue
            rows.append(dict(filtered_index=len(rows),source_config=month,source_row=source_i,
                published_date=date,title=r.get('title'),link=r.get('link'),content=r.get('content') or ''))
        print('MONTH_READY',month,len(rows),flush=True)
    old=[r for r in rows if r['filtered_index']%2==0]
    raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in old).encode()
    assert len(rows)==10643 and len(old)==5322
    assert hashlib.sha256(raw).hexdigest()==OLD_SHA,'Rebuilt old training membership/content differs'
    (ROOT/'data/old_training_articles.jsonl').write_bytes(raw)
    (ROOT/'data/frozen_all_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    return rows,old


def unique_rows(rows):
    by={}
    for r in rows:
        n=normalized(r['content'])
        if n:by.setdefault(n,r)
    return list(by.values())


def feature_matrix(doc_words):
    vocab={}
    indices=[]
    indptr=[0]
    for i,w in enumerate(doc_words):
        features={' '.join(w[j:j+5]) for j in range(max(0,len(w)-4))}
        cols=[]
        for feature in sorted(features):
            if feature not in vocab:vocab[feature]=len(vocab)
            cols.append(vocab[feature])
        indices.extend(sorted(cols))
        indptr.append(len(indices))
        if i%1000==0:print('FEATURES',i,'vocabulary',len(vocab),flush=True)
    matrix=csr_matrix((np.ones(len(indices),dtype=np.int32),np.array(indices,dtype=np.int32),
                      np.array(indptr,dtype=np.int64)),shape=(len(doc_words),len(vocab)))
    return matrix


def overlap_metrics(counts,left_sizes,right_sizes):
    # Every nonzero pair is evaluated, not an approximate nearest-neighbor shortlist.
    metrics=[]
    edges=[]
    for i in range(counts.shape[0]):
        start,end=counts.indptr[i:i+2]
        cols=counts.indices[start:end]
        intersection=counts.data[start:end].astype(np.float64)
        if len(cols):
            union=left_sizes[i]+right_sizes[cols]-intersection
            jac=intersection/np.maximum(1,union)
            containment=intersection/np.maximum(1,np.minimum(left_sizes[i],right_sizes[cols]))
            j=int(np.argmax(jac));k=int(np.argmax(containment))
            flags=(jac>=.5)|(containment>=.8)
            edges.append([int(x) for x in cols[flags]])
            metrics.append(dict(max_jaccard=float(jac[j]),max_containment=float(containment[k]),
                   best_jaccard_doc=int(cols[j]),best_containment_doc=int(cols[k]),near_duplicate=bool(flags.any())))
        else:
            edges.append([])
            metrics.append(dict(max_jaccard=0.,max_containment=0.,best_jaccard_doc=None,best_containment_doc=None,near_duplicate=False))
    return metrics,edges


class Groups:
    def __init__(self,n):self.parents=list(range(n))
    def find(self,x):
        while self.parents[x]!=x:
            self.parents[x]=self.parents[self.parents[x]];x=self.parents[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parents[b]=a


def representatives(indices,edges,doc_words,extra_edges=None):
    allowed=set(indices)
    groups=Groups(len(doc_words))
    for i in indices:
        for j in edges[i]:
            if j in allowed:groups.union(i,j)
    if extra_edges:
        for i,j in extra_edges:
            if i in allowed and j in allowed:groups.union(i,j)
    buckets=collections.defaultdict(list)
    for i in indices:buckets[groups.find(i)].append(i)
    # Retain the most complete version of each connected component, deterministic tie-break.
    result=[min(v,key=lambda i:(-len(doc_words[i]),i)) for v in buckets.values()]
    return sorted(result)


def main():
    for name in ['data','audits','protocol']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    started=time.time()
    rows,old=load_source()
    old_unique=unique_rows(old)
    old_norm={normalized(r['content']) for r in old_unique}
    odd=[r for r in rows if r['filtered_index']%2]
    clean_rows=[r for r in odd if normalized(r['content']) and normalized(r['content']) not in old_norm]
    assert len(old_unique)==4236 and len(clean_rows)==3345
    candidates=unique_rows(clean_rows)
    old_words=[words(r['content']) for r in old_unique]
    new_words=[words(r['content']) for r in candidates]
    old_folded={' '.join(w) for w in old_words}
    print('EXACT_AUDIT',len(old_unique),len(clean_rows),len(candidates),flush=True)
    all_matrix=feature_matrix(old_words+new_words)
    a=all_matrix[:len(old_unique)]
    b=all_matrix[len(old_unique):]
    old_sizes=np.diff(a.indptr).astype(np.float64)
    new_sizes=np.diff(b.indptr).astype(np.float64)
    print('COMPARING_OLD_NEW',flush=True)
    old_counts=(b@a.T).tocsr()
    metrics,_=overlap_metrics(old_counts,new_sizes,old_sizes)
    del old_counts
    print('COMPARING_NEW_NEW',flush=True)
    new_counts=(b@b.T).tocsr()
    _,edges=overlap_metrics(new_counts,new_sizes,new_sizes)
    del new_counts,all_matrix,a,b

    # Exact consecutive 50-word overlap, including boilerplate. These are literal
    # canonical word strings, so there is no MinHash/hash-collision approximation.
    old_50={}
    for i,w in enumerate(old_words):
        for j in range(max(0,len(w)-49)):old_50.setdefault(' '.join(w[j:j+50]),i)
    print('OLD_50_WORD_INDEX',len(old_50),flush=True)
    shared_old=[]
    for w in new_words:
        match=None
        for j in range(max(0,len(w)-49)):
            found=old_50.get(' '.join(w[j:j+50]))
            if found is not None:match=found;break
        shared_old.append(match)
    del old_50

    near_pool=[i for i,r in enumerate(candidates) if not metrics[i]['near_duplicate']
               and ' '.join(new_words[i]) not in old_folded]
    strict_pool=[i for i in near_pool if shared_old[i] is None]
    # Detect new-new long shared passages only in the old-disjoint strict pool.
    new_50={}
    extra=[]
    for i in strict_pool:
        w=new_words[i]
        for j in range(max(0,len(w)-49)):
            phrase=' '.join(w[j:j+50])
            previous=new_50.setdefault(phrase,i)
            if previous!=i:extra.append((i,previous))
    del new_50
    near_reps=representatives(near_pool,edges,new_words)
    strict_reps=representatives(strict_pool,edges,new_words,extra)
    eligible=[i for i in strict_reps if 200<=len(new_words[i])<=1500]

    for i,r in enumerate(candidates):
        metrics[i].update(filtered_index=r['filtered_index'],published_date=r['published_date'],
            source_config=r['source_config'],word_count=len(new_words[i]),
            folded_exact_overlap=' '.join(new_words[i]) in old_folded,
            shared_50_words_with_old=shared_old[i] is not None,
            shared_50_old_filtered_index=None if shared_old[i] is None else old_unique[shared_old[i]]['filtered_index'],
            near_clean_representative=i in set(near_reps),strict_clean_representative=i in set(strict_reps),
            eligible_200_to_1500_words=i in set(eligible))
        for key in ['best_jaccard_doc','best_containment_doc']:
            value=metrics[i][key]
            metrics[i][key]=None if value is None else old_unique[value]['filtered_index']
    with (ROOT/'audits/candidate_overlap.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    for filename,indices in [('strict_disjoint_pool.jsonl',strict_reps),('eligible_target_control_pool.jsonl',eligible),
                              ('near_duplicate_clean_pool.jsonl',near_reps)]:
        with (ROOT/'data'/filename).open('w') as f:
            for i in indices:f.write(json.dumps(candidates[i],ensure_ascii=False)+'\n')
    summary=dict(status='AUDIT_COMPLETE_TRAINING_NOT_STARTED',dataset_revision=REV,
        months=['2025-03','2025-04','2025-05','2025-06'],old_corpus_sha256=OLD_SHA,old_corpus_reproduced_exactly=True,
        date_eligible_rows=len(rows),old_training_nonempty_unique=len(old_unique),clean_odd_rows=len(clean_rows),
        unique_clean_odd_contents=len(candidates),candidate_near_duplicate_with_old=sum(x['near_duplicate'] for x in metrics),
        candidate_folded_exact_overlap=sum(x['folded_exact_overlap'] for x in metrics),
        candidate_shared_50_words_with_old=sum(x is not None for x in shared_old),
        near_clean_before_internal_dedup=len(near_pool),near_clean_unique_components=len(near_reps),
        strict_clean_before_internal_dedup=len(strict_pool),strict_clean_unique_components=len(strict_reps),
        strict_200_to_1500_word_articles=len(eligible),needed_targets=288,needed_controls=128,needed_total=416,
        enough_strict_targets_controls=len(eligible)>=416,
        eligible_by_month=dict(collections.Counter(candidates[i]['source_config'] for i in eligible)),
        definitions=dict(exact='Original whitespace-collapse SHA256-equivalent text equality; original contents preserved.',
          words='Unicode NFKC, casefold, Unicode word tokens; audit only.',
          near_duplicate='5-word shingle Jaccard >=0.5 OR intersection/min(shingle-set sizes) >=0.8.',
          strict='Also exclude any literal consecutive 50 canonical words shared with old training. Includes boilerplate.',
          internal='One longest representative per connected component under the same overlap rules.',
          evaluation_length='200 to 1500 canonical words; proposed target/control eligibility only.'),
        limitations=['Disjointness is relative to the prior fine-tuning corpus, not undocumented base pretraining.',
          'Semantic paraphrases and shorter shared passages may remain; this is an explicit lexical audit.',
          'Some candidates were earlier held-out diagnostic/evaluation articles, but were not optimizer training inputs.',
          'Strict shared-passage filtering includes boilerplate and can conservatively reject unrelated stories.',
          'No exposure assignment, model download, optimizer step, or training run has occurred.'],
        audit_elapsed_seconds=time.time()-started)
    write_json(ROOT/'audits/disjointness_summary.json',summary)
    print('AUDIT_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
