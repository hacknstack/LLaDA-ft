"""Freeze disjoint article identities and an exact shared occurrence schedule."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT=Path('/workspace/bbc-exposure')
LEVELS=[1,2,4,8,16,32,64,128,256]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def wc(text):
    return len(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))


def main():
    out=ROOT/'data/frozen_exposure'
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text())
        for name,h in manifest['file_sha256'].items():assert sha(out/name)==h
        print('FROZEN_CORPUS_ALREADY_VERIFIED',flush=True)
        return
    out.mkdir(exist_ok=True)
    strict=[json.loads(l) for l in (ROOT/'data/strict_disjoint_pool.jsonl').open()]
    eligible=[json.loads(l) for l in (ROOT/'data/eligible_target_control_pool.jsonl').open()]
    assert len(strict)==1950 and len(eligible)==1540
    rng=random.Random(42)
    roles={}
    matched=[]
    # Eight 13-article matched sets per month: nine doses and four held-out controls.
    for month in ['2025-03','2025-04','2025-05','2025-06']:
        pool=sorted([r for r in eligible if r['source_config']==month],key=lambda r:r['filtered_index'])
        vector=TfidfVectorizer(stop_words='english',max_features=16000,min_df=2,sublinear_tf=True)
        features=vector.fit_transform([r['content'] for r in pool])
        similarities=(features@features.T).toarray()
        lengths=np.array([wc(r['content']) for r in pool])
        available=set(range(len(pool)))
        for stratum in range(8):
            ordered=sorted(available,key=lambda i:(lengths[i],pool[i]['filtered_index']))
            position=round((stratum+.5)/8*(len(ordered)-1))
            anchor=ordered[position]
            # Match only observed covariates. No extraction or model-loss outcomes are used.
            distances=1-similarities[anchor]+2*np.abs(np.log(lengths/lengths[anchor]))
            chosen=sorted(available,key=lambda i:(distances[i],pool[i]['filtered_index']))[:13]
            assert len(chosen)==13
            available.difference_update(chosen)
            assignments=LEVELS+[0]*4
            rng.shuffle(assignments)
            set_id=month+':'+str(stratum)
            for i,exposure in zip(chosen,assignments):
                row=pool[i]
                roles[row['filtered_index']]=dict(role='control' if exposure==0 else 'target',exposures=exposure,matched_set=set_id)
            matched.append(dict(id=set_id,anchor=pool[anchor]['filtered_index'],members=[pool[i]['filtered_index'] for i in chosen],
                      min_words=int(lengths[chosen].min()),max_words=int(lengths[chosen].max())))
    articles=[]
    for row in strict:
        identity=row['filtered_index']
        role=roles.get(identity,dict(role='background',exposures=3,matched_set=None))
        articles.append(dict(row,article_id='bbc_'+str(identity),content_sha256=hashlib.sha256(row['content'].encode()).hexdigest(),
                             word_count=wc(row['content']),**role))
    counts=collections.Counter((r['role'],r['exposures']) for r in articles)
    assert counts[('control',0)]==128 and counts[('background',3)]==1534
    assert all(counts[('target',e)]==32 for e in LEVELS)
    # Independent jitter inside each article's equal-sized exposure strata spreads repeats.
    schedule=[]
    for row in articles:
        e=row['exposures']
        for ordinal in range(e):
            schedule.append(dict(article_id=row['article_id'],occurrence=ordinal+1,planned_position=(ordinal+rng.random())/e))
    schedule.sort(key=lambda r:(r['planned_position'],r['article_id']))
    actual=collections.Counter(r['article_id'] for r in schedule)
    assert all(actual[r['article_id']]==r['exposures'] for r in articles)
    assert len(schedule)==20954
    for i,r in enumerate(schedule):r['stream_index']=i
    with (out/'articles.jsonl').open('w') as f:
        for r in articles:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (out/'occurrences.jsonl').open('w') as f:
        for r in schedule:f.write(json.dumps(r)+'\n')
    header=['article_id','filtered_index','source_config','published_date','role','exposures','matched_set','word_count','content_sha256']
    with (out/'membership.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=header,extrasaction='ignore');writer.writeheader();writer.writerows(articles)
    groups=[]
    for exposure in [0]+LEVELS:
        rows=[r for r in articles if r['role']!='background' and r['exposures']==exposure]
        groups.append(dict(exposures=exposure,articles=len(rows),mean_words=float(np.mean([r['word_count'] for r in rows])),
                           min_words=min(r['word_count'] for r in rows),max_words=max(r['word_count'] for r in rows),
                           by_month=dict(collections.Counter(r['source_config'] for r in rows))))
    manifest=dict(status='FROZEN_PREPARATION_ONLY',seed=42,unique_articles=len(articles),targets=288,controls=128,background=1534,
       target_occurrences=16352,background_occurrences=4602,total_occurrences=len(schedule),groups=groups,matched_sets=matched,
       matching='32 sets of 13; 8 per month. TF-IDF cosine distance + 2*absolute log word-length ratio; random doses within each set.',
       occurrence_order='Per-article exposure strata with independent uniform jitter; global stable sort; no epoch/block reshuffle.',
       definitions='Counts are complete input presentations, not masked-loss or next-token-target counts.',
       max_to_min_target_mean_word_ratio=max(g['mean_words'] for g in groups[1:])/min(g['mean_words'] for g in groups[1:]),
       file_sha256={name:sha(out/name) for name in ['articles.jsonl','occurrences.jsonl','membership.csv']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['matched_sets','groups']},indent=2),flush=True)


if __name__=='__main__':main()
