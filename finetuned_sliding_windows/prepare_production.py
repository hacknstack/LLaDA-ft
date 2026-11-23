import os
os.environ.update(USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
import bisect,collections,csv,datetime,hashlib,json,shutil,sqlite3,zipfile
from pathlib import Path
from production_common import ROOT,atomic,connect,seed_for,sha,stamp,utc
from planning_inventory import corpus
from planning_speed_pilot import MASKS

def main():
    import sliding_window_extraction as extraction
    planning=ROOT/'planning_20261101'
    approved=planning/'documents_recommended.csv'
    selection_sha=sha(approved)
    assert selection_sha=='1bc1b0b2aec4d5ee0ea49e74043d9814d77f4a1a9e6203c532252a3d2c78b07a'
    chosen=list(csv.DictReader(approved.open()))
    assert collections.Counter(r['group'] for r in chosen)=={'train':350,'control':350}
    metadata=json.loads((planning/'budget_plan.json').read_text())
    edges=metadata['length_bin_edges_words'];cells=collections.defaultdict(lambda:dict(train=[],control=[]))
    for r in chosen:
        for k in ['filtered_index','words','common_windows']:r[k]=int(r[k])
        r['selected_windows']=json.loads(r['selected_windows']);r['original_filtered_indices']=json.loads(r['original_filtered_indices'])
        cells[r['source_config'],bisect.bisect_right(edges,r['words'])][r['group']].append(r)
    pairs=[];offset=0
    while True:
        added=0
        for cell in sorted(cells):
            group=cells[cell];assert len(group['train'])==len(group['control'])
            if offset<len(group['train']):pairs.append((group['train'][offset],group['control'][offset]));added+=1
        if not added:break
        offset+=1
    run=ROOT/'runs'/('bbc_12masks_'+stamp());export=run/'export';export.mkdir(parents=True)
    db=connect(run)
    db.executescript('''
    CREATE TABLE documents(doc_id TEXT PRIMARY KEY,doc_group TEXT,filtered_index INTEGER,pair_no INTEGER,wave INTEGER,rank INTEGER,expected INTEGER,metadata TEXT);
    CREATE TABLE inputs(input_id TEXT,model TEXT,token_ids TEXT,PRIMARY KEY(input_id,model));
    CREATE TABLE tasks(task_id TEXT PRIMARY KEY,model TEXT,doc_id TEXT,input_id TEXT,doc_group TEXT,filtered_index INTEGER,window_index INTEGER,char_start INTEGER,mask_id INTEGER,scheme TEXT,seed INTEGER,wave INTEGER,rank INTEGER,est_seconds REAL,status TEXT DEFAULT 'pending',attempts INTEGER DEFAULT 0,started REAL,gpu INTEGER,result_json TEXT,seconds REAL,batch_no INTEGER,error TEXT);
    CREATE INDEX tasks_dispatch ON tasks(model,status,wave,rank,window_index,mask_id,scheme);
    CREATE INDEX tasks_ar_dispatch ON tasks(model,status,rank,window_index);
    CREATE INDEX tasks_documents ON tasks(doc_id,status);
    CREATE INDEX tasks_wave ON tasks(model,wave,status);
    CREATE INDEX tasks_export ON tasks(status,batch_no);
    CREATE TABLE batches(batch_no INTEGER PRIMARY KEY,status TEXT,rows INTEGER,sha256 TEXT,bytes INTEGER);
    ''')
    models=json.loads((ROOT/'verification/verify_20261101T155551Z/models_manifest.json').read_text())
    for spec in models:
        folder=Path(spec['local_path']);assert sha(folder/'COMPLETE.json')==spec['complete_sha256']
        marker=json.loads((folder/'COMPLETE.json').read_text());assert marker['epoch']==3
        for n,v in marker['files'].items():assert (folder/n).stat().st_size==v['bytes']
    documents,_=corpus();lookup={(r['group'],r['filtered_index']):r for r in documents}
    assert sha(ROOT/'code/probabilistic_extraction.py')=='3fa3d3991237c6960a2aab950f71efb0df2c279436d96769f63f914b9a1705f4'
    timing=list(csv.DictReader((planning/'timings_by_mask.csv').open()))
    times={(r['group'],r['mask'],r['scheme']):float(r['mean_seconds']) for r in timing}
    specs={s['key']:s for s in models}
    tokenizers={key:extraction._load_tokenizer(spec['local_path']) for key,spec in specs.items()}
    import re
    records=[];token_rows=[];tasks=[]
    for pair_no,pair in enumerate(pairs):
        for side,r in enumerate(pair):
            rank=2*pair_no+side;doc_id=f"{r['group']}:{r['filtered_index']}";wave=pair_no//4
            source=lookup[r['group'],r['filtered_index']];text=source['content']
            assert source['normalized_sha256']==r['normalized_sha256']
            starts=[0]+[m.start() for m in re.finditer(r'\S+',text)][1:]
            snippets=[text[starts[w]:starts[w]+800] for w in r['selected_windows']]
            encoded={k:t(snippets,add_special_tokens=False)['input_ids'] for k,t in tokenizers.items()}
            record=dict(r,doc_id=doc_id,pair_no=pair_no,wave=wave,rank=rank,source_row=source['source_row'],content_sha256=hashlib.sha256(text.encode()).hexdigest())
            records.append(record)
            db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)',(doc_id,r['group'],r['filtered_index'],pair_no,wave,rank,len(r['selected_windows'])*28,json.dumps(record)))
            for j,w in enumerate(r['selected_windows']):
                input_id=f'{doc_id}:w{w}'
                for key in specs:
                    ids=encoded[key][j][:100];assert len(ids)==100
                    token_rows.append((input_id,key,json.dumps(ids)))
                    cases=[(i,name,mask,scheme) for i,(name,mask) in enumerate(MASKS,1) for scheme in ['low-confidence','threshold']] if key=='llada' else [(12,'suffix50',None,'exact')]
                    for mask_id,name,mask,scheme in cases:
                        seed=seed_for(r['group'],r['filtered_index'],w,mask_id,scheme) if key=='llada' else 0
                        task_id=hashlib.sha256(f'{selection_sha}|{key}|{specs[key]["complete_sha256"]}|{input_id}|{mask_id}|{scheme}|{seed}'.encode()).hexdigest()
                        estimate=times[r['group'],name,scheme] if key=='llada' else metadata['ar_seconds_per_window_by_model'][key]
                        tasks.append((task_id,key,doc_id,input_id,r['group'],r['filtered_index'],w,starts[w],mask_id,scheme,seed,wave,rank,estimate))
            if (rank+1)%100==0:print('PRODUCTION_INPUTS_PREPARED',rank+1,flush=True)
    db.executemany('INSERT INTO inputs VALUES(?,?,?)',token_rows)
    db.executemany('INSERT INTO tasks(task_id,model,doc_id,input_id,doc_group,filtered_index,window_index,char_start,mask_id,scheme,seed,wave,rank,est_seconds) VALUES('+','.join('?'*14)+')',tasks)
    db.commit()
    counts={row[0]:row[1] for row in db.execute('SELECT model,count(*) FROM tasks GROUP BY model')}
    assert counts['llada']==133968 and all(counts[k]==5582 for k in specs if k!='llada')
    origin=datetime.datetime(2026,11,1,16,26,37,tzinfo=datetime.timezone.utc)
    config=dict(run_id=run.name,created_utc=utc(),remote='researchdrive:LLaDA-ft/sliding-window-runs/'+run.name,
        rate_per_hour=6.4,balance_at_origin=490,budget_origin_utc=origin.isoformat(),
        soft_deadline=(origin+datetime.timedelta(hours=68.75)).timestamp(),
        compute_deadline=(origin+datetime.timedelta(hours=74.5)).timestamp(),hard_deadline=(origin+datetime.timedelta(hours=75)).timestamp(),
        models=models,mask_definitions=MASKS,expected_by_model=counts,expected_total=sum(counts.values()),expected_documents=700,
        selection_sha256=sha(approved),dataset_revision='8dd1ecdc92ac43f9c04a3da3e945537dbb08179b',
        dataset_sha256={p.name:sha(p) for p in (ROOT/'assets').glob('*.jsonl')},
        estimator_sha256=sha(ROOT/'code/probabilistic_extraction.py'),wrapper_sha256=sha(ROOT/'code/sliding_window_extraction.py'),
        settings=dict(sequence_tokens=100,prefix_tokens=50,suffix_tokens=50,chunk_chars=800,stride_words=1,
          decoding_scheme='full',temperature=1.,tau=.001,confidence_threshold=.9,num_samples_per_llada_scheme=10,
          llada_mode='path_sampling',ar_mode='exact',model_dtype='bfloat16',llada_estimator_dtype='float64',
          model_forward_batch_size=1,cpu_threads_per_worker=4,detailed=False,verbose=False,master_seed=20261101),
        gpu_count=7,upload_interval_seconds=300,task_timeout_seconds=300,max_task_attempts=3,
        automatic_shutdown=True,pod_id=os.environ['RUNPOD_POD_ID'])
    atomic(run/'config.json',config);atomic(export/'config.json',config)
    atomic(export/'documents.json',records);shutil.copy2(approved,export/'documents.csv')
    with zipfile.ZipFile(export/'code_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT/'code').glob('*.py'):z.write(p,p.name)
    atomic(run/'status.json',dict(status='prepared',updated_utc=utc(),counts={},total_done=0,expected_total=config['expected_total']))
    (ROOT/'current_production_run.txt').write_text(str(run))
    print('PRODUCTION_PREPARED',run,counts,flush=True)

if __name__=='__main__':main()
