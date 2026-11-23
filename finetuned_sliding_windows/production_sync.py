import argparse,csv,fcntl,gzip,json,os,sqlite3,subprocess,time
from pathlib import Path
from production_common import RCLONE,DRIVE_CONFIG,FIELDS,atomic,connect,sha,stamp,utc

def normal_row(encoded):
    row=json.loads(encoded);row['p_z']=row['probability'];row['log_p_z']=row['log_probability'];return row

def drive(args,timeout=180):
    p=subprocess.run([RCLONE,'--config',DRIVE_CONFIG,'--log-level','ERROR',*args],capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError('Drive operation failed: '+p.stderr[-1200:])
    return p.stdout

def batch_export(run,db):
    folder=run/'export/batches';folder.mkdir(exist_ok=True)
    while True:
        db.execute('BEGIN IMMEDIATE')
        ids=[r[0] for r in db.execute("SELECT task_id FROM tasks WHERE status='done' AND batch_no IS NULL LIMIT 10000")]
        if not ids:db.commit();break
        number=db.execute('SELECT coalesce(max(batch_no),0)+1 FROM batches').fetchone()[0]
        db.execute('INSERT INTO batches(batch_no,status,rows) VALUES(?,?,?)',(number,'pending',len(ids)))
        db.executemany('UPDATE tasks SET batch_no=? WHERE task_id=?',[(number,i) for i in ids]);db.commit()
    for row in db.execute("SELECT * FROM batches WHERE status='pending'").fetchall():
        number=row['batch_no'];path=folder/f'part-{number:06d}.csv.gz';temporary=path.with_suffix('.tmp')
        with gzip.open(temporary,'wt',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
            for r in db.execute('SELECT result_json FROM tasks WHERE batch_no=? ORDER BY task_id',(number,)):writer.writerow(normal_row(r[0]))
        with temporary.open('rb') as f:os.fsync(f.fileno())
        temporary.replace(path)
        db.execute("UPDATE batches SET status='local',sha256=?,bytes=? WHERE batch_no=?",(sha(path),path.stat().st_size,number));db.commit()

def consolidate(run,db,config):
    committed=[r[0] for r in db.execute("SELECT pair_no FROM documents WHERE doc_id IN (SELECT doc_id FROM tasks GROUP BY doc_id HAVING sum(status!='done')=0) GROUP BY pair_no HAVING count(*)=2")]
    pairs=set(committed);documents=[json.loads(r['metadata']) for r in db.execute('SELECT * FROM documents ORDER BY rank') if r['pair_no'] in pairs]
    complete_ids={d['doc_id'] for d in documents};folder=run/'export/results';folder.mkdir(exist_ok=True)
    records=[]
    for model in config['expected_by_model']:
        cases=[(i,s) for i in range(1,13) for s in ['low-confidence','threshold']] if model=='llada' else [(12,'exact')]
        for mask,scheme in cases:
            path=folder/(f'{model}_mask{mask:02d}_{scheme}.csv' if model=='llada' else model+'_exact.csv')
            count=0
            with path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
                for r in db.execute("SELECT doc_id,result_json FROM tasks WHERE model=? AND mask_id=? AND scheme=? AND status='done' ORDER BY rank,window_index",(model,mask,scheme)):
                    if r['doc_id'] in complete_ids:writer.writerow(normal_row(r['result_json']));count+=1
            records.append(dict(file='results/'+path.name,rows=count,bytes=path.stat().st_size,sha256=sha(path)))
    atomic(run/'export/committed_documents.json',documents)
    total_done=db.execute("SELECT count(*) FROM tasks WHERE status='done'").fetchone()[0]
    full=total_done==config['expected_total']
    summary=dict(status='complete' if full else 'partial',finalized_utc=utc(),expected_total=config['expected_total'],
        raw_valid_rows=total_done,committed_document_pairs=len(pairs),committed_documents=len(documents),
        committed_rows=sum(r['rows'] for r in records),files=records,
        analysis_policy='Final CSVs contain only fully completed training/control pairs across all five models; immutable batches preserve all valid partial work')
    atomic(run/'export/final_manifest.json',summary)
    return summary

def sync_once(run,final=False):
    config=json.loads((run/'config.json').read_text());db=connect(run)
    with (run/'sync.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        batch_export(run,db)
        summary=consolidate(run,db,config) if final else None
        folder=run/'export/batches';remote=config['remote']
        if any(folder.iterdir()):
            drive(['copy',str(folder),remote+'/batches','--immutable','--checksum','--transfers','4'])
            drive(['check',str(folder),remote+'/batches','--one-way'])
            db.execute("UPDATE batches SET status='uploaded' WHERE status='local'");db.commit()
        state=json.loads((run/'status.json').read_text())
        state['counts']={row[0]:row[1] for row in db.execute("SELECT model,count(*) FROM tasks WHERE status='done' GROUP BY model")}
        state['total_done']=sum(state['counts'].values());state['counts_refreshed_utc']=utc()
        for name in ['controller_events.jsonl','execution_finished.json','restart_status.json','monitoring_checks.json','guard_status.json']:
            source=run/name
            if source.exists():
                target=run/'export'/name;temporary=target.with_suffix('.tmp');temporary.write_bytes(source.read_bytes());temporary.replace(target)
        uploaded=db.execute("SELECT coalesce(sum(rows),0),count(*),coalesce(sum(bytes),0) FROM batches WHERE status='uploaded'").fetchone()
        heartbeat=dict(controller=state,upload_verified_utc=utc(),uploaded_rows=uploaded[0],uploaded_batches=uploaded[1],uploaded_bytes=uploaded[2])
        atomic(run/'export/latest_status.json',heartbeat)
        from datetime import datetime,timedelta
        from zoneinfo import ZoneInfo
        local=datetime.now(ZoneInfo('Europe/Zurich'))
        lines=['# Sliding-window run status','',f'Updated: {local:%Y-%m-%d %H:%M:%S %Z}',
            f"Run: {config['run_id']}",'',f"State: {state['status']}",
            f"Valid calculations: {state.get('total_done',0):,} / {config['expected_total']:,}",
            f"Verified in Drive batches: {uploaded[0]:,}",
            f"Fully completed training/control pairs: {state.get('completed_pairs',0)} / 350",'',
            '| Model | Completed | Planned |','|---|---:|---:|']
        for key,total in config['expected_by_model'].items():lines.append(f"| {key} | {state.get('counts',{}).get(key,0):,} | {total:,} |")
        if 'estimated_remaining_hours' in state:
            eta=local+timedelta(hours=state['estimated_remaining_hours'])
            lines+=['',f"Estimated remaining: {state['estimated_remaining_hours']:.1f} hours; estimated finish {eta:%Y-%m-%d %H:%M %Z}."]
        lines+=['','LLaDA: twelve masks; ten threshold and ten low-confidence samples per mask. AR: exact. No detailed sample traces.',
            'Results are partial until RUN_COMPLETE.json appears. Final CSVs include only fully completed pairs across all five models.',
            'Automatic recovery, verified Drive uploads and a budget deadline are configured.']
        (run/'export/STATUS.md').write_text('\n'.join(lines)+'\n')
        drive(['copy',str(run/'export'),remote,'--exclude','batches/**','--checksum','--transfers','4'],timeout=600 if final else 180)
        if final:
            drive(['check',str(run/'export'),remote,'--one-way'],timeout=600)
            atomic(run/'export'/('RUN_COMPLETE.json' if summary['status']=='complete' else 'RUN_PARTIAL.json'),dict(summary,drive_hash_check_passed=True))
            marker=run/'export'/('RUN_COMPLETE.json' if summary['status']=='complete' else 'RUN_PARTIAL.json')
            drive(['copyto',str(marker),remote+'/'+marker.name,'--checksum'])
            atomic(run/'final_upload_verified.json',dict(status=summary['status'],verified_utc=utc(),remote=remote))
        atomic(run/'sync_status.json',dict(last_success_epoch=time.time(),last_success_utc=utc(),uploaded_rows=uploaded[0],uploaded_batches=uploaded[1],uploaded_bytes=uploaded[2],remote=remote))
        db.close()
        print('DRIVE_SYNC_VERIFIED',uploaded[0],'rows',uploaded[1],'batches','final='+str(final),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--loop',action='store_true');p.add_argument('--final',action='store_true');a=p.parse_args();run=Path(a.run)
    if not a.loop:sync_once(run,a.final);return
    while not (run/'SYNC_STOP').exists():
        try:sync_once(run)
        except Exception as e:
            atomic(run/'sync_error.json',dict(time=utc(),error=str(e)));print('DRIVE_SYNC_RETRY',str(e),flush=True)
        for _ in range(60):
            if (run/'SYNC_STOP').exists():break
            time.sleep(5)

if __name__=='__main__':main()
