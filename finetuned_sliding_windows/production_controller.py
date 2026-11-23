import os
os.environ.update(USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import argparse,datetime,json,multiprocessing as mp,queue,signal,subprocess,time,uuid
from pathlib import Path
from production_common import atomic,connect,result_row,utc
from production_worker import work

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    run=Path(args.run);config=json.loads((run/'config.json').read_text());db=connect(run)
    db.execute("UPDATE tasks SET status='pending',started=NULL,gpu=NULL WHERE status='inflight'");db.commit()
    context=mp.get_context('spawn');outbox=context.Queue();workers={};last_status=0;first=time.time();stop=False
    events=(run/'controller_events.jsonl').open('a',buffering=1)
    def event(name,**kw):events.write(json.dumps(dict(time=utc(),event=name,**kw))+'\n')
    def spawn(gpu,key):
        generation=uuid.uuid4().hex;inbox=context.Queue(maxsize=1)
        p=context.Process(target=work,args=(str(run),gpu,key,generation,inbox,outbox));p.start()
        workers[gpu]=dict(process=p,inbox=inbox,model=key,generation=generation,ready=False,task=None,spawned=time.time(),ending=False)
        event('worker_started',gpu=gpu,model=key,pid=p.pid)
    def dispose(gpu):
        w=workers[gpu];p=w['process']
        if p.is_alive():p.terminate();p.join(5)
        if p.is_alive():p.kill();p.join(5)
        w['inbox'].close()
    def current_wave():
        r=db.execute("SELECT min(wave) FROM tasks WHERE model='llada' AND status!='done'").fetchone()
        return r[0]
    def update_status():
        nonlocal last_status
        now=time.time();counts={r[0]:r[1] for r in db.execute("SELECT model,count(*) FROM tasks WHERE status='done' GROUP BY model")}
        pending=db.execute("SELECT coalesce(sum(est_seconds),0) FROM tasks WHERE status!='done'").fetchone()[0]
        done_llada=db.execute("SELECT coalesce(sum(seconds),0),coalesce(sum(est_seconds),0),count(*) FROM tasks WHERE status='done' AND model='llada'").fetchone()
        factor=done_llada[0]/done_llada[1] if done_llada[1] else 1.
        complete_docs=[r[0] for r in db.execute("SELECT d.doc_id FROM documents d JOIN tasks t USING(doc_id) GROUP BY d.doc_id HAVING sum(t.status='done')=d.expected")]
        complete_pairs=db.execute("SELECT count(*) FROM (SELECT d.pair_no FROM documents d WHERE d.doc_id IN (SELECT doc_id FROM tasks GROUP BY doc_id HAVING sum(status!='done')=0) GROUP BY d.pair_no HAVING count(*)=2)").fetchone()[0]
        resources=[]
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10)
            for line in raw.splitlines():
                vals=[x.strip() for x in line.split(',')];resources.append(dict(gpu=int(vals[0]),utilization=float(vals[1]),memory_mib=float(vals[2]),temperature_c=float(vals[3]),power_w=float(vals[4])))
        except Exception:pass
        status=dict(status='running',run_id=config['run_id'],updated_utc=utc(),controller_pid=os.getpid(),
            total_done=sum(counts.values()),expected_total=config['expected_total'],counts=counts,expected_by_model=config['expected_by_model'],
            completed_documents=len(complete_docs),completed_pairs=complete_pairs,completed_waves=complete_pairs//4,
            active_wave=current_wave(),llada_measured_gpu_seconds=done_llada[0],llada_rows=done_llada[2],
            speed_ratio_to_pilot=factor,estimated_remaining_hours=pending*factor/6.52830191152408/3600,
            worker_models={str(g):w['model'] for g,w in workers.items()},gpu_resources=resources,
            failed_task_count=db.execute("SELECT count(*) FROM tasks WHERE status='failed'").fetchone()[0],
            retried_tasks=db.execute('SELECT count(*) FROM tasks WHERE attempts>1').fetchone()[0],
            soft_deadline_utc=datetime.datetime.fromtimestamp(config['soft_deadline'],datetime.timezone.utc).isoformat(),
            hard_deadline_utc=datetime.datetime.fromtimestamp(config['hard_deadline'],datetime.timezone.utc).isoformat())
        atomic(run/'status.json',status)
        with (run/'health_metrics.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
        last_status=now
    for gpu,key in enumerate(['llama2_7b','llama3_8b','mistral7b_v01','olmo7b','llada','llada','llada']):
        if key!='llada' and not db.execute("SELECT 1 FROM tasks WHERE model=? AND status!='done' LIMIT 1",(key,)).fetchone():key='llada'
        spawn(gpu,key)
    try:
        while True:
            now=time.time()
            if now>=config['compute_deadline'] or (run/'REQUEST_STOP_NOW').exists():event('compute_deadline');return 20
            wave=current_wave()
            if wave is None:
                if db.execute("SELECT count(*) FROM tasks WHERE status!='done'").fetchone()[0]==0:
                    update_status();event('all_tasks_complete');return 0
            soft=now>=config['soft_deadline'] or (run/'STOP_AFTER_WAVE').exists()
            if soft and wave is not None and not db.execute("SELECT 1 FROM tasks WHERE model='llada' AND wave=? AND status IN ('done','inflight') LIMIT 1",(wave,)).fetchone():
                event('soft_budget_stop',next_wave=wave);return 20
            try:msg=outbox.get(timeout=.2)
            except queue.Empty:msg=None
            if msg:
                gpu=msg['gpu'];w=workers.get(gpu)
                if not w or msg['generation']!=w['generation']:continue
                if msg['event']=='ready':w['ready']=True;event('worker_ready',gpu=gpu,model=w['model'])
                elif msg['event']=='result':
                    task=w['task'];assert task and task['task_id']==msg['task_id']
                    row=result_row(task,msg['result'],gpu)
                    db.execute("UPDATE tasks SET status='done',result_json=?,seconds=?,error=NULL WHERE task_id=? AND status='inflight'",(json.dumps(row),row['seconds'],task['task_id']));db.commit()
                    w['task']=None
                elif msg['event']=='error':
                    task=w['task'];event('worker_error',gpu=gpu,model=w['model'],task_id=msg.get('task_id'),error=msg['error'])
                    if task:
                        attempts=db.execute('SELECT attempts FROM tasks WHERE task_id=?',(task['task_id'],)).fetchone()[0]
                        state='failed' if attempts>=config['max_task_attempts'] else 'pending'
                        db.execute('UPDATE tasks SET status=?,error=? WHERE task_id=?',(state,msg['error'],task['task_id']));db.commit()
                        if state=='failed':raise RuntimeError('A task failed after three deterministic retries; see controller_events.jsonl')
                    dispose(gpu);spawn(gpu,w['model'])
                elif msg['event']=='finished':
                    w['process'].join(15)
                    if w['model']!='llada':dispose(gpu);spawn(gpu,'llada')
                    else:w['ending']=True
            for gpu,w in list(workers.items()):
                if w['ending']:continue
                p=w['process']
                if not p.is_alive():
                    task=w['task'];event('worker_exited',gpu=gpu,exitcode=p.exitcode)
                    if task:
                        db.execute("UPDATE tasks SET status='pending',error='worker exited' WHERE task_id=? AND status='inflight'",(task['task_id'],));db.commit()
                    dispose(gpu);spawn(gpu,w['model']);continue
                if not w['ready'] and now-w['spawned']>180:dispose(gpu);spawn(gpu,w['model']);continue
                if w['task']:
                    if now-w['task']['started']>config['task_timeout_seconds']:
                        task=w['task'];dispose(gpu)
                        db.execute("UPDATE tasks SET status='pending',error='task timeout' WHERE task_id=?",(task['task_id'],));db.commit();spawn(gpu,w['model'])
                    continue
                if not w['ready']:continue
                if w['model']=='llada':
                    task=db.execute("SELECT * FROM tasks WHERE model='llada' AND status='pending' AND wave=? ORDER BY rank,window_index,mask_id,scheme LIMIT 1",(wave,)).fetchone() if wave is not None else None
                else:task=db.execute("SELECT * FROM tasks WHERE model=? AND status='pending' ORDER BY rank,window_index LIMIT 1",(w['model'],)).fetchone()
                if task:
                    task=dict(task)
                    if task['attempts']>=config['max_task_attempts']:raise RuntimeError('Task retry limit reached')
                    task['started']=time.time();db.execute("UPDATE tasks SET status='inflight',attempts=attempts+1,started=?,gpu=? WHERE task_id=?",(task['started'],gpu,task['task_id']));db.commit()
                    w['task']=task;w['inbox'].put(task)
                elif w['model']!='llada':w['inbox'].put(None);w['ending']=True
            # Finished AR workers join the shared LLaDA pool after releasing their model.
            for gpu,w in list(workers.items()):
                if w['ending'] and w['model']!='llada' and not w['process'].is_alive():dispose(gpu);spawn(gpu,'llada')
            if now-last_status>=30:update_status()
            sync=run/'sync_status.json'
            if sync.exists():
                record=json.loads(sync.read_text())
                if now-record.get('last_success_epoch',first)>3600:raise RuntimeError('Drive uploads unavailable for an hour; stopping for recovery')
    finally:
        for gpu in list(workers):dispose(gpu)
        db.close();events.close()

if __name__=='__main__':raise SystemExit(main())
