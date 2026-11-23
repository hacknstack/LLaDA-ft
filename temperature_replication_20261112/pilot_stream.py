"""Work-conserving bounded benchmark; adopts the running standalone cache check."""
import hashlib
import json
import time
from collections import Counter
from pilot_controller import ROOT, DATA, atomic, task


def jobs():
    output=[]
    row=DATA['selected']['exact'][7]
    paired=[task(row,method,scheme,temp,500,reduced=True,purpose='cache_check')
            for method in ('exact','path_sampling')
            for temp,scheme in [(.5,'low-confidence'),(.5,'threshold'),(2.,'low-confidence'),
                                (2.,'threshold'),(1.,'threshold')]]
    output.append(dict(id='stream_cache_enabled',tasks=paired,shared_cache=True))
    rest=[]
    for row in DATA['selected']['validation']:
        for temp in (.5,2.):
            for scheme in ('low-confidence','threshold'):
                for method,samples in [('path_sampling',500),('monte-carlo',500)]:
                    rest.append(dict(tasks=[task(row,method,scheme,temp,samples,purpose='validation')]))
    for row in DATA['selected']['exact']:
        group=[task(row,method,scheme,temp,500,reduced=True,purpose='exact_shared')
               for method in ('exact','path_sampling')
               for temp,scheme in [(.5,'low-confidence'),(.5,'threshold'),(2.,'low-confidence'),
                                   (2.,'threshold'),(1.,'threshold')]]
        rest.append(dict(tasks=group,shared_cache=True))
    for cohort in ('application','negative'):
        for row in DATA['selected'][cohort]:
            for temp in (.5,2.):
                for scheme in ('low-confidence','threshold'):
                    rest.append(dict(tasks=[task(row,'path_sampling',scheme,temp,10,purpose=cohort)]))
    for row in DATA['selected']['runtime'][::3]:
        for temp in (.5,2.):
            for scheme in ('low-confidence','threshold'):
                for method,budgets in [('path_sampling',(62,125,250,500,1000)),
                                       ('monte-carlo',(125,250,500,1250,2500,5000))]:
                    for budget in budgets:
                        rest.append(dict(tasks=[task(row,method,scheme,temp,budget,timing=True,purpose='runtime')]))
    for row in DATA['selected']['validation'][::6]:
        for temp in (.5,2.):
            for batch in (1,8,32,64):
                rest.append(dict(tasks=[task(row,'random','random',temp,64,batch_size=batch,selected=True,purpose='random')]))
            rest.append(dict(tasks=[task(row,'greedy','greedy',temp,1,purpose='greedy')]))
    rest.sort(key=lambda j:hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest())
    for index,job in enumerate(rest):
        job['id']=f'stream_{index:04}'
    return output+rest


def main():
    if (ROOT/'results/runtime_study_excluded.json').exists():
        raise RuntimeError('Historical pilot controller disabled: it includes the excluded runtime study.')
    waiting=jobs()
    # Worker 0 is finishing the existing singleton-forward baseline.
    active={0:dict(id='cache_baseline-0',start=time.time())}
    completed=[]
    counts=Counter()
    start=time.time()
    deadline=start+2400  # Bounded pilot, not authorization for the full experiment.
    cache_baseline=cache_enabled=None
    while active or (waiting and time.time()<deadline):
        for gpu in range(8):
            folder=ROOT/'worker_io'/str(gpu)
            result=folder/'result.json'
            if gpu in active and result.exists():
                value=json.loads(result.read_text())
                assert value['job_id']==active[gpu]['id'], (gpu,value['job_id'],active[gpu])
                value['observed_job_seconds']=time.time()-active[gpu]['start']
                value['gpu']=gpu
                completed.append(value)
                with (ROOT/'results/stream_jobs.jsonl').open('a') as handle:
                    handle.write(json.dumps(value)+'\n')
                if 'error' in value:
                    print('ERROR',value['job_id'],value['error'][-400:],flush=True)
                else:
                    for row in value['rows']:counts[row.get('purpose','baseline')]+=1
                if value['job_id']=='cache_baseline-0':cache_baseline=value
                if value['job_id']=='stream_cache_enabled':cache_enabled=value
                del active[gpu]
                result.unlink()
                print('PROGRESS',len(completed),'REMAIN',len(waiting),dict(counts),flush=True)
            if gpu not in active and waiting and time.time()<deadline:
                job=waiting.pop(0)
                result.unlink(missing_ok=True)
                atomic(folder/'job.json',job)
                active[gpu]=dict(id=job['id'],start=time.time())
        if cache_baseline is not None and cache_enabled is not None and not (ROOT/'results/cache_validation.json').exists():
            assert 'error' not in cache_baseline and 'error' not in cache_enabled
            for a,b in zip(cache_baseline['rows'],cache_enabled['rows']):
                assert a['probability']==b['probability'] and a.get('log_probability')==b.get('log_probability')
            atomic(ROOT/'results/cache_validation.json',dict(bit_identical=True,
                baseline_seconds=sum(r['seconds'] for r in cache_baseline['rows']),
                cached_seconds=sum(r['seconds'] for r in cache_enabled['rows']),cache=cache_enabled['cache']))
        atomic(ROOT/'results/pilot_status.json',dict(phase='streaming_representative',completed_jobs=len(completed),
              queued_jobs=len(waiting),active_jobs={str(k):v['id'] for k,v in active.items()},
              rows_by_purpose=dict(counts),elapsed_seconds=time.time()-start,updated=time.time()))
        if any(time.time()-v['start']>1500 for v in active.values()):
            raise TimeoutError('Pilot task exceeded 25-minute bound')
        time.sleep(.1)
    atomic(ROOT/'results/PILOT_COMPLETE.json',dict(completed=time.time(),full_experiment_started=False,
           jobs_completed=len(completed),jobs_not_run=len(waiting),rows_by_purpose=dict(counts),elapsed_seconds=time.time()-start))
    print('PILOT_COMPLETE',flush=True)


if __name__=='__main__':
    try:
        main()
    finally:
        for gpu in range(8):atomic(ROOT/'worker_io'/str(gpu)/'job.json',dict(stop=True))
