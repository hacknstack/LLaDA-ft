"""Short, source-stratified timing coverage after the large-budget probes drain."""
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pilot_controller import ROOT,atomic,task,phase,DATA


def main():
    while not (ROOT/'results/PRIMARY_PILOT_COMPLETE.json').exists():time.sleep(2)
    cohorts=json.loads((ROOT/'input_cohorts.json').read_text())
    jobs=[]
    selection=[]
    for cohort in ('application','negative','validation'):
        documents=defaultdict(list)
        for row in cohorts[cohort]:documents[row['document']].append(row)
        for index,(document,rows) in enumerate(sorted(documents.items())):
            mask_id=index%12+1
            pool=[r for r in rows if r['mask']==DATA['labels'][mask_id-1]]
            row=dict(min(pool,key=lambda r:hashlib.sha256((cohort+document+r['window_number']).encode()).hexdigest()))
            row['mask_id']=mask_id
            weight=len(rows)/len(cohorts[cohort])
            selection.append(dict(cohort=cohort,source_weight=weight,**row))
            for temperature in (.5,2.):
                for scheme in ('low-confidence','threshold'):
                    jobs.append(dict(tasks=[task(row,'path_sampling',scheme,temperature,10,
                                      purpose='coverage_'+cohort,source_weight=weight)]))
                    if cohort=='validation':
                        jobs.append(dict(tasks=[task(row,'monte-carlo',scheme,temperature,125,
                                          purpose='coverage_validation_mc',source_weight=weight)]))
    atomic(ROOT/'results/coverage_selection.json',selection)
    gpus=(0,1,2,3,5,6,7)
    # Use the remaining slice for shape/precision microbenchmarks concurrently.
    log=open(ROOT/'pilot_batch_shapes.log','w')
    batch=subprocess.Popen([str(ROOT/'venv/bin/python'),'-u',str(ROOT/'code/pilot_batch_shapes.py')],
          stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
          env=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='1'))
    processes=[]
    for gpu in gpus:
        folder=ROOT/'worker_io'/str(gpu)
        for name in ('ready.json','result.json','job.json'):(folder/name).unlink(missing_ok=True)
        log=open(folder/'coverage_worker.log','w')
        p=subprocess.Popen([str(ROOT/'venv/bin/python'),'-u',str(ROOT/'code/pilot_worker.py'),'--gpu',str(gpu)],
             stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
             env=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='1'))
        processes.append(p)
    start=time.time()
    while not all((ROOT/'worker_io'/str(gpu)/'ready.json').exists() for gpu in gpus):
        if time.time()-start>180:raise TimeoutError('Coverage worker startup')
        time.sleep(1)
    try:phase('source_stratified_coverage',jobs,gpus=gpus)
    finally:
        for gpu in gpus:atomic(ROOT/'worker_io'/str(gpu)/'job.json',dict(stop=True))
        for process in processes:process.wait(timeout=120)
    print('BATCH_AUDIT_EXIT',batch.wait(timeout=600),flush=True)
    atomic(ROOT/'results/PILOT_COMPLETE.json',dict(status='benchmark_complete',completed=time.time(),
           full_experiment_started=False,coverage_jobs=len(jobs)))
    print('COVERAGE_PILOT_COMPLETE',flush=True)


if __name__=='__main__':main()
