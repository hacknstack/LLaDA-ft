"""Short calibration sweeps followed by representative, full-budget measurements."""
import json
import os
import signal
import time
from pathlib import Path
from pilot_controller import ROOT, DATA, EVENTS, atomic, task, phase


def main():
    if (ROOT/'results/runtime_study_excluded.json').exists():
        raise RuntimeError('Historical pilot controller disabled: it includes the excluded runtime study.')
    # Use short tasks for repeated throughput calibration; full budgets follow.
    row = DATA['selected']['validation'][11]
    calibration = [task(row,'path_sampling','low-confidence',.5,10),
                   task(row,'path_sampling','threshold',2.,10),
                   task(row,'monte-carlo','low-confidence',2.,100)]
    phase('solo_calibration_v3',[dict(tasks=calibration)],gpus=(0,))
    for threads in (3,2,4):
        phase(f'eight_gpu_threads{threads}',
              [dict(tasks=[dict(t,threads=threads) for t in calibration]) for _ in range(8)])

    row = DATA['selected']['exact'][7]
    paired = [task(row,method,scheme,temp,500,reduced=True)
              for method in ('exact','path_sampling')
              for temp,scheme in [(.5,'low-confidence'),(.5,'threshold'),(2.,'low-confidence'),
                                  (2.,'threshold'),(1.,'threshold')]]
    baseline = phase('cache_baseline',[dict(tasks=paired)],gpus=(0,))
    cached = phase('cache_enabled',[dict(tasks=paired,shared_cache=True)],gpus=(0,))
    assert 'error' not in baseline[0] and 'error' not in cached[0]
    for a,b in zip(baseline[0]['rows'],cached[0]['rows']):
        assert a['probability'] == b['probability'] and a.get('log_probability') == b.get('log_probability')
    atomic(ROOT/'results/cache_validation.json',dict(bit_identical=True,
           baseline_seconds=sum(r['seconds'] for r in baseline[0]['rows']),
           cached_seconds=sum(r['seconds'] for r in cached[0]['rows']),cache=cached[0]['cache']))

    jobs=[]
    for row in DATA['selected']['validation']:
        for temp in (.5,2.):
            for scheme in ('low-confidence','threshold'):
                for method,samples in [('path_sampling',500),('monte-carlo',500)]:
                    jobs.append(dict(tasks=[task(row,method,scheme,temp,samples)]))
    phase('validation_representative',jobs)

    jobs=[]
    for row in DATA['selected']['exact']:
        group=[task(row,method,scheme,temp,500,reduced=True)
               for method in ('exact','path_sampling')
               for temp,scheme in [(.5,'low-confidence'),(.5,'threshold'),(2.,'low-confidence'),
                                   (2.,'threshold'),(1.,'threshold')]]
        jobs.append(dict(tasks=group,shared_cache=True))
    phase('exact_representative_shared',jobs)

    jobs=[]
    for cohort in ('application','negative'):
        for row in DATA['selected'][cohort]:
            for temp in (.5,2.):
                for scheme in ('low-confidence','threshold'):
                    jobs.append(dict(tasks=[task(row,'path_sampling',scheme,temp,10,cohort=cohort)]))
    phase('ten_sample_application_controls',jobs)

    jobs=[]
    for row in DATA['selected']['runtime'][::3]:
        for temp in (.5,2.):
            for scheme in ('low-confidence','threshold'):
                for method,budgets in [('path_sampling',(62,125,250,500,1000)),
                                       ('monte-carlo',(125,250,500,1250,2500,5000))]:
                    for budget in budgets:
                        jobs.append(dict(tasks=[task(row,method,scheme,temp,budget,timing=True)]))
    phase('runtime_budget_sweep',jobs)

    jobs=[]
    for row in DATA['selected']['validation'][::6]:
        for temp in (.5,2.):
            for batch in (1,8,32,64):
                jobs.append(dict(tasks=[task(row,'random','random',temp,64,batch_size=batch,selected=True)]))
            jobs.append(dict(tasks=[task(row,'greedy','greedy',temp,1)]))
    phase('other_schemes',jobs)
    atomic(ROOT/'results/PILOT_COMPLETE.json',dict(completed=time.time(),full_experiment_started=False))
    print('PILOT_COMPLETE',flush=True)


if __name__=='__main__':
    try:
        main()
    finally:
        for gpu in range(8):
            atomic(ROOT/'worker_io'/str(gpu)/'job.json',dict(stop=True))
