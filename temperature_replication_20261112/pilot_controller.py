import hashlib
import json
import time
import os
import uuid
from pathlib import Path

ROOT = Path('/workspace/temperature-study')
DATA = json.loads((ROOT/'pilot_inputs.json').read_text())
EVENTS = ROOT/'results/pilot_phases.jsonl'


def atomic(path, value):
    temp = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.new')
    temp.write_text(json.dumps(value))
    temp.replace(path)


def task(row, method, scheme, temperature, samples=10, **extra):
    result = {k: row[k] for k in ('document', 'window_number', 'mask_id')}
    result.update(method=method, scheme=scheme, temperature=temperature, samples=samples, **extra)
    result['id'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:20]
    return result


def phase(name, jobs, gpus=tuple(range(8))):
    start = time.time()
    waiting = list(enumerate(jobs))
    active = {}
    results = []
    while waiting or active:
        for gpu in gpus:
            folder = ROOT/'worker_io'/str(gpu)
            if gpu not in active and waiting:
                index, job = waiting.pop(0)
                job = dict(job, id=f'{name}-{index}')
                result_path = folder/'result.json'
                result_path.unlink(missing_ok=True)
                atomic(folder/'job.json', job)
                active[gpu] = (job['id'], time.time())
            if gpu in active and (folder/'result.json').exists():
                value = json.loads((folder/'result.json').read_text())
                assert value['job_id'] == active[gpu][0]
                results.append(value)
                del active[gpu]
                if 'error' in value:
                    print('ERROR', name, gpu, value['error'][-500:], flush=True)
                print('PROGRESS', name, len(results), '/', len(jobs), flush=True)
        if any(time.time()-launched > 1200 for _,launched in active.values()):
            raise TimeoutError(f'Bounded pilot task timeout in {name}')
        atomic(ROOT/'results/pilot_status.json', dict(phase=name, completed=len(results), jobs=len(jobs),
                                                     elapsed_seconds=time.time()-start, updated=time.time()))
        time.sleep(.1)
    summary = dict(phase=name, wall_seconds=time.time()-start, jobs=len(jobs),
                   errors=sum('error' in r for r in results), results=results)
    with EVENTS.open('a') as handle:
        handle.write(json.dumps(summary)+'\n')
    print('PHASE_DONE', name, round(summary['wall_seconds'], 2), flush=True)
    return results


def main():
    if (ROOT/'results/runtime_study_excluded.json').exists():
        raise RuntimeError('Historical pilot controller disabled: it includes the excluded runtime study.')
    for gpu in range(8):
        assert (ROOT/'worker_io'/str(gpu)/'ready.json').exists()
    row = DATA['selected']['validation'][11]
    calibration = [task(row, 'path_sampling', 'low-confidence', .5, 10),
                   task(row, 'path_sampling', 'threshold', 2., 10),
                   task(row, 'monte-carlo', 'low-confidence', 2., 5000)]
    phase('solo', [dict(tasks=calibration)], gpus=(0,))
    phase('eight_gpu_threads3', [dict(tasks=calibration) for _ in range(8)])
    for threads in (2, 4):
        phase(f'eight_gpu_threads{threads}',
              [dict(tasks=[dict(t, threads=threads) for t in calibration]) for _ in range(8)])

    # Paired cache verification uses exactly the same seed and singleton model forwards.
    row = DATA['selected']['exact'][7]
    paired = [task(row, method, scheme, temp, 500, reduced=True)
              for method in ('exact', 'path_sampling')
              for temp, scheme in [(.5,'low-confidence'),(.5,'threshold'),(2.,'low-confidence'),
                                   (2.,'threshold'),(1.,'threshold')]]
    baseline = phase('cache_baseline', [dict(tasks=paired)], gpus=(0,))
    cached = phase('cache_enabled', [dict(tasks=paired, shared_cache=True)], gpus=(0,))
    if 'error' not in baseline[0] and 'error' not in cached[0]:
        for a,b in zip(baseline[0]['rows'], cached[0]['rows']):
            assert a['probability'] == b['probability']
            assert a.get('log_probability') == b.get('log_probability')
        atomic(ROOT/'results/cache_validation.json', dict(bit_identical=True,
               baseline_seconds=sum(r['seconds'] for r in baseline[0]['rows']),
               cached_seconds=sum(r['seconds'] for r in cached[0]['rows']), cache=cached[0]['cache']))
    else:
        raise RuntimeError('Cache validation failed')

    jobs = []
    for row in DATA['selected']['validation']:
        for temp in (.5, 2.):
            for scheme in ('low-confidence', 'threshold'):
                for method, samples in [('path_sampling',500),('monte-carlo',5000)]:
                    jobs.append(dict(tasks=[task(row, method, scheme, temp, samples)]))
    phase('validation_representative', jobs)

    jobs = []
    for row in DATA['selected']['exact']:
        group = [task(row, method, scheme, temp, 500, reduced=True)
                 for method in ('exact','path_sampling')
                 for temp,scheme in [(.5,'low-confidence'),(.5,'threshold'),(2.,'low-confidence'),
                                     (2.,'threshold'),(1.,'threshold')]]
        jobs.append(dict(tasks=group, shared_cache=True))
    phase('exact_representative_shared', jobs)

    jobs = []
    for cohort in ('application','negative'):
        for row in DATA['selected'][cohort]:
            for temp in (.5,2.):
                for scheme in ('low-confidence','threshold'):
                    jobs.append(dict(tasks=[task(row,'path_sampling',scheme,temp,10,cohort=cohort)]))
    phase('ten_sample_application_controls', jobs)

    jobs = []
    for row in DATA['selected']['runtime'][::3]:
        for temp in (.5,2.):
            for scheme in ('low-confidence','threshold'):
                for method,budgets in [('path_sampling',(62,125,250,500,1000)),
                                       ('monte-carlo',(125,250,500,1250,2500,5000))]:
                    for budget in budgets:
                        jobs.append(dict(tasks=[task(row,method,scheme,temp,budget,timing=True)]))
    phase('runtime_budget_sweep', jobs)

    jobs = []
    for row in DATA['selected']['validation'][::6]:
        for temp in (.5,2.):
            for batch in (1,4,8):
                jobs.append(dict(tasks=[task(row,'random','random',temp,16,batch_size=batch,selected=True)]))
            jobs.append(dict(tasks=[task(row,'greedy','greedy',temp,1)]))
    phase('other_schemes',jobs)
    atomic(ROOT/'results/PILOT_COMPLETE.json',dict(completed=time.time(),full_experiment_started=False))
    for gpu in range(8):
        atomic(ROOT/'worker_io'/str(gpu)/'job.json',dict(stop=True))
    print('PILOT_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
