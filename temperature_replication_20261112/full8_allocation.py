"""Validate allocation throughput at the full pod's CPU/GPU concurrency."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path('/workspace/temperature-study')
OUT=ROOT/'allocation_pilot'


def cpu_stat():
    return {line.split()[0]:int(line.split()[1]) for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines()}


def phase(mode,contexts,bulk):
    before=cpu_stat();start=time.time();processes=[]
    for gpu in range(8):
        log=open(OUT/f'full8_{mode}_gpu{gpu}.log','w')
        args=[str(ROOT/'venv/bin/python'),'-u',str(ROOT/'code/pilot_allocation.py'),'--gpu',str(gpu),
              '--mode',mode,'--contexts',str(contexts),'--scale','2','--cases','4']
        if bulk:args.append('--bulk')
        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
             start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        processes.append((gpu,p))
    (OUT/f'full8_{mode}_pids.json').write_text(json.dumps({str(g):p.pid for g,p in processes}))
    while any(p.poll() is None for _,p in processes) and time.time()-start<900:time.sleep(2)
    for gpu,p in processes:
        if p.poll() is None:p.terminate()
        code=p.wait(timeout=60)
        assert code==0,(mode,gpu,code)
    after=cpu_stat();results=[]
    for gpu in range(8):
        suffix='_bulk' if bulk else ''
        path=OUT/f'{mode}_contexts{contexts}_gpu{gpu}{suffix}_scale2_cases4.json'
        results.append(json.loads(path.read_text()))
    elapsed=max(r['total_seconds'] for r in results)
    output=dict(mode=mode,contexts=contexts,bulk=bulk,compute_makespan_seconds=elapsed,
                cpu_core_seconds=(after['usage_usec']-before['usage_usec'])/1e6,
                cpu_throttled_seconds=(after['throttled_usec']-before['throttled_usec'])/1e6,
                results=results)
    print('FULL8_PHASE_DONE',mode,elapsed,flush=True)
    return output


def main():
    while not (OUT/'COMPLETE.json').exists() or not (OUT/'BULK_COMPLETE.json').exists():time.sleep(2)
    baseline=phase('within-case',1,False)
    optimized=phase('broker',4,True)
    reference=baseline['results'][0]['tasks']
    for result in baseline['results']+optimized['results']:
        assert len(result['tasks'])==len(reference)
        assert all(a['signature']==b['signature'] for a,b in zip(reference,result['tasks']))
    report=dict(budget_only=True,scientific_runtime_study=False,full_experiment_started=False,
                bit_identical=True,all_eight_slices=True,
                baseline_seconds=baseline['compute_makespan_seconds'],optimized_seconds=optimized['compute_makespan_seconds'],
                throughput_speedup=baseline['compute_makespan_seconds']/optimized['compute_makespan_seconds'],
                baseline=baseline,optimized=optimized)
    (OUT/'full8_summary.json').write_text(json.dumps(report,indent=2))
    print('FULL8_VALIDATED',report['throughput_speedup'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        (OUT/'ERROR_full8.json').write_text(json.dumps(dict(error=traceback.format_exc())))
        raise
