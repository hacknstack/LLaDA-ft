"""End task admission after a bounded pilot and let current calculations finish."""
import json
import os
import signal
import time
from pathlib import Path
from pilot_controller import ROOT,atomic


def alive(pid,needle):
    p=Path(f'/proc/{pid}/cmdline')
    return p.exists() and needle.encode() in p.read_bytes()


def main():
    status=json.loads((ROOT/'results/pilot_status.json').read_text())
    beginning=status['updated']-status['elapsed_seconds']
    while time.time()<beginning+1400 and alive(3052,'pilot_stream.py'):
        if (ROOT/'results/PRIMARY_PILOT_COMPLETE.json').exists():return
        time.sleep(2)
    if (ROOT/'results/PRIMARY_PILOT_COMPLETE.json').exists():return
    if alive(3052,'pilot_stream.py'):
        os.kill(3052,signal.SIGTERM)
    for gpu in range(8):atomic(ROOT/'worker_io'/str(gpu)/'job.json',dict(stop=True))
    print('PILOT_ADMISSION_CLOSED_DRAINING',flush=True)
    while True:
        running=[]
        for gpu in range(8):
            pid=json.loads((ROOT/'worker_io'/str(gpu)/'ready.json').read_text())['pid']
            if alive(pid,'pilot_worker.py'):running.append(gpu)
        atomic(ROOT/'results/pilot_status.json',dict(phase='draining_bounded_pilot',active_gpus=running,
                elapsed_seconds=time.time()-beginning,updated=time.time()))
        if not running:break
        if time.time()>beginning+4200:
            for gpu in running:
                pid=json.loads((ROOT/'worker_io'/str(gpu)/'ready.json').read_text())['pid']
                os.kill(pid,signal.SIGTERM)
            break
        time.sleep(2)
    results_path=ROOT/'results/stream_jobs.jsonl'
    existing=[json.loads(l) for l in results_path.read_text().splitlines()]
    seen={j['job_id'] for j in existing}
    for gpu in range(8):
        path=ROOT/'worker_io'/str(gpu)/'result.json'
        if not path.exists():continue
        value=json.loads(path.read_text())
        if value['job_id'] not in seen:
            with results_path.open('a') as handle:handle.write(json.dumps(value)+'\n')
            seen.add(value['job_id'])
    atomic(ROOT/'results/PRIMARY_PILOT_COMPLETE.json',dict(status='bounded_benchmark_complete',
           completed=time.time(),full_experiment_started=False,jobs_completed=len(seen),
           elapsed_seconds=time.time()-beginning,admission_seconds=1400))
    print('PILOT_DRAIN_COMPLETE',flush=True)


if __name__=='__main__':main()
