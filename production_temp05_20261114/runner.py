"""Scalar LLaDA T=.5 production run, verified Drive saves and RunPod stop.

No paper runtime study; progress timestamps serve operational monitoring only.
"""
import argparse
import configparser
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path('/workspace/temperature-study')
PRIVATE = Path('/root/.config')


def atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.new')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def request(url, method='GET', data=None, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, data=data, method=method, headers=headers or {}), timeout=90) as r:
        return r.status, dict(r.headers), r.read()


class Drive:
    def __init__(self):
        self.token = None
        self.expiry = 0

    def auth(self):
        if time.time() < self.expiry:
            return {'Authorization': 'Bearer '+self.token}
        c = configparser.ConfigParser()
        c.read(PRIVATE/'research-rclone/rclone.conf')
        s = c['researchdrive']
        refresh = json.loads(s['token'])['refresh_token']
        body = urllib.parse.urlencode(dict(client_id=s['client_id'], client_secret=s['client_secret'],
                     refresh_token=refresh, grant_type='refresh_token')).encode()
        _, _, raw = request('https://oauth2.googleapis.com/token', method='POST', data=body)
        r = json.loads(raw)
        self.token = r['access_token']
        self.expiry = time.time()+r.get('expires_in', 3600)-120
        return {'Authorization': 'Bearer '+self.token}

    def json(self, url, method='GET', value=None):
        h = self.auth() | {'Content-Type': 'application/json'}
        _, _, raw = request(url, method, None if value is None else json.dumps(value).encode(), h)
        return json.loads(raw) if raw else {}

    def folder(self, name, parent):
        r = self.json('https://www.googleapis.com/drive/v3/files?fields=id,name,webViewLink', 'POST',
                      dict(name=name, mimeType='application/vnd.google-apps.folder', parents=[parent]))
        check = self.json('https://www.googleapis.com/drive/v3/files/'+r['id']+'?fields=id,name,mimeType,webViewLink')
        assert check['name'] == name and check['mimeType'].endswith('.folder')
        return check

    def upload(self, path, folder, file_id=None):
        # The caller owns an immutable snapshot for the duration of this upload.
        size = path.stat().st_size
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        url = 'https://www.googleapis.com/upload/drive/v3/files'
        if file_id:
            url += '/'+file_id
        url += '?uploadType=resumable&fields=id,name,size,md5Checksum,webViewLink'
        meta = {'name': path.name}
        if not file_id:
            meta['parents'] = [folder]
        headers = self.auth() | {'Content-Type': 'application/json', 'X-Upload-Content-Type': 'application/octet-stream',
                                 'X-Upload-Content-Length': str(size)}
        _, h, _ = request(url, 'PATCH' if file_id else 'POST', json.dumps(meta).encode(), headers)
        location = next(v for k,v in h.items() if k.lower() == 'location')
        result = None
        with path.open('rb') as f:
            offset = 0
            while offset < size:
                chunk = f.read(8*1024**2)
                end = offset+len(chunk)-1
                headers = self.auth() | {'Content-Type': 'application/octet-stream',
                                  'Content-Range': f'bytes {offset}-{end}/{size}'}
                try:
                    status, _, raw = request(location, 'PUT', chunk, headers)
                    result = json.loads(raw) if raw else None
                except urllib.error.HTTPError as e:
                    if e.code != 308:
                        raise
                offset = end+1
        assert result and result.get('id')
        check = self.json('https://www.googleapis.com/drive/v3/files/'+result['id']+'?fields=id,name,size,md5Checksum,webViewLink')
        assert int(check['size']) == size and check['md5Checksum'] == md5
        return check


def pod_request(config, action=None):
    key = (PRIVATE/'research-runpod/api_key').read_text(encoding='utf-8-sig').strip()
    url = 'https://api.runpod.io/v2/pods/'+config['pod_id']
    if action:
        url += '/action'
    headers = {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json', 'User-Agent': 'research-experiment/1.0'}
    _, _, raw = request(url, 'POST' if action else 'GET', None if action is None else json.dumps({'action': action}).encode(), headers)
    return json.loads(raw) if raw else {}


def jobs_for(config):
    jobs = []
    for index, row in enumerate(config['cohort']):
        for f in config['formats']:
            identity = '|'.join([config['run_id'], row['document'], row['window_number'], row['mask'], f['scheme'], f['estimator']])
            seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little') % (2**63-1)
            jobs.append(dict(id=len(jobs), index=index, row=row, scheme=f['scheme'], method=f['estimator'], seed=seed, file=f['file']))
    assert len(jobs) == 14400
    assert len({(r['document'], r['window_number'], r['mask']) for r in config['cohort']}) == 3600
    return jobs


def worker(gpu, config, run, tasks, events):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sys.path.insert(0, str(ROOT/'code'))
    try:
        import torch
        from transformers import AutoModel
        import probabilistic_extraction as pe
        torch.set_num_threads(config['threads_per_worker'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = AutoModel.from_pretrained(ROOT/'models/llada-base', trust_remote_code=True, local_files_only=True,
                   torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={'': 0}).eval()
        model.config.use_cache = False
        data = json.loads((ROOT/'pilot_inputs.json').read_text())
        log = (run/f'worker_{gpu}.jsonl').open('a', buffering=1)
        events.put({'type': 'ready', 'gpu': gpu})
        while True:
            job = tasks.get()
            if job is None:
                break
            row = job['row']
            sequence = torch.tensor([data['tokens'][row['document']+':'+row['window_number']]], device='cuda')
            mask = data['masks'][data['labels'].index(row['mask'])]
            assert len(mask) == 50 and sequence.shape == (1,100)
            common = dict(model=model, sequence_tokens=sequence, masked_indexes=mask, steps=50,
                   attention_mask=None, mask_id=126336, temperature=.5, seed=job['seed'],
                   num_samples=500 if job['method']=='path_sampling' else 5000)
            extra = {'confidence_threshold': .9} if job['scheme']=='threshold' else {}
            with torch.inference_mode():
                if job['method']=='path_sampling':
                    fn = (pe._path_sampling_low_confidence_probability_fast_from_partially_masked
                          if job['scheme']=='low-confidence' else pe._path_sampling_threshold_probability_fast_from_partially_masked)
                    result = fn(**common, **extra)
                    weights = [float(v) for v in result['sample_log_probabilities']]
                    assert len(weights)==500 and all(v <= 1e-9 and not math.isnan(v) for v in weights)
                    probability = float(result['probability'])
                else:
                    fn = (pe._monte_carlo_probability_temperature_fast_from_partially_masked
                          if job['scheme']=='low-confidence' else pe._monte_carlo_threshold_probability_fast_from_partially_masked)
                    result = fn(**common, **extra, decoding_scheme='full', k=1, scale=False, return_sample_logs=False)
                    weights = None
                    probability = float(result.estimate)
                assert math.isfinite(probability) and 0 <= probability <= 1
            # Encode -inf log weights as JSON null; final CSV restores literal -inf.
            record = dict(id=job['id'], probability=probability,
                     log_weights=None if weights is None else [None if v==-math.inf else v for v in weights])
            log.write(json.dumps(record, allow_nan=False)+'\n')
            log.flush()
            os.fsync(log.fileno())
            events.put({'type': 'done', 'id': job['id'], 'gpu': gpu})
        log.close()
        events.put({'type': 'exit', 'gpu': gpu})
    except BaseException:
        error = traceback.format_exc()
        (run/f'worker_{gpu}_ERROR.txt').write_text(error)
        events.put({'type': 'error', 'gpu': gpu, 'traceback': error})
        raise


def records_in(run):
    records = {}
    for p in sorted(run.glob('worker_*.jsonl')):
        with p.open() as f:
            for line in f:
                if not line.endswith('\n'):
                    break
                record = json.loads(line)
                if record['id'] in records:
                    assert records[record['id']] == record, 'Conflicting duplicate result'
                records[record['id']] = record
    return records


def export(config, run, jobs, records, final=False):
    folder = run/'exports'
    folder.mkdir(exist_ok=True)
    coverage = {}
    for fmt in config['formats']:
        header = json.loads(fmt['header_json'])
        path = folder/fmt['file']
        temp = path.with_suffix('.csv.new')
        selected = sorted((j for j in jobs if j['file']==fmt['file'] and j['id'] in records), key=lambda j:j['index'])
        if final:
            assert len(selected)==3600
        with temp.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for job in selected:
                record = records[job['id']]
                row = job['row']
                values = [row['document'], row['window_number'], row['mask']]
                if job['method']=='path_sampling':
                    weights = [-math.inf if x is None else x for x in record['log_weights']]
                    assert len(weights)==500
                    averages = [math.fsum(math.exp(v) for v in weights[:n])/n for n in (1,5,10,25,50,100,250,500)]
                    assert math.isclose(averages[-1],record['probability'],rel_tol=1e-10,abs_tol=1e-300), 'STS prefix/result mismatch'
                    values += averages+weights
                else:
                    values += [record['probability']]
                assert len(values)==len(header)
                writer.writerow(values)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(path)
        # Independently read the written file and verify width, header and key order.
        with path.open(newline='') as f:
            reader = csv.reader(f)
            assert next(reader)==header
            count = 0
            for written, job in zip(reader,selected,strict=True):
                assert len(written)==len(header)
                expected = job['row']
                assert written[:3]==[expected['document'],expected['window_number'],expected['mask']]
                count += 1
        coverage[path.name] = dict(rows=count, columns=len(header), bytes=path.stat().st_size,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    atomic(folder/'validation.json', dict(final=final, completed=len(records), expected=14400, files=coverage,
              correct_reference_headers=True, paper_runtime_study_excluded=True))
    return folder


def save_drive(config, run, jobs, final=False, outcome='running'):
    records = records_in(run)
    folder = export(config, run, jobs, records, final)
    manifest_path = run/'drive_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    drive = Drive()
    atomic(folder/'status.json', dict(run_id=config['run_id'], outcome=outcome, completed=len(records), expected=14400,
             updated_unix=time.time(), scalar_forwards=True, temperature=.5, automatic_stop=True))
    files = list(folder.glob('*.csv'))+[folder/'validation.json',folder/'status.json',run/'config.json']
    for err in run.glob('*ERROR.txt'):
        files.append(err)
    for p in files:
        previous = manifest['files'].get(p.name)
        meta = drive.upload(p, manifest['folder']['id'], None if previous is None else previous['id'])
        manifest['files'][p.name] = meta
        atomic(manifest_path, manifest)
    manifest['last_verified_completed'] = len(records)
    manifest['final_verified'] = final
    manifest['updated_unix'] = time.time()
    atomic(manifest_path, manifest)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run = ROOT/'runs'/config['run_id']
    run.mkdir(parents=True,exist_ok=True)
    if not (run/'config.json').exists():
        atomic(run/'config.json',config)
    jobs = jobs_for(config)
    pod = pod_request(config)
    assert pod.get('id')==config['pod_id'] and 'stop' in pod.get('actions',[]), 'Pod stop permission not available'
    drive = Drive()
    manifest_path = run/'drive_manifest.json'
    if not manifest_path.exists():
        folder = drive.folder(config['run_id'],config['drive_parent'])
        atomic(manifest_path,dict(folder=folder,files={}))
    if args.preflight:
        n=save_drive(config,run,jobs)
        print(json.dumps(dict(preflight='passed',completed=n,pod_id=config['pod_id'],folder=json.loads(manifest_path.read_text())['folder'],run=str(run))),flush=True)
        return
    done = records_in(run)
    pending = [j for j in jobs if j['id'] not in done]
    if args.smoke:
        pending = pending[:8]
    else:
        pending.sort(key=lambda j:hashlib.sha256(str(j['id']).encode()).digest())
    ctx = mp.get_context('spawn')
    tasks, events = ctx.Queue(),ctx.Queue()
    for job in pending:
        tasks.put(job)
    for _ in range(config['num_gpus']):
        tasks.put(None)
    processes=[]
    initial=len(done)
    started=time.time()
    last_progress=last_save=started
    completed=initial
    outcome='failed'
    error=None
    uploader=None
    upload_errors=[]

    def upload_async():
        try:
            n=save_drive(config,run,jobs)
            print('DRIVE_CHECKPOINT_VERIFIED',n,flush=True)
        except Exception:
            upload_errors.append(traceback.format_exc())
            print('DRIVE_CHECKPOINT_RETRY_NEEDED',flush=True)

    def interrupted(signum, frame):
        raise RuntimeError('Controller interrupted by signal '+str(signum))

    signal.signal(signal.SIGTERM,interrupted)
    try:
        for gpu in range(config['num_gpus']):
            p=ctx.Process(target=worker,args=(gpu,config,run,tasks,events))
            p.start()
            processes.append(p)
        exits=set()
        last_status=0
        while len(exits)<len(processes):
            try:
                event=events.get(timeout=2)
                if event['type']=='done':
                    completed+=1
                    last_progress=time.time()
                elif event['type']=='error':
                    raise RuntimeError('Worker failed: '+event['traceback'])
                elif event['type']=='exit':
                    exits.add(event['gpu'])
            except queue.Empty:
                pass
            for p in processes:
                if p.exitcode not in (None,0):
                    raise RuntimeError(f'Worker process {p.pid} exited {p.exitcode}')
            now=time.time()
            if now-last_progress>config['stall_timeout_seconds']:
                raise RuntimeError('No completed case within stall timeout')
            if now-last_status>20:
                atomic(run/'progress.json',dict(state='running',completed=completed,expected=14400,started_unix=started,
                       updated_unix=now,smoke=args.smoke,active_workers=sum(p.is_alive() for p in processes),
                       drive_verified=json.loads(manifest_path.read_text()).get('last_verified_completed',0)))
                print('PROGRESS',completed,14400,'elapsed_minutes',round((now-started)/60,2),flush=True)
                last_status=now
            if now-last_save>config['upload_interval_seconds'] and (uploader is None or not uploader.is_alive()):
                uploader=threading.Thread(target=upload_async)
                uploader.start()
                last_save=now
        for p in processes:
            p.join(timeout=30)
            assert p.exitcode==0
        completed=len(records_in(run))
        assert completed==initial+len(pending)
        if not args.smoke:
            assert completed==14400
        outcome='smoke_passed' if args.smoke else 'complete'
    except BaseException:
        error=traceback.format_exc()
        (run/'CONTROLLER_ERROR.txt').write_text(error)
        print('CONTROLLER_FAILED',error,flush=True)
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=20)
                if p.is_alive():
                    p.kill()
        if uploader:
            uploader.join(timeout=600)
            assert not uploader.is_alive(), 'Uploader did not exit'
        saved=False
        for attempt in range(6):
            try:
                n=save_drive(config,run,jobs,final=(outcome=='complete'),outcome=outcome)
                saved=True
                break
            except Exception:
                (run/'UPLOAD_ERROR.txt').write_text(traceback.format_exc())
                print('FINAL_UPLOAD_RETRY',attempt+1,flush=True)
                time.sleep(10*(attempt+1))
        atomic(run/'progress.json',dict(state=outcome,completed=len(records_in(run)),expected=14400,updated_unix=time.time(),
                started_unix=started,drive_saved=saved,automatic_stop=not args.smoke,smoke=args.smoke,error=bool(error)))
        if args.smoke:
            assert outcome=='smoke_passed' and saved
            print('SMOKE_PASSED_DRIVE_VERIFIED',flush=True)
        elif saved:
            atomic(run/'shutdown_requested.json',dict(pod_id=config['pod_id'],action='stop',outcome=outcome,
                   results_verified_on_drive=True,time_unix=time.time()))
            # Upload the intent before the control-plane action can remove this container.
            manifest=json.loads(manifest_path.read_text())
            Drive().upload(run/'shutdown_requested.json',manifest['folder']['id'])
            for attempt in range(10):
                try:
                    pod_request(config,'stop')
                    print('POD_STOP_ACCEPTED',config['pod_id'],flush=True)
                    break
                except Exception:
                    print('POD_STOP_RETRY',attempt+1,flush=True)
                    time.sleep(10)
            else:
                raise RuntimeError('Automatic pod stop failed; external guardian must stop pod')
        else:
            # External guardian secures a local copy before issuing its stop fallback.
            atomic(run/'NEEDS_LOCAL_BACKUP_AND_STOP.json',dict(outcome=outcome,drive_upload_failed=True))
            raise RuntimeError('Drive upload failed; external guardian must back up and stop pod')


if __name__=='__main__':
    main()
