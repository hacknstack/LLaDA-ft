import datetime,hashlib,json,os,sqlite3
from pathlib import Path

ROOT=Path('/workspace/finetuned-extraction')
PYTHON=str(ROOT/'venv/bin/python')
RCLONE=str(ROOT/'tools/rclone-v1.75.2-linux-amd64/rclone')
DRIVE_CONFIG='/root/.config/research-rclone/rclone.conf'
FIELDS=['task_id','model','group','filtered_index','window_index','char_start','mask_id','scheme','num_samples','seed','p_z','log_p_z','probability','log_probability','extracted','seconds','gpu']

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stamp():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)
def connect(run):
    c=sqlite3.connect(str(Path(run)/'state.sqlite'),timeout=60)
    c.row_factory=sqlite3.Row;c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA synchronous=FULL');c.execute('PRAGMA busy_timeout=60000')
    return c
def seed_for(group,index,window,mask,scheme):
    s=f'20261101|{group}|{index}|{window}|{mask}|{scheme}'
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'big')%(2**63-1)
def result_row(task,result,gpu):
    return dict(task_id=task['task_id'],model=task['model'],group=task['doc_group'],filtered_index=task['filtered_index'],
        window_index=task['window_index'],char_start=task['char_start'],mask_id=task['mask_id'],scheme=task['scheme'],
        num_samples=10 if task['model']=='llada' else 1,seed=task['seed'],probability=result['probability'],
        log_probability=result['log_probability'],extracted=int(result['probability']>=.001),seconds=result['seconds'],gpu=gpu)
