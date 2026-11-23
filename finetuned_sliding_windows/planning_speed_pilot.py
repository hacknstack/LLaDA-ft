"""Seven-GPU speed pilot; original estimator functions and native precision."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
import argparse,hashlib,json,math,multiprocessing as mp,queue,random,re,statistics,sys,time,traceback
from pathlib import Path
from planning_inventory import ROOT,OUT,KEYS,corpus

MASKS=[
 ('odd',[i for i in range(1,101) if i%2==1]),
 ('blocks10_first',[i for i in range(1,101) if ((i-1)//10)%2==0]),
 ('outer24_26',[i for i in range(1,101) if not 24<i<75]),
 ('blocks25_first',[i for i in range(1,101) if i%50 in range(1,26)]),
 ('prefix50',list(range(1,51))),
 ('even',[i for i in range(1,101) if i%2==0]),
 ('pairs34',[i for i in range(1,101) if i%4 in [0,3]]),
 ('blocks5_second',[i for i in range(1,101) if (i-1)%10 in range(5,10)]),
 ('blocks10_second',[i for i in range(1,101) if (i-1)%20 in range(10,20)]),
 ('middle25_74',list(range(25,75))),
 ('blocks25_second',[i for i in range(1,101) if (i-1)%50 in range(25,50)]),
 ('suffix50',list(range(51,101)))]
assert all(len(v)==50 for _,v in MASKS)

def prepare():
    from tokenizers import Tokenizer
    docs,counts=corpus();selected=[]
    # Two length strata in every month/group; selection is independent of model scores.
    for month_index,month in enumerate(['2025-03','2025-04','2025-05','2025-06']):
        for group_index,group in enumerate(['train','control']):
            pool=sorted([d for d in docs if d['group']==group and d['source_config']==month and d['words']>=180],key=lambda d:(d['words'],d['filtered_index']))
            assert len(pool)>=4,(month,group,len(pool))
            for replicate in range(2):
                quartile=(month_index+2*group_index+2*replicate)%4
                part=pool[len(pool)*quartile//4:len(pool)*(quartile+1)//4]
                row=min(part,key=lambda d:hashlib.sha256(('speed-pilot-20261101:'+str(d['filtered_index'])).encode()).hexdigest())
                selected.append(dict(row,length_quartile=quartile,full_mask_pilot=replicate==0))
    tokenizers={key:Tokenizer.from_file(str(ROOT/'models'/key/'tokenizer.json')) for key in KEYS}
    from planning_inventory import count_windows
    windows=[]
    for row in selected:
        text=row['content'];starts=[0]+[m.start() for m in re.finditer(r'\S+',text)][1:]
        counts_by={key:count_windows(tok,text) for key,tok in tokenizers.items()};common=min(counts_by.values())
        assert common>=16
        requested=sorted({round((common-1)*j/15) for j in range(16)})
        for j,index in enumerate(requested):
            snippet=text[starts[index]:starts[index]+800]
            windows.append(dict(group=row['group'],filtered_index=row['filtered_index'],source_config=row['source_config'],
                words=row['words'],length_quartile=row['length_quartile'],window_index=index,char_start=starts[index],
                full_mask_pilot=row['full_mask_pilot'],llada_pilot=j in [2,12],
                tokens={key:tok.encode(snippet,add_special_tokens=False).ids[:100] for key,tok in tokenizers.items()}))
    (OUT/'speed_inputs.json').write_text(json.dumps(windows))
    (OUT/'speed_selection.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='content'} for r in selected],indent=2))
    print('SPEED_PILOT_SELECTION_READY',len(selected),'documents',len(windows),'AR windows',flush=True)
    return windows

def worker(gpu,key,inbox,outbox):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false')
    try:
        import torch
        import sliding_window_extraction as ex
        torch.set_num_threads(2)
        family={'llada':'llada','llama2_7b':'llama2','llama3_8b':'llama','mistral7b_v01':'mistral','olmo7b':'olmo'}[key]
        folder=str(ROOT/'models'/key)
        marker=json.loads((Path(folder)/'COMPLETE.json').read_text());assert marker['epoch']==3
        model=ex._load_model(folder,family,'cuda:0');model.config.use_cache=False
        tokenizer=ex._load_tokenizer(folder)
        data=json.loads((OUT/'speed_inputs.json').read_text())
        # Match the fast backend inventory against the actual Transformers tokenizer.
        documents,_=corpus();lookup={(d['group'],d['filtered_index']):d['content'] for d in documents}
        for example in data[::16]:
            text=lookup[(example['group'],example['filtered_index'])];start=example['char_start']
            assert tokenizer(text[start:start+800],add_special_tokens=False)['input_ids'][:100]==example['tokens'][key]
        sys.argv=['sliding_window_extraction.py','unused.txt','--model-name',folder,'--model-family',family,
                  '--mode','path_sampling' if key=='llada' else 'exact','--decoding-scheme','full',
                  '--temperature','1','--num-samples','10','--seed','12345']
        args=ex.parse_args()
        with torch.no_grad():
            x=torch.tensor([data[0]['tokens'][key]],device='cuda')
            if key=='llada':x[:,50:]=126336
            for _ in range(3):model(x)
        torch.cuda.synchronize();outbox.put(dict(event='ready',gpu=gpu,model=key))
        while True:
            task=inbox.get()
            if task is None:break
            torch.set_num_threads(task['threads'])
            example=data[task['input_index']];ids=example['tokens'][key]
            args.remasking=task.get('scheme','low-confidence');args.masked_indexes=task.get('masked_indexes')
            args.confidence_threshold=.9
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
            p,_=ex._compute_probability(model,ids[:50],ids[50:],args)
            torch.cuda.synchronize();seconds=time.perf_counter()-t
            logp=args._last_log_probability
            assert math.isfinite(p) and 0<=p<=1+1e-10 and logp<=1e-10
            if p>0:assert abs(math.log(p)-logp)<1e-8
            outbox.put(dict(event='result',gpu=gpu,model=key,task=task,
                group=example['group'],filtered_index=example['filtered_index'],month=example['source_config'],
                window_index=example['window_index'],length_quartile=example['length_quartile'],
                seconds=seconds,probability=p,log_probability=logp,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    except BaseException:
        outbox.put(dict(event='error',gpu=gpu,model=key,traceback=traceback.format_exc()))
        raise

def start_workers(keys):
    context=mp.get_context('spawn');out=context.Queue();processes=[];inboxes=[]
    for gpu,key in enumerate(keys):
        inbox=context.Queue();child=context.Process(target=worker,args=(gpu,key,inbox,out));child.start()
        inboxes.append(inbox);processes.append(child)
    for _ in keys:
        msg=out.get(timeout=300)
        if msg['event']=='error':raise RuntimeError(msg['traceback'])
        assert msg['event']=='ready';print('SPEED_WORKER_READY',msg['gpu'],msg['model'],flush=True)
    return processes,inboxes,out

def stop_workers(processes,inboxes):
    for q in inboxes:q.put(None)
    for p in processes:
        p.join(timeout=120)
        assert p.exitcode==0,(p.pid,p.exitcode)

def phase(name,assignments,processes,inboxes,out):
    started=time.perf_counter();pending=0;rows=[]
    for gpu,tasks in assignments.items():
        for task in tasks:inboxes[gpu].put(dict(task,phase=name));pending+=1
    with (OUT/'speed_results.jsonl').open('a') as stream:
        for i in range(pending):
            try:result=out.get(timeout=300)
            except queue.Empty:raise RuntimeError('GPU pilot worker stalled')
            if result['event']=='error':raise RuntimeError(result['traceback'])
            assert result['event']=='result'
            rows.append(result);stream.write(json.dumps(result)+'\n');stream.flush()
            if (i+1)%28==0:print('SPEED_PHASE_PROGRESS',name,i+1,'/',pending,flush=True)
    info=dict(phase=name,rows=len(rows),wall_seconds=time.perf_counter()-started,
              gpu_seconds=sum(r['seconds'] for r in rows))
    with (OUT/'speed_phases.jsonl').open('a') as f:f.write(json.dumps(info)+'\n')
    print('SPEED_PHASE_COMPLETE',json.dumps(info),flush=True)
    return rows,info

def main():
    windows=prepare();run_started=time.monotonic()
    processes,inboxes,out=start_workers(KEYS[1:])
    tasks=[dict(input_index=i,threads=2) for i in range(len(windows))]
    phase('ar_four_parallel',{i:tasks for i in range(4)},processes,inboxes,out)
    stop_workers(processes,inboxes)
    processes,inboxes,out=start_workers(['llada']*7)
    # Match the same six computations on one GPU and on all seven.
    calibration_inputs=[i for i,w in enumerate(windows) if w['llada_pilot']][:3]
    def calibrate(threads):return [dict(input_index=i,threads=threads,scheme=s,mask='suffix50',masked_indexes=list(range(51,101))) for i in calibration_inputs for s in ['low-confidence','threshold']]
    _,solo=phase('solo_threads4',{0:calibrate(4)},processes,inboxes,out)
    _,parallel2=phase('seven_threads2',{i:calibrate(2) for i in range(7)},processes,inboxes,out)
    _,parallel4=phase('seven_threads4',{i:calibrate(4) for i in range(7)},processes,inboxes,out)
    threads=2 if parallel2['wall_seconds']<=parallel4['wall_seconds'] else 4
    tasks=[]
    for i,w in enumerate(windows):
        if not w['llada_pilot']:continue
        masks=MASKS if w['full_mask_pilot'] else [MASKS[-1]]
        for name,mask in masks:
            for scheme in ['low-confidence','threshold']:
                tasks.append(dict(input_index=i,threads=threads,scheme=scheme,mask=name,masked_indexes=mask))
    random.Random(20261101).shuffle(tasks)
    rows,measured=phase('representative_seven',{i:tasks[i::7] for i in range(7)},processes,inboxes,out)
    stop_workers(processes,inboxes)
    summary=dict(status='complete',benchmark_seconds=time.monotonic()-run_started,chosen_cpu_threads=threads,
                 single_gpu_calibration=solo,seven_gpu_threads2=parallel2,seven_gpu_threads4=parallel4,
                 representative=measured,benchmark_articles=len({(w['group'],w['filtered_index']) for w in windows}),
                 mask_definitions=MASKS,full_sweep_started=False)
    (OUT/'speed_summary.json').write_text(json.dumps(summary,indent=2))
    print('SPEED_PILOT_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
