"""Measured budget and deterministic shared document/window manifests. No full run."""
import bisect,collections,csv,datetime,hashlib,json,math,statistics,time
from pathlib import Path
from planning_inventory import OUT

TURN_START=datetime.datetime(2026,11,1,16,26,37,tzinfo=datetime.timezone.utc)
RATE=6.40;START_BALANCE=490.;RESERVE=50.;SLOWDOWN=1.25

def ranked(doc):
    return hashlib.sha256(('bbc-main-20261101:'+doc['group']+':'+str(doc['filtered_index'])).encode()).hexdigest()

def windows_for(doc,cap):
    n=doc['common_windows']
    if cap is None or n<=cap:return list(range(n))
    width=cap//4
    starts=[round((n-width)*i/3) for i in range(4)]
    result=sorted({j for start in starts for j in range(start,start+width)})
    assert len(result)==cap
    return result

def write_documents(label,selected,cap):
    with (OUT/('documents_'+label+'.csv')).open('w',newline='') as f:
        fields=['group','filtered_index','source_config','published_date','words','common_windows','selected_windows','normalized_sha256','original_filtered_indices']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for d in selected:
            row={k:d[k] for k in fields if k not in ['selected_windows','original_filtered_indices']}
            row['selected_windows']=json.dumps(windows_for(d,cap));row['original_filtered_indices']=json.dumps(d['original_filtered_indices']);writer.writerow(row)

def main():
    assert (OUT/'inventory_repaired.json').exists()
    speed=json.loads((OUT/'speed_summary.json').read_text())
    inventory=json.loads((OUT/'corpus_inventory.json').read_text())
    results=[json.loads(s) for s in (OUT/'speed_results.jsonl').open()]
    selection=json.loads((OUT/'speed_selection.json').read_text())
    pilot={(r['group'],r['filtered_index']) for r in selection}
    fullpilot={(r['group'],r['filtered_index']) for r in selection if r['full_mask_pilot']}
    reps=[r for r in results if r['task']['phase']=='representative_seven']
    costs={};timing_rows=[]
    for group in ['train','control']:
        costs[group]=0.
        for name,_ in speed['mask_definitions']:
            for scheme in ['low-confidence','threshold']:
                values=[r['seconds'] for r in reps if r['group']==group and r['task']['mask']==name and r['task']['scheme']==scheme and (r['group'],r['filtered_index']) in fullpilot]
                assert len(values)==8,(group,name,scheme,len(values))
                avg=statistics.mean(values);costs[group]+=avg
                timing_rows.append(dict(group=group,mask=name,scheme=scheme,measurements=len(values),mean_seconds=avg,median_seconds=statistics.median(values),min_seconds=min(values),max_seconds=max(values)))
    ar={key:statistics.mean(r['seconds'] for r in results if r['model']==key) for key in ['llama2_7b','llama3_8b','mistral7b_v01','olmo7b']}
    ar_per_window=sum(ar.values())
    effective=min(6.75,speed['representative']['gpu_seconds']/speed['representative']['wall_seconds'])
    eligible=[d for d in inventory['documents'] if d['common_windows']>0]
    available=[d for d in eligible if (d['group'],d['filtered_index']) not in pilot]
    lengths=sorted(d['words'] for d in eligible)
    edges=[lengths[round((len(lengths)-1)*f)] for f in [.25,.5,.75]]
    cells=collections.defaultdict(lambda:dict(train=[],control=[]))
    for d in available:
        cell=(d['source_config'],bisect.bisect_right(edges,d['words']))
        cells[cell][d['group']].append(d)
    capacities={cell:min(len(v['train']),len(v['control'])) for cell,v in cells.items()}
    capacities={k:v for k,v in capacities.items() if v>0};total_capacity=sum(capacities.values())
    for cell in capacities:
        for g in ['train','control']:cells[cell][g].sort(key=ranked)
    now=datetime.datetime.now(datetime.timezone.utc)
    elapsed=(now-TURN_START).total_seconds()/3600
    spent=elapsed*RATE;available_dollars=START_BALANCE-spent-RESERVE
    allowed_hours=available_dollars/RATE
    def quota(n):
        alloc={c:n*cap//total_capacity for c,cap in capacities.items()}
        order=sorted(capacities,key=lambda c:(-(n*capacities[c]%total_capacity),c))
        for c in order[:n-sum(alloc.values())]:alloc[c]+=1
        return alloc
    scenarios=[];selections={}
    for cap in [8,16,32,64,None]:
        prefixes={}
        for c in capacities:
            for group in ['train','control']:
                vals=[0]
                for d in cells[c][group][:capacities[c]]:vals.append(vals[-1]+min(d['common_windows'],cap) if cap else vals[-1]+d['common_windows'])
                prefixes[c,group]=vals
        def estimate(n):
            alloc=quota(n)
            totals={g:sum(prefixes[c,g][alloc[c]] for c in capacities) for g in ['train','control']}
            gpu_seconds=sum(totals[g]*costs[g] for g in totals)
            ar_seconds=sum(totals.values())*ar_per_window
            nominal=(gpu_seconds+ar_seconds)/effective/3600+.25
            guarded=(gpu_seconds+ar_seconds)*SLOWDOWN/effective/3600+.25
            return alloc,totals,nominal,guarded,gpu_seconds,ar_seconds
        best=None
        for n in range(1,total_capacity+1):
            stats=estimate(n)
            if stats[3]<=allowed_hours:best=(n,stats)
        assert best
        n,(alloc,totals,nominal,guarded,gpu,ar_gpu)=best
        scenario='full_article' if cap is None else f'{cap}_windows'
        selected=[d for c in sorted(capacities) for g in ['train','control'] for d in cells[c][g][:alloc[c]]]
        selections[scenario]=selected
        result=dict(scenario=scenario,windows_cap=cap,train_documents=n,control_documents=n,total_documents=2*n,
            shared_window_starts=sum(totals.values()),windows_by_group=totals,
            llada_scores=24*sum(totals.values()),ar_scores_per_model=sum(totals.values()),
            nominal_pod_hours=nominal,guarded_pod_hours=guarded,guarded_cost=guarded*RATE,
            llada_gpu_hours=gpu/3600,ar_all_four_gpu_hours=ar_gpu/3600,
            train_pct_eligible=100*n/sum(d['group']=='train' for d in eligible),
            control_pct_eligible=100*n/sum(d['group']=='control' for d in eligible),
            train_pct_all_unique=100*n/inventory['counts']['train']['unique_documents'],
            control_pct_all_unique=100*n/inventory['counts']['control']['unique_documents'],
            represented_training_rows=sum(len(d['original_filtered_indices']) for d in selected if d['group']=='train'),
            stratum_counts=[dict(month=c[0],length_bin=c[1],pairs=count) for c,count in sorted(alloc.items())])
        scenarios.append(result)
        write_documents(scenario,selected,cap)
        if cap==8:
            nrec=max(1,n//10*10)
            ra,rt,rn,rg,rl,rarg=estimate(nrec)
            chosen=[d for c in sorted(capacities) for g in ['train','control'] for d in cells[c][g][:ra[c]]]
            recommendation=dict(result,scenario='recommended',train_documents=nrec,control_documents=nrec,total_documents=2*nrec,
                shared_window_starts=sum(rt.values()),windows_by_group=rt,llada_scores=24*sum(rt.values()),ar_scores_per_model=sum(rt.values()),
                nominal_pod_hours=rn,guarded_pod_hours=rg,guarded_cost=rg*RATE,llada_gpu_hours=rl/3600,ar_all_four_gpu_hours=rarg/3600,
                train_pct_eligible=100*nrec/sum(d['group']=='train' for d in eligible),control_pct_eligible=100*nrec/sum(d['group']=='control' for d in eligible),
                train_pct_all_unique=100*nrec/inventory['counts']['train']['unique_documents'],control_pct_all_unique=100*nrec/inventory['counts']['control']['unique_documents'],
                represented_training_rows=sum(len(d['original_filtered_indices']) for d in chosen if d['group']=='train'),
                stratum_counts=[dict(month=c[0],length_bin=c[1],pairs=count) for c,count in sorted(ra.items())])
            write_documents('recommended',chosen,8)
    result=dict(status='complete',generated_utc=now.isoformat(),pod_rate=RATE,starting_balance=START_BALANCE,
        elapsed_pod_hours_since_budget_received=elapsed,spent_since_budget_received=spent,reserve_dollars=RESERVE,
        available_main_dollars=available_dollars,main_hours_allowance=allowed_hours,slowdown_factor=SLOWDOWN,
        effective_gpus_used_in_prediction=effective,seconds_per_window_all_12_masks_both_schemes=costs,
        ar_seconds_per_window_by_model=ar,corpus_counts=inventory['counts'],
        eligible_counts={g:sum(d['group']==g for d in eligible) for g in ['train','control']},
        full_corpus_common_windows={g:sum(d['common_windows'] for d in eligible if d['group']==g) for g in ['train','control']},
        length_bin_edges_words=edges,pilot_documents_excluded_from_main=len(pilot),
        matched_stratum_pair_capacity=total_capacity,scenarios=scenarios,recommended_scenario='8_windows',recommendation=recommendation,
        assumptions=['All five models use identical article IDs and word-start window indices',
          'LLaDA uses all twelve masks, ten threshold and ten low-confidence samples per mask',
          'Four AR models use exact suffix scoring at the same selected windows',
          'Balanced train/control counts matched within month and pooled word-count quartile',
          'No article selection based on model probabilities; pilot articles excluded',
          'Partial overlap and near duplicates can remain after normalized exact deduplication',
          'Full sweep has not started'])
    (OUT/'budget_plan.json').write_text(json.dumps(result,indent=2))
    with (OUT/'timings_by_mask.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(timing_rows[0]));w.writeheader();w.writerows(timing_rows)
    print('BUDGET_PLAN_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['assumptions','scenarios']}),flush=True)
    print('SCENARIOS',json.dumps(scenarios),flush=True)

if __name__=='__main__':main()
