"""Forecast pod time from bounded pilot observations; no training/inference runs."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
PRICE=5.66


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    raw=read_jsonl(RESULTS/'pilot_rows.jsonl')
    # The pilot uses repeated calibration cases. Keep distinct scientific timing tasks.
    rows={r['id']:r for r in raw}
    rows=list(rows.values())
    primary=read_jsonl(RESULTS/'stream_jobs.jsonl')
    stats=[]
    for purpose in sorted({r.get('purpose','calibration') for r in rows}):
        groups=defaultdict(list)
        for r in rows:
            if r.get('purpose','calibration')==purpose:
                groups[(r['method'],r['scheme'],r['temperature'],r['samples'])].append(r['seconds'])
        for (method,scheme,temp,n),values in sorted(groups.items()):
            stats.append(dict(purpose=purpose,method=method,scheme=scheme,temperature=temp,samples=n,
                              pilot_cases=len(values),mean_seconds=statistics.mean(values),
                              min_seconds=min(values),max_seconds=max(values)))
    with (ROOT/'pilot_timing_summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)

    def measured(purpose,method,scheme,temp,n):
        values=[r['seconds'] for r in rows if r.get('purpose')==purpose and r['method']==method
                and r['scheme']==scheme and r['temperature']==temp and r['samples']==n]
        return statistics.mean(values) if values else None

    def coverage(purpose,method,scheme,temp,n):
        selected=[r for r in rows if r.get('purpose')==purpose and r['method']==method
                  and r['scheme']==scheme and r['temperature']==temp and r['samples']==n]
        assert selected,(purpose,method,scheme,temp,n)
        weights=sum(r['source_weight'] for r in selected)
        return sum(r['seconds']*r['source_weight'] for r in selected)/weights

    def exponent(method,scheme,temp):
        groups=defaultdict(list)
        for r in rows:
            if r.get('purpose')=='runtime' and r['method']==method and r['scheme']==scheme and r['temperature']==temp:
                groups[(r['document'],r['window_number'],r['mask_id'])].append(r)
        slopes=[]
        for group in groups.values():
            group=sorted(group,key=lambda r:r['samples'])
            for a,b in zip(group,group[1:]):
                if b['samples']>a['samples'] and min(a['seconds'],b['seconds'])>1:
                    slope=math.log(b['seconds']/a['seconds'])/math.log(b['samples']/a['samples'])
                    slopes.append(max(0,min(1,slope)))
        if not slopes:
            # Conservative fallback for incomplete pilot budget pairs.
            return (.35 if temp==.5 else (.95 if method=='path_sampling' else .85)),0
        return statistics.median(slopes),len(slopes)

    gpu_work=defaultdict(float)
    baseline_gpu_work=defaultdict(float)
    assumptions=[]
    allocation_path=RESULTS/'allocation_full8_summary.json'
    allocation=json.loads(allocation_path.read_text()) if allocation_path.exists() else None
    allocation_gain=1.
    if allocation and allocation.get('bit_identical') and allocation.get('all_eight_slices'):
        # Discount the short-probe incremental gain; retain the separate 10% scheduling reserve.
        allocation_gain=min(1.25,1+.75*max(0,allocation['throughput_speedup']-1))
    for temp in (.5,2.):
        for scheme in ('low-confidence','threshold'):
            sts_alpha,sts_pairs=exponent('path_sampling',scheme,temp)
            mc_alpha,mc_pairs=exponent('monte-carlo',scheme,temp)
            primary_sts500=measured('validation','path_sampling',scheme,temp,500)
            coverage_sts500=coverage('coverage_validation','path_sampling',scheme,temp,10)*50**sts_alpha
            sts500=statistics.mean([x for x in (primary_sts500,coverage_sts500) if x is not None])
            primary_mc500=measured('validation','monte-carlo',scheme,temp,500)
            coverage_mc500=coverage('coverage_validation_mc','monte-carlo',scheme,temp,125)*4**mc_alpha
            mc500=statistics.mean([x for x in (primary_mc500,coverage_mc500) if x is not None])
            prefetch=RESULTS/'prefetch_optimization.json'
            use_prefetch=prefetch.exists() and json.loads(prefetch.read_text()).get('validated')
            sts_gain=1.35 if temp==2 and use_prefetch else 1.
            mc_opt=RESULTS/'prefetch_mc_optimization.json'
            mc_gain=1.5 if temp==2 and mc_opt.exists() and json.loads(mc_opt.read_text()).get('validated') else 1.
            cross_gain=allocation_gain if temp==2 else 1.
            baseline_gpu_work['STS versus MC']+=3600*(sts500+mc500*10**mc_alpha)
            gpu_work['STS versus MC']+=3600*(sts500/sts_gain+mc500*10**mc_alpha/mc_gain)/cross_gain
            for case,population,purpose in [
                ('Geometry and LLaDA AR comparison',22788,'coverage_application'),
                ('Post-release controls',15384,'coverage_negative')]:
                seconds=population*coverage(purpose,'path_sampling',scheme,temp,10)
                baseline_gpu_work[case]+=seconds
                gpu_work[case]+=seconds/sts_gain/cross_gain
            assumptions.append(dict(temperature=temp,scheme=scheme,STS_scaling_exponent=sts_alpha,
                  STS_observed_budget_pairs=sts_pairs,MC_scaling_exponent=mc_alpha,
                  MC_observed_budget_pairs=mc_pairs,validation_STS500_seconds=sts500,
                  validation_MC5000_predicted_seconds=mc500*10**mc_alpha,
                  primary_STS500_seconds=primary_sts500,coverage_STS500_seconds=coverage_sts500,
                  primary_MC500_seconds=primary_mc500,coverage_MC500_seconds=coverage_mc500,
                  STS_prefetch_forecast_speedup=sts_gain,MC_prefetch_forecast_speedup=mc_gain,
                  cross_case_allocation_forecast_speedup=cross_gain))
    exact=[]
    for job in primary:
        if 'error' not in job and any(r.get('purpose')=='exact_shared' for r in job['rows']):
            exact.append(sum(r['seconds'] for r in job['rows']))
    assert exact
    gpu_work['Exact versus STS (including T1 threshold)']=12000*statistics.mean(exact)
    baseline_gpu_work['Exact versus STS (including T1 threshold)']=gpu_work['Exact versus STS (including T1 threshold)']

    random_groups=defaultdict(list)
    for r in rows:
        if r.get('purpose')=='random':random_groups[(r['temperature'],r['batch_size'])].append(r)
    available_batches=sorted({b for t,b in random_groups})
    chosen_batch=max(available_batches)
    for temp in (.5,2.):
        candidates=random_groups.get((temp,chosen_batch))
        if not candidates:candidates=[r for (t,b),rr in random_groups.items() if t==temp for r in rr]
        if not candidates:candidates=[r for rr in random_groups.values() for r in rr]
        assert candidates
        estimates=[]
        for r in candidates:
            # All 50 steps are evaluated. Scale transformer batches, not just path count.
            factor=(1+49*math.ceil(500/r['batch_size']))/(1+49*math.ceil(r['samples']/r['batch_size']))
            estimates.append(r['seconds']*factor)
        gpu_work['Random remasking']+=3600*statistics.mean(estimates)
    baseline_gpu_work['Random remasking']=gpu_work['Random remasking']
    joint=RESULTS/'joint_random_optimization.json'
    if joint.exists() and json.loads(joint.read_text()).get('validated'):
        gpu_work['Random remasking']/=1.9  # Below the measured approximately 2x gain.
    greedy=[r['seconds'] for r in rows if r.get('purpose')=='greedy']
    gpu_work['Greedy-confidence remasking']=7200*statistics.mean(greedy) if greedy else 7200*3.5
    ar_files=list(RESULTS.glob('ar_pilot_*.json'))
    assert len(ar_files)==4,ar_files
    gpu_work['Four AR baselines']=sum(json.loads(p.read_text())['projected_gpu_seconds'] for p in ar_files)
    for key,value in gpu_work.items():baseline_gpu_work.setdefault(key,value)

    output=[]
    for case,seconds in gpu_work.items():
        hours=seconds/7.2/3600  # 10% utilization/scheduling allowance versus measured eight-way scaling.
        output.append(dict(experiment=case,estimated_GPU_hours=seconds/3600,
                           estimated_pod_hours=hours,estimated_cost_USD=hours*PRICE,
                           baseline_without_prefetch_or_joint_random_USD=baseline_gpu_work[case]/7.2/3600*PRICE))
    with (ROOT/'budget_estimate.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    total=sum(r['estimated_cost_USD'] for r in output)
    report=dict(price_per_pod_hour=PRICE,effective_GPUs=7.2,measured_exact_bundle_cases=len(exact),
          random_batch_size=chosen_batch,completed_budget_observations=len(rows),
          estimated_cost_USD=total,estimated_pod_hours=total/PRICE,
          native_baseline_cost_USD=sum(baseline_gpu_work.values())/7.2/3600*PRICE,
          cross_case_allocation_forecast_speedup=allocation_gain,
          planning_range_USD=[total*.6,total*1.6],runtime_study_excluded=True,
          rows=output,scaling_assumptions=assumptions,
          caveats=['Planning range is sensitivity analysis, not a statistical confidence interval.',
                   'Validation MC5000 times extrapolate measured MC500 using runtime budget scaling.',
                   'No wall-clock study is included; it is reserved for A100 hardware.',
                   'Source coverage samples one mask/window per document; within-document variation remains.',
                   'T2 confidence STS includes a conservative 1.35x prefetch gain, verified on short seeded checks.',
                   'T2 ordinary MC includes a conservative 1.5x prefetch gain, verified against identical 125-trial outcomes.',
                   'Random-temperature sharing includes a conservative 1.9x gain, verified against separate complete paths.',
                   'Cross-case allocation gain is discounted from an eight-slice T2 short probe; applies only to T2 confidence sampled workloads, beyond existing prefetch gains.',
                   'Central validation costs average the larger primary probes and source-stratified short-probe extrapolations.',
                   'No complete replication has been launched.'])
    (ROOT/'budget_estimate.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
