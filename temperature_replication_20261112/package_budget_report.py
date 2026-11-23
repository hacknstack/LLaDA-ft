import csv
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parent
DOWNLOADS=Path.home()/'Downloads'


def main():
    budget=json.loads((ROOT/'budget_estimate.json').read_text())
    cases=[]
    def add(name,cohort,temp,scheme,method,n,masked,rows,reference,model='llada-base'):
        with (DOWNLOADS/reference).open(newline='',encoding='utf-8-sig') as handle:header=next(csv.reader(handle))
        cases.append(dict(file=name+'.csv',cohort=cohort,temperature=temp,model=model,
              scheme=scheme,estimator=method,samples_per_estimate=n,masked_tokens=masked,
              expected_rows=rows,format_reference=reference,header_json=json.dumps(header)))
    for temp,tag in [(.5,'temp0p5'),(2.,'temp2')]:
        for scheme,short in [('low-confidence','LowConf'),('threshold','fast')]:
            add(f'validation_{tag}_{scheme}_STS500','validation_300',temp,scheme,'path_sampling',500,50,3600,
                f'ASTRA_{short}STS_samples500_estimators_with_logs.csv')
            add(f'validation_{tag}_{scheme}_MC5000','validation_300',temp,scheme,'monte-carlo',5000,50,3600,
                'ASTRA_LowConfMC_samples5000_avg.csv' if short=='LowConf' else 'ASTRA_fastMC_samples5000_pz.csv')
            add(f'exact_comparison_{tag}_{scheme}_exact','exact_1000',temp,scheme,'exact',0,10,12000,
                'ASTRA_LowConfexact_pz.csv' if short=='LowConf' else 'ASTRA_Fastexact_pz.csv')
            add(f'exact_comparison_{tag}_{scheme}_STS500','exact_1000',temp,scheme,'path_sampling',500,10,12000,
                'ASTRA_LowConfTrajSamplingBis_samples500_estimators_with_logs.csv' if short=='LowConf'
                else 'ASTRA_LowConfTrajSamplingFastBis_samples500_estimators_with_logs.csv')
            old='low-confidence' if scheme=='low-confidence' else 'fast-dllm'
            add(f'application_{tag}_{scheme}_STS10','application_csv',temp,scheme,'path_sampling',10,50,22788,
                f'Application_path_sampling_{old}_samples100_pz.csv')
            add(f'negative_{tag}_{scheme}_STS10','negative_csv',temp,scheme,'path_sampling',10,50,15384,
                f'Negative_path_sampling_{old}_samples100_pz.csv')
        add(f'remasking_{tag}_random500','validation_300',temp,'random','path_sampling',500,50,3600,
            'ASTRA_RANDOMSTS_samples500_pz.csv')
        add(f'remasking_{tag}_greedy','validation_300',temp,'greedy-confidence','exact_path',0,50,3600,
            'ASTRA_DUEL_pz.csv')
        for key,reference in [('llama2_7b','Llama 2 7B.csv'),('llama3_8b','Llama 3 8B.csv'),
                              ('mistral7b_v01','Mistral-7B.csv'),('olmo7b','OLMo-7B.csv')]:
            add(f'AR_{key}_{tag}','application_csv',temp,'suffix','exact_AR',0,50,1899,reference,model=key)
    add('exact_comparison_temp1_threshold_exact','exact_1000',1.,'threshold','exact',0,10,12000,'ASTRA_Fastexact_pz.csv')
    add('exact_comparison_temp1_threshold_STS500','exact_1000',1.,'threshold','path_sampling',500,10,12000,
        'ASTRA_LowConfTrajSamplingFastBis_samples500_estimators_with_logs.csv')
    assert len(cases)==38
    assert sum(c['expected_rows'] for c in cases)==331080
    with (ROOT/'planned_csv_outputs.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    rows=budget['rows'];by_name={r['experiment']:r for r in rows}
    smaller=sum(r['estimated_cost_USD']/(3 if r['experiment'] in ('STS versus MC','Random remasking','Greedy-confidence remasking') else 1) for r in rows)
    lines=['# Temperature replication: measured budget plan','',
        '**Status: the benchmark is complete; the full experiment has not started.**','',
        f"At the supplied $5.66/hour pod price, the central estimate is **${budget['estimated_cost_USD']:,.0f}**, "
        f"or **{budget['estimated_pod_hours']:,.0f} pod-hours ({budget['estimated_pod_hours']/24:.1f} continuous days)**.",
        f"The planning sensitivity range is ${budget['planning_range_USD'][0]:,.0f}–${budget['planning_range_USD'][1]:,.0f}. "
        'It reflects configuration variation and extrapolation, not a statistical confidence interval.','',
        '**The paper wall-clock study is excluded and reserved for A100 hardware.** The five-million-trial study is also excluded. '
        'MIG durations are budget and optimization observations only; they are not A100 runtime results.','',
        '| Experiment | Estimated pod-hours | Estimated cost |','|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['experiment']} | {r['estimated_pod_hours']:.1f} | ${r['estimated_cost_USD']:,.0f} |")
    lines.extend(['',
        'The plan retains the original 300-window correctness/remasking cohort, 1,000-window exact cohort, '
        '1,899 application starts, and 15,384 negative-control window-mask pairs. Both temperatures are 0.5 and 2; '
        'threshold exact/STS additionally gets a fresh temperature-1 pass. Correctness uses 500 STS / 5,000 MC; '
        'exact validation uses 500 STS with 10 masked tokens; application and controls use 10 STS; AR remains exact.','',
        'planned_csv_outputs.csv lists all 38 planned scientific CSVs and their exact header templates from Downloads '
        '(331,080 aggregate rows). Decoders and temperatures are separated by filename. AR mask fields retain the '
        'original CSV convention. No scientific CSV results are claimed complete by this budget package.','',
        'The PDF 300-, 1,000-, and 100-window index lists exactly match the supplied CSVs. The CSVs and notebook '
        'mask boundaries are authoritative. All 1,899 application starts have 100 valid native tokens under all '
        'four AR tokenizers. All four AR models passed 80-window/two-temperature probes.','',
        'The pod has eight 23.625-GiB MIG slices across four RTX PRO 6000 cards. Explicit MIG UUID assignment '
        'exposes all eight to separate persistent workers. A repeated calibration batch took 54.9 seconds alone '
        'and 54.6 seconds with eight workers simultaneously. CPU thread counts 2/3/4 had similar throughput; '
        'two threads per worker leaves CPU headroom. The forecast allows 10% scheduling/utilization overhead.','',
        'Exact-state native-logit reuse reduced the checked five-exact/five-STS bundle from 394 to 130 seconds '
        'with identical probabilities/log probabilities; that cache benefit is included in measured exact-bundle costs. '
        'State prefetch batching preserved the checked individual STS log weights and MC outcomes. The forecast '
        'uses conservative 1.35x T=2 STS and 1.5x T=2 MC gains. Only batch shapes 1/2/16 are used by that prototype; '
        'other tested shapes sometimes changed native logits and are not assumed interchangeable. '
        'The bounded lookahead holds at most 16 projected states instead of thousands.','',
        'Cross-case allocation now serves four independent estimator contexts from each resident model, '
        'combining ready inference requests across contexts. The all-eight-slice probe improved throughput '
        'by 17.6% with bit-identical checked results. The forecast discounts this incremental gain to '
        '1.132x and applies it only to T=2 confidence sampled workloads, retaining the 10% utilization reserve. '
        'See ALLOCATION_PLAN.md for validation scope and remaining orchestration work.','',
        'Random reveal paths are independent of temperature, so the same 500 complete 50-step paths can be '
        'scored at both temperatures from shared forwards. The paired prototype matched both separate runs '
        'exactly and measured approximately 2x speedup; the forecast uses 1.9x. No shortened-path or reduced-sample '
        'approximation is used. AR similarly shares each forward between temperatures.','',
        'The projected bill before the new prefetch and paired-random optimizations was '
        f"${budget['native_baseline_cost_USD']:,.0f}; exact-state caching was already included in that baseline. "
        'The extrapolation details, sample-count scaling assumptions, and measured case counts are in budget_estimate.json. '
        'MC5000 costs are forecasts from shorter measured cases and observed scaling; no long full-suite run was performed.','',
        f"A smaller alternative, **not adopted**, would use 100 rather than 300 windows for correctness and remasking "
        f"while retaining their sample counts, all masks, and the other cohorts. Its proportional central estimate is "
        f"approximately **${smaller:,.0f}**. A fixed, source-balanced subset would need to be chosen before launch. This "
        'would change the original study size.','',
        'All benchmark processes were checked stopped. The pod itself remains running and continues to be billed. '
        'No automatic pod-stop command has been issued. Full-run orchestration and renewable Drive upload '
        'authorization still need preparation before the complete suite is launched. Model weights stay on the pod.'])
    (ROOT/'BUDGET_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive=DOWNLOADS/'temperature_replication_budget_plan.zip'
    audits=['ar_full_window_preflight.json','batch_shape_audit.json','cache_validation.json',
            'prefetch_path_lazy_validation.json','prefetch_mc_optimization.json','joint_random_optimization.json',
            'final_process_check.json','runtime_study_excluded.json','allocation_full8_summary.json']
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for name in ['BUDGET_REPORT.md','budget_estimate.csv','budget_estimate.json','planned_csv_outputs.csv',
                     'pilot_timing_summary.csv','COHORT_AUDIT.md','cohort_audit.json','PLAN.md','ALLOCATION_PLAN.md']:
            z.write(ROOT/name,name)
        for name in audits:
            p=ROOT/'results'/name
            if p.exists():z.write(p,'audits/'+name)
        for name in ['estimate_budget.py','audit_cohorts.py','pilot_prefetch.py','joint_random_temperatures.py',
                     'allocation_broker.py','pilot_allocation.py','full8_allocation.py']:
            z.write(ROOT/name,'implementation/'+name)
    print(json.dumps(dict(zip=str(archive),bytes=archive.stat().st_size,planned_files=len(cases),
                         planned_rows=sum(c['expected_rows'] for c in cases),smaller_option_USD=smaller),indent=2))


if __name__=='__main__':main()
