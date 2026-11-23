"""Choose measured safe allocations, estimate three-GPU makespan, gate launch."""
import itertools
import json
from pathlib import Path

ROOT=Path('/workspace/bbc-exposure')
KEYS=['llada8b','llama2_7b','llama3_8b','mistral7b_v01','olmo7b']


def main():
    assert (ROOT/'audits/microbatch_sweep_complete.json').exists()
    assert (ROOT/'audits/final_concurrency_complete.json').exists()
    checks={}
    for name in ['async_optimizer_validation','foreach_optimizer_validation','batch_objective_validation']:
        checks[name]=json.loads((ROOT/'audits'/(name+'.json')).read_text())['passed']
    native=json.loads((ROOT/'audits/all_native_input_audits.json').read_text())
    assert all(checks.values()) and all(native[k]['passed'] for k in KEYS)
    rows={}
    for key in KEYS:
        candidates=[]
        for p in (ROOT/'audits').glob('probe_async_mb*_'+key+'.json'):
            r=json.loads(p.read_text())
            if r.get('status')=='passed' and r['peak_gpu_reserved_gib']<75.5:candidates.append(r)
        assert candidates,key
        fastest=min(r['warm_full_update_seconds'] for r in candidates)
        candidate=min((r for r in candidates if r['warm_full_update_seconds']<=fastest*1.05),key=lambda r:r['physical_microbatch'])
        final_path=ROOT/'audits'/('probe_final_'+key+'.json')
        if final_path.exists():
            final=json.loads(final_path.read_text())
            assert final['status']=='passed' and final['peak_gpu_reserved_gib']<75.5
            candidate=final
        config=json.loads((ROOT/'inputs'/key/'training_config.json').read_text())
        original=json.loads((ROOT/'audits'/('probe_initial_'+key+'.json')).read_text())
        warm=candidate['warm_full_update_seconds']
        tail=candidate['updates'][-1]['total_seconds']
        # All candidate probes include the actual final partial update.
        seconds=(config['total_updates']-1)*warm+tail
        row=dict(model_key=key,microbatch=candidate['physical_microbatch'],accumulation_microsteps=16//candidate['physical_microbatch'],
             effective_batch_sequences=16,cpu_threads=12,async_pinned_transfers=True,foreach=candidate.get('foreach',False),
             tokens=config['actual_train_tokens'],updates=config['total_updates'],warm_update_seconds=warm,
             projected_training_seconds=seconds,projected_training_hours=seconds/3600,
             original_update_seconds=original['warm_full_update_seconds'],
             estimated_update_speedup=original['warm_full_update_seconds']/warm,
             peak_gpu_allocated_gib=candidate['peak_gpu_allocated_gib'],peak_gpu_reserved_gib=candidate['peak_gpu_reserved_gib'],
             peak_host_rss_gib=candidate['peak_host_rss_gib'],measured_concurrent_validation=final_path.exists(),
             selected_probe_tag=candidate['tag'],worst_case_mask_stress=candidate.get('worst_case_mask_stress',False))
        same_batch=[r for r in candidates if r['physical_microbatch']==candidate['physical_microbatch']]+[candidate]
        row['peak_gpu_reserved_gib']=max(r['peak_gpu_reserved_gib'] for r in same_batch)
        row['peak_gpu_allocated_gib']=max(r['peak_gpu_allocated_gib'] for r in same_batch)
        rows[key]=row
    # Enumerate all assignments: exact minimum makespan for these five estimates.
    best=None
    for assignment in itertools.product(range(3),repeat=5):
        loads=[0.,0.,0.]
        for key,gpu in zip(KEYS,assignment):loads[gpu]+=rows[key]['projected_training_seconds']
        score=(max(loads),sum(x*x for x in loads),assignment)
        if best is None or score<best[0]:best=(score,assignment,loads)
    queues={g:[] for g in range(3)}
    for key,gpu in zip(KEYS,best[1]):queues[gpu].append(key)
    for queue in queues.values():queue.sort(key=lambda k:-rows[k]['projected_training_seconds'])
    compute=max(best[2])/3600
    plan=dict(status='PREPARED_NOT_TRAINING',models=rows,gpu_queues=queues,
        estimated_gpu_compute_hours=sum(r['projected_training_hours'] for r in rows.values()),
        estimated_three_gpu_training_hours=compute,estimated_total_hours=compute*1.10+.25,
        planning_range_total_hours=[compute*.90+.15,compute*1.35+.50],
        estimate_method='All 243 three-GPU assignments evaluated for the five model estimates; 10% training/diagnostic/serialization reserve plus 15-minute final-upload allowance.',
        gpu_release='Compute process exits after final serialization; CPU-only upload overlaps the next model on that GPU.',
        invariant_settings='Full parameters, native objectives, 4096 tokens, BF16 model, FP32 gradient accumulation/master/moments, AdamW settings, effective batch 16.',
        numerical_scope='Pinned-copy and foreach optimizations match reference on a four-update test network. Larger physical batches can change rounding and RNG assignment; no full-run bitwise equivalence is claimed.',
        estimate_caveats=['Warm updates are short-probe estimates, not a completed training run.',
          'CPU NUMA traffic and simultaneous uploads can change throughput.',
          'Full-size final serialization and Drive throughput have not been benchmarked on this pod.',
          'Planning range is a sensitivity range, not a statistical confidence interval.',
          'Largest LLaDA masking case is checked in the final resource probe when indicated.'])
    (ROOT/'protocol/allocation_plan.json').write_text(json.dumps(plan,indent=2))
    locations_path=ROOT/'protocol/output_locations.json'
    drive_path=ROOT/'audits/drive_preflight.json'
    drive_ok=drive_path.exists() and json.loads(drive_path.read_text()).get('passed',False)
    blockers=[] if drive_ok else ['Renewed Drive authorization, quota and verified upload preflight pending.']
    smoke_path=ROOT/'audits/production_path_smoke.json'
    smoke_ok=smoke_path.exists() and all(r['passed'] for r in json.loads(smoke_path.read_text()).values())
    if not smoke_ok:blockers.append('Actual training-loop smoke tests pending.')
    readiness=dict(ready_to_launch=not blockers,models=KEYS,blockers=blockers,
          run_name=json.loads(locations_path.read_text())['run_name'] if locations_path.exists() else None,
          exposure_counts_verified=True,all_models_full_parameter_probes_passed=True,
          optimizer_numerical_checks=checks,production_training_started=False)
    (ROOT/'protocol/readiness.json').write_text(json.dumps(readiness,indent=2))
    print(json.dumps(dict(total_hours=plan['estimated_total_hours'],range_hours=plan['planning_range_total_hours'],
          gpu_queues=queues,models=rows,readiness=readiness),indent=2))


if __name__=='__main__':main()
