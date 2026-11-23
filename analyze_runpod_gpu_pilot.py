"""Compare measured singleton LLaDA extraction throughput on the two RunPods."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
folder=root/'runpod_gpu_pilot_results'
a=json.loads((folder/'a100.json').read_text())
r=json.loads((folder/'rtx4090.json').read_text())
assert a['status']==r['status']=='complete'
assert a['inputs_sha256']==r['inputs_sha256']
assert a['code_sha256']==r['code_sha256']
assert a['model_revision']==r['model_revision']
assert a['num_samples']==r['num_samples']==10
assert a['model_forward_batch_size']==r['model_forward_batch_size']==1
assert a['estimator_dtype']==r['estimator_dtype']=='float64'
assert len(a['results'])==len(r['results'])==16
fields=['repeat','group','filtered_index','mask','scheme']
assert [tuple(x[f] for f in fields) for x in a['results']]==[tuple(x[f] for f in fields) for x in r['results']]
ta=sum(x['seconds'] for x in a['results']);tr=sum(x['seconds'] for x in r['results'])
advantage=ta*1.79/(tr*.89)
summary=dict(recommended_gpu='RTX 4090',a100_evaluation_seconds=ta,rtx4090_evaluation_seconds=tr,
    rtx4090_throughput_speedup=ta/tr,rtx4090_results_per_dollar_advantage=advantage,
    prices_from_user=dict(a100_per_hour=1.79,rtx4090_per_hour=.89),
    caveat='Pinned Base checkpoint, two articles and two mask patterns; not a fine-tuned-checkpoint benchmark')
(folder/'comparison.json').write_text(json.dumps(summary,indent=2))
text='''# RunPod GPU comparison — 1 November 2026

The RTX 4090 was faster and cheaper for the measured pipeline on these two RunPod instances. This is an inference-only hardware pilot, not the full extraction experiment.

| Measurement | A100 SXM4 80 GB | RTX 4090 24 GB |
|---|---:|---:|
'''
text+=f'| Quoted GPU price/hour | $1.79 | $0.89 |\n| Total for 16 evaluations | {ta:.2f} s | {tr:.2f} s |\n'
for scheme in ['low-confidence','threshold']:
    text+=f"| {scheme}: mean seconds/evaluation | {a['summary'][scheme]['mean_seconds']:.2f} | {r['summary'][scheme]['mean_seconds']:.2f} |\n"
text+=f"| Peak allocated GPU memory | {max(x['peak_allocated_gib'] for x in a['results']):.2f} GiB | {max(x['peak_allocated_gib'] for x in r['results']):.2f} GiB |\n"
text+=f"| Peak reserved GPU memory | {max(x['peak_reserved_gib'] for x in a['results']):.2f} GiB | {max(x['peak_reserved_gib'] for x in r['results']):.2f} GiB |\n"
text+=f'''
Across this case mix, the RTX delivered **{ta/tr:.2f}x throughput** and **{advantage:.2f}x as many evaluations per GPU dollar**. Prices are the user's quoted rental rates; installation/download time, idle time and storage charges are excluded from these steady evaluation measurements.

Both pods ran the same unchanged `probabilistic_extraction.py` and `get_log_likelihood.py`, verified by source SHA-256. The instrumentation code hash, public model revision and selected input token hashes also match. Both use PyTorch 2.8.0+cu128 and Transformers 4.46.3, four CPU threads, BF16 model weights, FP64 estimator arithmetic, model batch size one, full decoding, temperature 1, ten trajectories and state caching. Threshold is 0.9. The environment sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` for the estimator's existing deterministic-evaluation requirement; deterministic execution was not disabled to make the benchmark run.

The pilot uses the pinned **untrained LLaDA Base** (`0f2787f2d87eac5eed8a087d5ecd24277e6255b2`). No authenticated Colab runtime was available for Drive access, so the fine-tuned checkpoint has not been timed. Input articles are from the frozen BBC dataset revision and original global even/odd membership rule, with normalized training overlaps excluded from controls. One training article (filtered index 1734) and one clean odd control (1695) supply one 100-token window each. Suffix-50 and alternating-50 masks, both remasking schemes, and two matched timing repetitions give sixteen evaluations per GPU. These are not all twelve masks or a representative sample of the whole corpus.

The probabilities pass the estimator's numerical checks, but BF16 execution on different GPU architectures does not yield identical logits or sampled trajectories even with the same seed. The GPU reports record different probabilities and forward-call counts. This is not a cross-hardware bitwise-equivalence result. For the scientific sweep, use a consistent GPU/backend policy, retain seeds/configuration/provenance, and recheck performance and numerical behavior on the fine-tuned model before deciding the final subset or mixing hardware types.

The A100 evaluated {sum(x['model_forward_calls'] for x in a['results']):,} model states and the RTX {sum(x['model_forward_calls'] for x in r['results']):,}; the RTX's advantage therefore was not caused by evaluating fewer states. Host CPU and GPU platform differences are part of these measured pod results; this is not a claim that every RTX 4090 setup universally outperforms every A100.

Both pilot processes completed. No production sweep, code adaptation to fine-tuned checkpoints, Google Drive result upload, pod stop or pod termination was performed. No large file was copied to Windows. Raw per-evaluation results and source hashes are in the neighboring JSON files.
'''
(root/'RUNPOD_GPU_COMPARISON_20261101.md').write_text(text,encoding='utf-8')
print(json.dumps(summary,indent=2))
