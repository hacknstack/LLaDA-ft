"""Append measurements to the protocol and freeze the proposed AR configurations."""
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
folder=HERE/'ar_probe_results'
reports=[json.loads(p.read_text()) for p in sorted(folder.glob('*_4096.json'))]
aux_path=folder/'auxiliary_audits.json'
aux=json.loads(aux_path.read_text()) if aux_path.exists() else {}
base=json.loads((HERE/'llada_training_config_v3.json').read_text())
configs=[]
rows=[]
pins=[]
storage=0
for report in reports:
    name=report['key']
    pins.append(f"- `{report['model_id']}`: `{report.get('model_revision','unresolved')}`.")
    if report['status']!='ok':
        rows.append(f"| {name} | {report['status']} | — | — | — | — |")
        continue
    updates=report['three_epoch_updates']
    rows.append(f"| {name} | Passed | {report['train_tokens']:,} | {updates} | {report['peak_gpu_allocated_gib']:.2f} / {report['peak_gpu_reserved_gib']:.2f} | {report['peak_cpu_rss_gib']:.2f} |")
    config=dict(base)
    for key in ['run_name','mask_token_id','minimum_mask_probability','model_path_override']:
        config.pop(key,None)
    config.update(model_id=report['model_id'],model_revision=report['model_revision'],
        run_name_template=name+'_bbc_ar_3ep_<unique UTC timestamp>',
        objective='causal next-token CE; per-sequence target mean then mean over actual accumulation window; length-1 contributes differentiable zero',
        attention='native causal SDPA',checkpointing='native whole decoder layer non-reentrant',
        sequence_length=4096, expected_tokens=report['train_tokens'],
        expected_blocks_per_epoch=report['blocks_per_epoch'],
        expected_updates_per_epoch=report['updates_per_epoch'],expected_total_updates=updates,
        warmup_updates=math.ceil(.05*updates),decay_updates=math.ceil(.1*updates),
        frozen_article_file_sha256=report['frozen_article_file_sha256'],
        expected_native_token_file_sha256=report['token_file_sha256'],
        native_eos_token_id=report['eos_token_id'],
        context_extrapolation=report['context_extrapolation'],
        original_context=report['native_config'].get('max_position_embeddings'),
        methods_filename='AR_COMPARISON_METHOD.md',
        status='PROPOSED_NOT_STARTED',
        checkpoint_validation='Validate AR native state tensor names/shapes/dtypes and index; server checksums; completion marker last',
        diagnostic_mask_draws_per_block=0,diagnostic_passes_per_block=1,
        storage_plan='Retain three epochs and one rolling snapshot per active model; check Drive API quota for overlapping replacement uploads; do not delete existing user snapshots')
    config['checkpoint_format']='BF16 native AR safetensors, tokenizer/configuration; local serialization, Drive API checksum verification'
    configs.append(config)
    storage+=report['model_tensor_bytes']

method=HERE/'AR_COMPARISON_METHOD.md'
text=method.read_text(encoding='utf-8').split('\n## Measured feasibility results\n')[0]
section='''
## Measured feasibility results

All sizes below are actual measurements in the existing Colab A100-SXM4-80GB runtime, which exposes 79.25 GiB VRAM and approximately 167 GiB host RAM. “Passed” means the complete three-update probe and the one-token test passed, not merely that a model loaded.

| Model | 4096 full-parameter probe | Native train tokens | Updates in 3 epochs | Peak GPU allocated / reserved (GiB) | Peak process host RAM (GiB) |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'\n'
for r in reports:
    if r['status']=='blocked':
        reason='request is awaiting author review' if 'awaiting a review' in r.get('error','') else 'account lacks model authorization'
        section+=f"\nLatest authenticated access check for `{r['key']}` at `{r['started_utc']}` returned HTTP 403: {reason}. No full-model forward/backward or optimizer test has run for this model.\n"
section+='\n### Pinned revisions\n\n'+'\n'.join(pins)+'\n\n### Timing and output size\n\n'
section+='| Model | Warm full update (seconds) | Approx. 3-epoch compute (hours) | One BF16 model (GB) |\n|---|---:|---:|---:|\n'
for r in reports:
    if r['status']=='ok':
        section+=f"| {r['key']} | {r['steady_full_update_seconds']:.1f} | {r['three_epoch_compute_hours']:.2f} | {r['model_tensor_bytes']/1e9:.2f} |\n"
section+='\nCompute projections use one warmed full-window measurement per model, scaled to the actual model-specific update count. They are rough estimates, not confidence intervals or total wall times. Initialization, native-token data preparation, diagnostics, local serialization, hashing, network upload and API retry time are extra. Final short accumulation windows can be faster. No trained AR weights were retained from these probes.\n'
if storage:
    section+=f'\nThe probed models together require about {storage/1e9:.2f} GB for one final BF16 snapshot each, or {3*storage/1e9:.2f} GB for three epoch snapshots each, before rolling snapshots and headroom. Existing LLaDA checkpoints are additional.\n'
if aux.get('drive_free_bytes') is not None:
    section+=f"\nAt the auxiliary audit, Drive API reported {aux['drive_free_bytes']/1e9:.2f} GB ({aux['drive_free_bytes']/2**30:.2f} GiB) free. This is a current account quota observation, not the mounted filesystem's reported capacity.\n"
if aux:
    section+='\n### Independent correctness/corpus audit\n\n'
    for k in ['objective_matches_native_hf','causal_future_invariance','one_token_zero_loss','full_and_remainder_optimizer_equivalence','frozen_articles_reproduce_canonical_llada_token_sha256','frozen_controls_reproduce_canonical_llada_token_sha256']:
        section+=f"- `{k}`: `{aux.get(k)}`.\n"
    section+='\nSource shard tensor dtypes, inspected without loading weight data:\n\n'
    for k,v in aux.get('source_weight_dtypes',{}).items():
        section+=f'- `{k}`: `{v}`.\n'
    section+='\nNative-token control stream sizes:\n\n'
    for k,v in aux.get('native_control_tokens',{}).items():
        section+=f'- `{k}`: {v:,} tokens across the same 3,345 clean odd-control articles.\n'
    section+='\nThe independent objective test matches the native Hugging Face shifted CE on a tiny Llama architecture; changing future inputs leaves earlier logits unchanged. The optimizer test compares full-batch CE against accumulated single-example CE across 16- and 3-example windows. These CPU checks complement actual full-model GPU probes; they do not replace those probes.\n'
section+='\n### Remaining decisions / evidence limits\n\n- Original OLMo at unscaled 4096 extrapolates beyond its original 2048 context. Computational feasibility does not establish the scientific effect of that extension.\n- The final-only notebooks retain one final model per run, plus text records, without deleting existing user snapshots. The earlier three-epoch-snapshot storage proposal is historical.\n- The final-only production trainer preserves the specified data/control/RNG/optimizer rules and validates native serialization. Mistral and OLMo production results are audited separately; Llama runs here remain feasibility probes until their notebooks are run.\n- Few-update probes alone do not establish full-run stability, checkpoint loading/inference or shared memorization results. Saved-model loading/inference checks, where performed, are recorded in the separate saved-run audit report.\n'
method.write_text(text+section,encoding='utf-8')
(HERE/'ar_training_protocol_proposals.json').write_text(json.dumps(configs,indent=2))
print('UPDATED_METHOD_AND_PROTOCOL_PROPOSALS',len(configs),'models passed')
