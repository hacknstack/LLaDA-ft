"""Write the completed probe and saved-model test evidence summary."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
folder=HERE/'ar_probe_results'
audit=json.loads((folder/'mistral_olmo_run_audit.json').read_text())
text='''# AR training verification and Llama feasibility — 27 October 2026

Verified from a new authenticated Colab A100-SXM4-80GB runtime. No production Llama training was started, no existing model was modified/deleted and no large file was written to Windows. Saved-model weights were downloaded only onto Colab's local disk for loading/inference checks.

## Llama feasibility and notebooks

Both approved original base repositories passed actual full-parameter 4096-token training probes with BF16 computation, FP32 accumulated gradients, CPU FP32 master AdamW and non-reentrant activation checkpointing. Each passed two full 16-example updates, an actual-size final-window update including the short tail, finite losses/norms, changed FP32 masters and one-token zero-loss handling.

| Model | Native train tokens | Three-epoch updates | Peak GPU allocated / reserved (GiB) | Peak host process RAM (GiB) | Rough compute-only hours |
|---|---:|---:|---:|---:|---:|
'''
for key,title in [('llama2_7b','Llama 2 7B'),('llama3_8b','Llama 3 8B')]:
    p=json.loads((folder/(key+'_4096.json')).read_text())
    assert p['status']=='ok' and len(p['updates'])==3
    text+=f"| {title} | {p['train_tokens']:,} | {p['three_epoch_updates']} | {p['peak_gpu_allocated_gib']:.2f} / {p['peak_gpu_reserved_gib']:.2f} | {p['peak_cpu_rss_gib']:.2f} | {p['three_epoch_compute_hours']:.2f} |\n"
text+='''
The standalone A100/high-RAM notebooks are `Llama2_7B_BBC_3epochs_Colab.ipynb` and `Llama3_8B_BBC_3epochs_Colab.ipynb`. Each defaults to exactly one final model save, no periodic/epoch model snapshots or optimizer snapshots, and supports `SAVE_FINAL_MODEL=False` for text-only output. Hugging Face login uses a widget; no credentials are embedded. Compute-only projections exclude initialization, diagnostics and the final upload and use one warmed timing per model.

Both notebook data-preparation paths were executed successfully against the frozen articles and controls. Native token counts/hashes match the measured probes. CPU checks passed for loss/zero-target behavior, schedules including totals 177/207, native safetensors validation, completion-marker ordering, preserving local staging after a failed upload, and exactly one guarded export outside the training loops. Notebook format and all Python code cells were validated; long three-epoch Llama runs are left for the user to execute.

## Completed Mistral and OLMo runs

| Model | Epochs / updates | Weight size (GB) | Train diagnostic CE: Base → final | Clean-control CE: Base → final | Final model |
|---|---:|---:|---:|---:|---|
'''
for run in sorted(audit['runs'],key=lambda r:r['config']['model_key']):
    assert run['verified'] and not run['errors']
    config=run['config'];key=config['model_key']
    smoke=json.loads((folder/(key+'_saved_model_smoke.json')).read_text())
    assert smoke['success'] and smoke['finite_forward_and_loss'] and smoke['trained_values_differ_from_bf16_base']
    assert smoke['run']==run['run']
    first=next(d for d in run['diagnostics'] if d['epoch']==0)['means']
    final=next(d for d in run['diagnostics'] if d['epoch']==3)['means']
    label='Mistral 7B v0.1' if key.startswith('mistral') else 'OLMo 7B'
    text+=f"| {label} | 3 / {run['status']['update']} | {run['weight_bytes']/1e9:.2f} | {first['train']:.4f} → {final['train']:.4f} | {first['clean_odd_controls']:.4f} → {final['clean_odd_controls']:.4f} | [Open Drive]({run['folder_url']}) |\n"
    assert run['data_manifest']['frozen_article_file_sha256']==config['frozen_article_file_sha256']
    assert run['data_manifest']['token_files']['train_tokens.npy']['sha256']==config['expected_native_token_file_sha256']
text+='''
For each completed run, the independent Drive API journal reports epoch 3 and the expected total updates. All final file IDs, sizes, MD5 and server SHA-256 match the upload receipt; the completion manifest and model index reference the correct existing shards. The frozen article hash and native train-token hash match the specified corpus. Every file downloaded for testing was SHA-256 verified again locally.

Both complete saved models were loaded with their saved configuration/tokenizer into native Transformers 4.46.3, with expected parameter counts. Both produced finite logits/CE and a 16-token greedy continuation from the same short generic prompt. Two parameter slices (512 values each, first-layer query projection and MLP down projection) were compared against the pinned original base safetensors via bounded HTTP range reads, after matching the actual BF16 initialization conversion. Both models have changed values in both slices. This provides direct evidence of saved weight updates, in addition to the training records.

Control diagnostics rise after epoch 1 while train diagnostics continue falling. That is consistent with increased fitting to the selected training corpus, not a serialization failure. It is not itself a proof or quantitative measure of verbatim memorization. Native-token CE should not be compared numerically across different tokenizers/objectives. A shared extraction/membership protocol remains separate work.

All tests above passed. Raw evidence is in `ar_probe_results/mistral_olmo_run_audit.json`, each model's `*_saved_model_smoke.json`, the Llama `*_4096.json` probes and the dated auxiliary audits. Original OLMo's 4096 extrapolation beyond its native 2048 pretraining context remains an explicitly documented comparison limitation.
'''
(HERE/'AR_TRAINING_TESTS_20261027.md').write_text(text,encoding='utf-8')
print('WROTE_COMPLETED_AR_TEST_REPORT')
