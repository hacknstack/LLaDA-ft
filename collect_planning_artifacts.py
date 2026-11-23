"""Collect small planning artifacts only; article text and weights stay on pod."""
import collections,csv,hashlib,json,subprocess,zipfile
from pathlib import Path

WORK=Path(__file__).resolve().parent
DEST=WORK/'runpod_planning_results/20261101_7x4090';DEST.mkdir(parents=True,exist_ok=True)
PRIVATE=Path.home()/'.codex/rclone-tools'
INSTALL=json.loads((PRIVATE/'install.json').read_text())
BASE=[INSTALL['executable'],'--config',str(PRIVATE/'research-drive-auth/rclone.conf'),'--log-level','ERROR','--checksum']
REMOTE='researchdrive:LLaDA-ft/sliding-window-planning/20261101_7x4090'
subprocess.run([*BASE,'copy',REMOTE,str(DEST),'--max-size','10M','--exclude','*.zip','--immutable'],check=True)
plan=json.loads((DEST/'budget_plan.json').read_text());rec=plan['recommendation']
speed=json.loads((DEST/'speed_summary.json').read_text())
docs=list(csv.DictReader((DEST/'documents_recommended.csv').open()))
assert collections.Counter(d['group'] for d in docs)=={'train':350,'control':350}
assert len({(d['group'],d['filtered_index']) for d in docs})==700
assert sum(len(json.loads(d['selected_windows'])) for d in docs)==rec['shared_window_starts']
assert all(len(json.loads(d['selected_windows']))<=8 and all(0<=x<int(d['common_windows']) for x in json.loads(d['selected_windows'])) for d in docs)
assert all(int(d['filtered_index'])%2==(d['group']=='control') for d in docs)
pilot=json.loads((DEST/'speed_selection.json').read_text())
assert not ({(d['group'],int(d['filtered_index'])) for d in docs}&{(d['group'],d['filtered_index']) for d in pilot})
assert not ({d['normalized_sha256'] for d in docs if d['group']=='train'}&{d['normalized_sha256'] for d in docs if d['group']=='control'})
meta=json.loads(subprocess.check_output([*BASE,'lsjson','researchdrive:LLaDA-ft/sliding-window-planning','--dirs-only'],text=True))
folder=next(r for r in meta if r['Name']=='20261101_7x4090')
url='https://drive.google.com/drive/folders/'+folder['ID']
strata=collections.defaultdict(dict)
for r in rec['stratum_counts']:strata[r['month']][r['length_bin']]=r['pairs']
month_lines='\n'.join('| '+month+' | '+' | '.join(str(counts.get(i,0)) for i in range(4))+' | '+str(sum(counts.values()))+' |' for month,counts in sorted(strata.items()))
options=[]
for item in [rec,*[s for s in plan['scenarios'] if s['windows_cap']!=8]]:
    label='Recommended' if item['scenario']=='recommended' else ('Complete article' if item['windows_cap'] is None else str(item['windows_cap'])+' windows')
    options.append(f"| {label} | {item['train_documents']} + {item['control_documents']} | {item['train_pct_all_unique']:.2f}% | {item['control_pct_all_unique']:.2f}% | {item['shared_window_starts']:,} | {item['nominal_pod_hours']:.1f} / {item['guarded_pod_hours']:.1f} h | ${item['guarded_cost']:.0f} |")
option_table='\n'.join(options)
ar_table='\n'.join(f'| {key} | {value:.4f} s | {value*rec["shared_window_starts"]/60:.1f} min |' for key,value in plan['ar_seconds_per_window_by_model'].items())
masks=[
 '1,3,5,…,99','1–10,21–30,41–50,61–70,81–90','1–24 and 75–100','1–25 and 51–75','1–50',
 '2,4,6,…,100','3–4,7–8,…,99–100','6–10,16–20,…,96–100','11–20,31–40,51–60,71–80,91–100',
 '25–74','26–50 and 76–100','51–100']
mask_table='\n'.join(f'| {i} | {text} |' for i,text in enumerate(masks,1))
text=f'''# Sliding-window experiment plan — 1 November 2026

Recommendation: the same **350 distinct training articles and 350 distinct excluded controls** for all five fine-tuned models, with up to **eight matched windows per article**. LLaDA runs **all twelve original masks**, with **ten threshold samples and ten low-confidence samples per mask**. AR models use exact suffix scoring. Detailed output stays disabled. This file is a plan; the full experiment has not started.

## Budget and measured throughput

At $6.40/hour, $490 buys 76.56 pod-hours before continuing usage. The measured core takes **{rec['nominal_pod_hours']:.2f} pod-hours**, approximately **${rec['nominal_pod_hours']*6.4:.2f}**. Applying a **25% slowdown allowance** gives **{rec['guarded_pod_hours']:.2f} hours / ${rec['guarded_cost']:.2f}**. An additional **$50 reserve** is kept for interruptions, completion and uploads. Usage since the budget message was **${plan['spent_since_budget_received']:.2f}** when this projection was generated. Recheck the actual balance at launch; the pod continues billing while idle.

The pilot used 16 articles spanning both groups, March–June and multiple length strata, with at least 180 words per article. All twelve masks were measured on eight articles at two locations each; the other eight supplied additional suffix-mask measurements. There were 416 representative LLaDA computations, 90 scaling-calibration computations, and 256 exact windows for each AR model. Each LLaDA computation used ten samples. Probability helpers were unchanged. The main cohort can include shorter eligible articles; the timing allowance is a planning contingency, not a statistical confidence bound or a guaranteed runtime ceiling.

Matched calibration: six computations took **66.55 s on one GPU**; seven copies of them took **67.57 s on seven GPUs** with four CPU threads per worker. The matched log probabilities were identical across GPUs and the two tested thread settings. The broader 416-computation run used **{speed['representative']['wall_seconds']:.2f} s wall time** and **{speed['representative']['gpu_seconds']:.2f} summed GPU-seconds**. Predictions use the measured aggregate equivalent of **{plan['effective_gpus_used_in_prediction']:.3f} continuously busy GPUs**, rather than assuming perfect seven-way scaling.

A complete window under all twelve masks and both schemes averaged **{plan['seconds_per_window_all_12_masks_both_schemes']['train']:.2f} GPU-seconds for training** and **{plan['seconds_per_window_all_12_masks_both_schemes']['control']:.2f} for controls**. Rates vary by mask, article and scheme; see `timings_by_mask.csv`. Four-thread workers were slightly faster than two-thread workers in the calibration.

| AR model | Seconds/window on its GPU | Time for recommended windows, excluding loading |
|---|---:|---:|
{ar_table}

## Coverage choices

All rows retain the twelve masks and both ten-sample schemes. Each uses exactly the same chosen documents and word-start windows for all five models. Percentages use distinct nonempty documents after normalized exact deduplication: 4,236 training articles and 2,579 controls. The full-article row is for its specific deterministic sample, not a universal maximum achievable by selecting only short articles.

| Option | Training + controls | % distinct training | % distinct controls | Shared windows | Expected / with 25% allowance | Cost with allowance |
|---|---:|---:|---:|---:|---:|---:|
{option_table}

I recommend the eight-window option because consecutive word-stride-one windows overlap heavily. Two adjacent windows at four locations preserve local sliding checks while covering about twice as many distinct articles as the sixteen-window option. It is a sampled-window experiment, not an exhaustive article scan. The denser options provide more local curve information with fewer articles.

## Exact document set

Use `documents_recommended.csv` unchanged for LLaDA, Llama 2 7B, Llama 3 8B, Mistral-7B v0.1 and the original OLMo-7B. It contains each original filtered index, group, source month/date, word count, normalized text hash, represented original indices, common valid-window count and selected zero-based window indices. CSV SHA-256: `{hashlib.sha256((DEST/'documents_recommended.csv').read_bytes()).hexdigest()}`.

The source is the frozen BBC selection actually used for fine-tuning: dataset revision `8dd1ecdc92ac43f9c04a3da3e945537dbb08179b`, March–June 2025, publication after February 2025. Training uses the original even filtered indices; controls use original odd indices. The training JSONL SHA-256 is `8f22ff46d5dfc7d85a2c58d598c72252b754a335514e3bd02ca485a18b87e91e`.

The original training selection has 5,322 rows, of which 5,315 are nonempty and 4,236 distinct after whitespace-normalized exact deduplication. There are 2,579 distinct nonempty controls after removing normalized exact training overlaps. The 350 chosen training texts represent **{rec['represented_training_rows']} original training rows** when repeated normalized copies are counted. We score one actual training text per distinct-document class, not repeatedly score duplicate rows.

Under all five tokenizers, 3,776 training documents and 2,371 controls support a complete 100-token window. The recommendation covers **{rec['train_pct_eligible']:.2f}% of eligible training documents** and **{rec['control_pct_eligible']:.2f}% of eligible controls**, or **{rec['train_pct_all_unique']:.2f}% / {rec['control_pct_all_unique']:.2f}%** of the respective distinct populations including short documents. Articles that are too short are ineligible; they are not selected based on a model score.

Selection is deterministic, independent of extraction probabilities, and excludes the 16 pilot articles. Training/control counts are matched within source month and pooled length quartile, using seed label `bbc-main-20261101` and SHA-256 ordering. Quartile boundaries are 404, 606 and 876 words. Each table entry is the number selected **in each group**:

| Month | <404 words | 404–605 | 606–875 | ≥876 | Total in each group |
|---|---:|---:|---:|---:|---:|
{month_lines}

This targets the selected month/length-matched population, not an unweighted random sample of all raw dataset rows. Near duplicates and partially shared passages can remain after exact deduplication: the negative label means the article was excluded from fine-tuning, not that every passage is unseen.

Report training/control extraction rates separately by model, mask and scheme. Use article-level summaries and resample articles for uncertainty estimates; overlapping windows and the twelve masks are not independent document samples. Keep mask 12 as the common conditioning task for comparisons with AR.

## Exact windows and parameters

For each article, let W be the minimum valid-window count across the five native tokenizers. A valid window follows the supplied runner: start at character zero, then advance one word at a time, tokenize up to 800 characters without added special tokens, take the first 100 native tokens, and stop when fewer than 100 remain. Counts preserve the original runner's treatment of leading whitespace.

If W≥8, take two consecutive windows at each block start `round((W-2)*f)` for f in `[0, 1/3, 2/3, 1]`. If W<8, score all W windows. For example, W=1000 gives windows `[0,1,333,334,665,666,998,999]`. The manifest freezes the indices. The same documents and word starts are used by all five models. Native tokenizers imply different character endpoints for their 100-token windows; the suffix probabilities therefore do not always concern identical literal continuations.

The recommendation has **{rec['shared_window_starts']:,} shared window starts**, including {rec['windows_by_group']['train']:,} training and {rec['windows_by_group']['control']:,} control windows. LLaDA produces **{rec['llada_scores']:,} probability rows** (24 per start). Each AR model produces **{rec['ar_scores_per_model']:,} rows**, for **{rec['llada_scores']+4*rec['ar_scores_per_model']:,} normal rows overall**.

LLaDA: completed epoch-3 fine-tuned checkpoint, native BF16 weights and forwards, original FP64 estimator arithmetic, `--mode path_sampling`, `--num-samples 10` for each scheme separately, `--remasking threshold` or `low-confidence`, `--confidence-threshold 0.9`, `--temperature 1`, `--decoding-scheme full`, `--tau 0.001`, 100-token sequence, 50 masked positions, and the existing state cache enabled. No quantization, model-parallel split, changed probability helper, `--scale`, `--verbose` or detailed sample output. Model-forward batch size stays one; independent computations are distributed across GPUs.

Masks use the notebook's exact one-based definitions; each has 50 positions:

| Notebook mask | Masked positions |
|---|---|
{mask_table}

All four AR models: completed fine-tuned checkpoints, BF16, `--mode exact`, full vocabulary, temperature 1, first 50 tokens as prefix and final 50 as target, no extra BOS/EOS or chat template. Exact teacher forcing needs one causal forward per window. The suffix task corresponds to LLaDA mask 12; LLaDA's other eleven masks use different conditioning and are supplementary comparisons, not additional AR tasks.

For production, use master seed 20261101 with deterministic independent per-task seeds derived from group, article ID, window index, mask and scheme. Store each task's actual seed. This changes the random realization from the fixed-seed timing pilot, not the sampling distribution or number of samples, and makes retries independent of GPU scheduling.

## Parallel schedule, saves and cost control

Initially use GPUs 0–3 for one AR model each and GPUs 4–6 for LLaDA. The four AR passes should finish in roughly five minutes plus loading. As each AR process exits, start a LLaDA worker on that freed GPU. All seven GPUs then run LLaDA, one model replica per GPU, four CPU threads per worker. A single shared task queue distributes article/window/mask/scheme blocks and prevents one slow mask from leaving other GPUs idle. Reuse the single disk copy of the checkpoint; GPU replicas do not require seven disk copies.

Run small document waves, initially four training/control pairs per wave. Mark a wave committed only when all five models and all required LLaDA configurations have finished it. Reassess speed after the first wave. Keep the same completed cohort for every model if execution must end early; do not report a partially completed mask/model cross-product as complete.

Write normal probability CSV rows and compact provenance/completion records to pod disk. Upload immutable compressed batches through the private Drive API connection every five minutes and after each completed wave. Retrying uses stable task IDs and stored seeds. At completion, consolidate one normal CSV per LLaDA mask/scheme and one per AR model, plus summaries and manifests. Keep `p_z`, `log_p_z`, extraction decision, model/checkpoint ID, article ID, window ID, mask, scheme and seed; omit per-sample traces. Expect tens of MB of uncompressed normal results, not new model checkpoints.

Articles and 74 GB of cached weights remain on the pod; the frozen source JSONLs already occupy about 39 MB there. Windows receives only this small plan, metadata, code and results. Drive receives results, not duplicate model weights.

Before the full job, recompute the balance-based deadline, verify the available RunPod stop credential, and configure pod shutdown on successful final upload or a hard deadline. `runpodctl` and a runtime API credential are present; no pod stop has been executed. Stopping the Python process alone does not stop GPU billing. Keep a final upload grace period within the $50 reserve and stop adding waves before the reserve is consumed.

The full experiment has **not started**. Only inventory and timing pilots have run. Drive planning folder: {url}
'''
(DEST/'PLAN.md').write_text(text,encoding='utf-8')
(WORK/'RUNPOD_SLIDING_WINDOW_PLAN_20261101.md').write_text(text,encoding='utf-8')
subprocess.run([*BASE,'copyto',str(DEST/'PLAN.md'),REMOTE+'/PLAN.md'],check=True)
archive=WORK/'runpod_planning_results/runpod_sliding_window_plan_20261101.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for f in DEST.iterdir():
        if f.is_file():z.write(f,f.name)
    for f in (WORK/'finetuned_sliding_windows').glob('*'):
        if f.is_file() and f.suffix in {'.py','.md'}:z.write(f,'code/'+f.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
downloads=Path.home()/'Downloads'/archive.name;downloads.write_bytes(archive.read_bytes())
subprocess.run([*BASE,'copyto',str(archive),REMOTE+'/'+archive.name],check=True)
print(json.dumps(dict(status='complete',recommended_documents=len(docs),shared_windows=rec['shared_window_starts'],drive=url,zip=str(downloads)),indent=2))
