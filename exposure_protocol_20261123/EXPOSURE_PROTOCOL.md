# BBC exposure-controlled full-fine-tuning protocol

Status: PAUSED AFTER DISJOINTNESS AUDIT. No model weights downloaded, no article exposure assignments, no optimizer steps, and no training run started. The user requested this review point before proceeding.

## Verified corpus

Use RealTimeData/bbc_news_alltime revision 8dd1ecdc92ac43f9c04a3da3e945537dbb08179b, months March?June 2025 and publication dates from 2025-03-01. The old even-index training JSONL was reconstructed byte for byte: SHA256 8f22ff46d5dfc7d85a2c58d598c72252b754a335514e3bd02ca485a18b87e91e. It contains 5,322 rows, including seven empty rows and 4,236 unique nonempty normalized contents.

The old-training-disjoint odd pool has 3,345 rows and 2,579 unique normalized contents. Near-duplicate screening against the old training corpus rejects 522 candidates. Additional long-passage screening and internal deduplication leave 1,950 conservative representatives; 1,540 have 200?1,500 canonical words (March 386, April 352, May 400, June 402).

Near-duplicate means 5-word-shingle Jaccard at least 0.5 OR shared-shingle count divided by the smaller document's shingle count at least 0.8. Comparisons use exact sparse intersections over every pair, with no approximate candidate search. Strict screening additionally excludes any consecutive 50 canonical words shared with old training. Normalization is Unicode NFKC, casefold and Unicode word tokens, used only for auditing. Internal near-duplicate/shared-passage connected components contribute one longest representative. Shared boilerplate is conservatively included. Original article contents remain unchanged.

This establishes lexical disjointness under stated rules relative to the prior fine-tuning corpus. It does not prove absence of semantic paraphrases, shorter shared passages, or exposure during original base pretraining. Some fresh candidates were earlier held-out evaluation/control texts; they were not used in optimizer training.

## Proposed experiment, not yet assigned

Randomly assign 32 articles to each total exposure level 1, 2, 4, 8, 16, 32, 64, 128, and 256. Reserve 128 zero-exposure controls. This requires 416 eligible articles and produces 16,352 target-article occurrences. The verified pool is sufficient, with 1,534 strict-disjoint representatives remaining for optional background after reserving those 416.

Balance target/control selection and exposure allocation by publication month, article length and a reproducible topic proxy. Do not inspect extraction outcomes to select articles. Freeze original row IDs, exact content hashes, split membership and seed before training. Use the exact same article identities and ordered occurrence list for every model.

Background articles must be disjoint from targets and controls under the same rules. If used, repeat their original text three times in total. Choose and freeze the actual background set and token budget before training. Its size is not yet fixed.

An exposure is one complete original-text presentation in the input token stream, not one epoch and not one guaranteed loss contribution at each token. Spread occurrences across the run with balanced placement and record repeats within an optimizer update. Low-exposure presentations cannot have identical timing coverage to high-exposure articles; record first/last positions and recency. A single full-training endpoint contains all dose groups; results reflect assigned exposure within a jointly trained model.

## Settings preserved

Models: original pinned LLaDA-8B-Base, Llama-2-7b-hf, Meta-Llama-3-8B, Mistral-7B-v0.1, and OLMo-7B-hf. All model revisions and inherited settings are in draft_protocol.json. Start each from its original base release; all parameters trainable.

Preserve native tokenizers/architectures, BF16 model computation, FP32 gradient accumulation, FP32 CPU master weights and AdamW moments, microbatch 1, accumulation 16, 4096-token concatenated blocks, original EOS policy, unpadded final block, non-reentrant activation checkpointing, and 12 CPU threads per worker. Retain the existing model-specific loss normalization and actual-size remainder accumulation.

AdamW: peak LR 1e-5, final LR 1e-6, 5% warmup, 10% terminal decay, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.1 on the inherited all-parameter grouping, gradient clip norm 1.0. Compute actual update counts and schedule rounding after each native-token manifest is built; do not copy the old three-epoch update counts.

LLaDA retains its unconditional masked reconstruction objective, mask ID 126336, fresh masks per occurrence, mask-rate floor 0.001, and inverse-rate weighting. AR models retain native causal next-token CE and their earlier per-sequence/accumulation normalization. Original OLMo's unscaled 4096 computation still extrapolates beyond its configured 2048 pretraining context; this inherited limitation must remain explicit.

## Necessary protocol differences

1. Explicit total article counts replace three repeated epochs. Three passes over a repeated corpus would multiply the requested counts by three.
2. Disable the previous 1% random-length cropping to retain every scheduled article token. Do not truncate long articles; track spans across packed blocks.
3. Deduplicate and screen the new corpus, unlike the old corpus which retained training duplicates.
4. A balanced occurrence schedule replaces the old source ordering and epoch-level block shuffling. Never reshuffle packed blocks in a way that invalidates the frozen occurrence schedule or its exposure accounting.
5. Recompute native-token totals, block boundaries, updates and learning-rate step counts for each model. Equal article presentations do not make token counts or loss opportunities equal across different tokenizers/objectives.

Each schedule compiler must assert exact requested article counts and total tokens before training. The trainer must audit consumed token spans and counts independently. Log LLaDA masked-target opportunities and AR first-token-of-block omissions. Targets, controls and background have separate membership manifests. Keep controls out of optimization, model selection and early stopping.

## Execution and storage, pending feasibility checks

The current pod has three H100 80 GB GPUs, a 753,999,998,976-byte host-memory limit and 66.3 CPU cores of quota. Proposed execution uses independent single-GPU model workers in waves, retaining the effective batch rather than silently changing it for distributed training. New-pod full-parameter forward/backward and optimizer feasibility must be checked after corpus review.

Keep datasets, token arrays, base weights and staging on the pod. Save one final BF16 model per model, no intermediate model/optimizer snapshots. Approximately 74 GB of final weights across all five models is a prior size estimate; current Drive quota and renewable upload authorization remain unchecked. Validate serialized weights and server checksums before writing completion markers. Automatic shutdown behavior is not configured for this new preparation-only pod.

## Evaluation

Compare each untouched base with its final model using frozen article-relative windows and common extraction settings. Match evaluation across exposure groups, retain zero-exposure controls, and aggregate at article level. Record uncertainty from the modest 32-article group size. Do not treat correlated windows as independent article samples. Any additional assignment seeds are separate proposed replicates, not authorized runs.

## Files and review point

Pod workspace: /workspace/bbc-exposure

- protocol/draft_protocol.json: inherited settings and pending choices.
- protocol/EXPOSURE_PROTOCOL.md: this protocol.
- audits/disjointness_summary.json: counts, definitions and limitations.
- audits/candidate_overlap.csv: per-candidate overlap evidence, without article text.
- data/strict_disjoint_pool.jsonl: 1,950 original-content representatives.
- data/eligible_target_control_pool.jsonl: 1,540 proposed target/control candidates.
- code/audit_disjoint.py: reproducible audit.

Preparation stops here for the requested disjointness report. No training, gated-model downloads, target assignment or background selection has been performed. The pod itself remains running.
