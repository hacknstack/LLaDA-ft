# LLaDA-8B-Base continued pre-training for a BBC memorization study

**Protocol version:** 1.0  
**Date:** 21 October 2026 (Europe/Zurich)  
**Run identifier:** `bbc_fp32_3ep_v1`  
**Status:** Pre-specified production protocol. Feasibility probes have passed; a production run is recorded separately in its status and event files.  
**Primary machine:** Existing Google Colab runtime, NVIDIA A100-SXM4-80GB; approximately 167 GiB host RAM and 12 CPU threads.

## 1. Purpose and interpretation

We will continue training the released LLaDA-8B-Base checkpoint on the even-indexed half of BBC articles published on or after 1 March 2025. The objective is to obtain checkpoints at controlled exposure levels for a subsequent memorization study. We update every model parameter.

This is a continuation experiment on a small, deliberately selected corpus. It does not reproduce the original model's complete training history or identify how often a document appeared in that history. A successful run supports statements about changes caused by this additional training. Claims about extraction, membership inference, or original pre-training memorization need their own evaluation protocol and controls.

Three epochs is our nominal exposure budget. An epoch visits each packed training block once; the rare random-length operation omits the remaining suffix of that occurrence. Thus three epochs does not mean that every token is necessarily used in the loss exactly three times. Masking is resampled on every occurrence. Short-run losses cannot establish that a model has memorized article text.

## 2. Evidence and classification rules

The comparison uses these primary sources, reviewed on 21 October 2026:

- **P:** Nie et al., *Large Language Diffusion Models*, arXiv:2502.09992v3, especially sections 2.1-2.3 and appendices A.3, B.1, and B.2. [Paper](https://arxiv.org/html/2502.09992v3)
- **G:** Authors' `GUIDELINES.md`, the released reference training objective. [Guidelines](https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md)
- **M:** Released model repository and its pinned configuration, tokenizer, custom Python code, and weight files. [Model](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
- **D:** Frozen dataset repository revision, monthly configurations, and our saved selection manifest. [Dataset](https://huggingface.co/datasets/RealTimeData/bbc_news_alltime)

The compact published-setting register is: P reports 2.3T pre-training tokens, concatenated documents, 4096-token sequences, random lengths for 1% of examples, independent token masks at a sampled rate, an inverse-rate weighted loss, AdamW, weight decay 0.1, global batch 1280, and local batch 4. Its learning-rate trajectory warms up for 2000 updates to 4e-4, later uses 1e-4, and ends at 1e-5. P's three-epoch setting is for instruction SFT. G supplies token-averaged loss code, mask token 126336, and a 0.001 lower bound for mask probability.

**Match** means the relevant published behavior is implemented, or the released model component is preserved. **Adaptation** means a deliberate difference with a stated reason. **Unknown** means the reviewed sources do not establish an exact original value or implementation; an explicit assumption is used instead. Matching a high-level setting does not establish bitwise equivalence. G is reference code, not a release of the complete original training system.

## 3. Frozen inputs and membership

| Item | Frozen value |
|---|---|
| Model repository | `GSAI-ML/LLaDA-8B-Base` |
| Model revision | `0f2787f2d87eac5eed8a087d5ecd24277e6255b2` |
| Trainable parameters | 8,015,581,184 |
| Dataset repository | `RealTimeData/bbc_news_alltime` |
| Dataset revision | `8dd1ecdc92ac43f9c04a3da3e945537dbb08179b` |
| Monthly configurations | `2025-03`, `2025-04`, `2025-05`, `2025-06` |
| Date criterion | Parsed `published_date >= 2025-03-01` |
| Membership rule | Sort monthly configurations; preserve each configuration's original row order; filter by date; assign a global zero-based index; select indices 0, 2, 4, ... |
| Date-eligible articles | 10,643 |
| Selected even-index articles | 5,322 |
| Empty selected contents | 7; retained in the membership file, omitted from tokenization |
| Nonempty selected articles | 5,315 |
| Training tokens, including one EOS per nonempty article | 3,941,989 |
| Packed training blocks | 963: 962 full blocks and one 1,637-token final block |

Membership is frozen before shuffling or tokenization. We never recompute parity after removing empty rows, checking overlap, or shuffling. Only `content` is trained; titles, links, dates, images, and other metadata are kept for traceability and are not inserted into the text.

The cutoff restricts dataset publication dates; it does not prove that the wording is new to the base model. Updated/reprinted articles can contain older material. The untouched base checkpoint remains the epoch-0 reference.

### Odd-index controls and duplicate audit

Odd-index records are collected separately and never enter the training optimizer. We normalize whitespace, trim the resulting string, and compute SHA256 of content. Odd records with the same normalized content as any training record are flagged and excluded from the diagnostic control stream. Empty controls are also excluded. We retain every even-index training member, including duplicates, to preserve the requested selection.

The resulting control counts, duplicate training groups, token counts, and token-file hashes are written to `datasets/training_data_manifest.json`. Exact normalized matching is an audit, not a semantic deduplication method. Near-duplicates, syndicated passages, shared boilerplate, and partial overlaps remain possible. Any later extraction or membership analysis must address these separately. Within-training duplicates can cause more than one effective exposure per epoch and must be reported in that analysis.

**Measured audit:** the 5,315 nonempty training rows contain 4,236 unique normalized contents and 1,079 additional duplicate rows. Of 5,321 odd-index records, seven are empty and 1,969 have exact normalized content overlap with training. The remaining 3,345 records supply the clean diagnostic stream, containing 2,617,971 tokens including EOS. These controls may still duplicate each other or share near-duplicate passages; no stronger independence claim is made.

## 4. Executable training specification

The companion `llada_training_config.json` is machine-readable. The production trainer records its exact configuration, source hashes, method-document hash, package versions, and data hashes before updating weights.

| Setting | Our fixed choice |
|---|---|
| Initialization | Untouched pinned Base checkpoint; no benchmark-trained weights reused |
| Parameters updated | All 8,015,581,184; no adapters, frozen blocks, or weight quantization |
| Forward/backward | BF16 model and per-microbatch gradients; FP32 cross-entropy evaluation |
| Accumulation | GPU FP32 buffers; each microbatch's BF16 gradients are cast and added, then ordinary `.grad` is cleared |
| Optimizer weights | Independent CPU FP32 master copies |
| Optimizer moments | CPU FP32, stock `torch.optim.AdamW`, `foreach=False`, `fused=False` |
| AdamW assumptions | Betas (0.9, 0.999), epsilon 1e-8 |
| Weight decay | 0.1 on all trainable parameters; no excluded norm/bias group |
| Gradient clipping | Global L2 norm 1.0, after averaging the accumulation window, on CPU FP32 gradients |
| GPU microbatch | One sequence |
| Effective batch | 16 sequences; final window of each epoch contains 3 sequences and is averaged over 3 |
| Updates | 61 per epoch, 183 total; accumulation windows do not cross epoch boundaries |
| Training length | Three epochs; no early stopping based on controls |
| Nominal context | 4096 tokens |
| Length variation | Independently for each occurrence, probability 0.01 of cropping to a uniformly sampled integer length in [1, current block length] |
| Final corpus tail | Retain the final short block without padding; no synthetic tail targets |
| Input formatting | Content + one tokenizer EOS per nonempty article; no BOS, chat template, prompt/response split, or causal attention mask |
| Packing/shuffle | Pack in frozen source order; deterministically shuffle block indices each epoch |
| Noise | Fresh per-example rate p = 0.001 + 0.999 u, with u uniform in [0,1); independent Bernoulli token masks |
| Mask token | 126336 |
| Loss | Sum masked-token CE/p, divided by the actual sequence length; average examples across the actual accumulation window |
| Optimizer step | Transfer averaged FP32 gradients to CPU, clip, update FP32 masters/moments, then copy masters to GPU BF16 weights |
| Activation memory | Whole-layer non-reentrant activation checkpointing; attention remains bidirectional; KV cache disabled |
| Randomness | Seed 42; deterministic logged block permutations/cropping stream; fresh torch mask draws; no claim of deterministic CUDA kernels |
| CPU threads | 12 |
| Snapshots | BF16 safetensors model-only checkpoints after epochs 1, 2, and 3 |

All ordinary tokens, including the inserted EOS delimiters, are eligible for masking and loss. A loss is calculated at the same positions as the masked inputs; there is no autoregressive label shift. With no masked tokens, the sampled loss contribution is zero rather than resampling until a mask appears. Resampling would change the noise distribution. The reference lower bound reduces, but does not eliminate, empty-mask events for short crops.

### Learning-rate schedule

The continuation schedule is an explicit experimental adaptation. It uses 183 planned optimizer updates, 10 warmup updates (`ceil(0.05 x 183)`), a stable phase, and 19 final decay updates (`ceil(0.10 x 183)`). For one-based update number s:

- s = 1,...,10: `lr = 1e-5 x s / 10`.
- s = 11,...,164: `lr = 1e-5`.
- s = 165,...,183: `lr = 1e-5 + (1e-6 - 1e-5) x (s - 164) / 19`.

This starts at 1e-6, reaches 1e-5, and ends at 1e-6. We choose a modest continuation rate rather than restart the released model at its from-scratch peak. This choice is not established as optimal or claimed as the authors' continuation recipe. No learning-rate or epoch-count tuning will use the odd controls.

## 5. Similarities preserved

**S01 - Released model and vocabulary.** We load M unchanged: the original architecture, tokenizer, vocabulary, mask identity, positional behavior, normalization modules, and bidirectional attention implementation remain intact. No model resizing, new embeddings, instruction template, or architectural adaptation is introduced. Pinning makes our starting artifact reproducible; it does not recover the original training software.

**S02 - Full-parameter learning.** Every parameter in the loaded model participates in backpropagation and AdamW. There are no LoRA/DoRA modules or frozen layers. This preserves the model's ability to change all of its weights, which matters when studying how additional training changes memorization.

**S03 - Pre-training task on raw text.** Each article contributes its original content tokens. The task has no instruction-response boundary and no permanently protected prompt tokens. This follows the unconditional branch of G rather than its SFT branch.

**S04 - Masking and targets.** We implement G's per-example masking procedure, use the designated mask token, predict the original token at each selected position, and use its inverse-probability and sequence-length normalization. A new corruption is generated for each visit, including repeated epochs.

**S05 - Main sequence geometry and packing.** Fixed-size full blocks, document concatenation, and the rare short-context operation follow the mechanisms in P/G. Block boundaries are allowed to cross article boundaries. The specific EOS separator and short final block are documented below as boundary assumptions/adaptations.

**S06 - Optimizer family and regularization strength.** Our optimizer is standard AdamW with the published weight-decay strength, rather than a different optimization family. Parameter-group details remain an assumption because the original grouping is unavailable.

**S07 - Schedule family.** Our learning rate has warmup, stable, and decay phases. We match the schedule's shape, not its original absolute step counts or complete trajectory.

**S08 - Mathematical implementation under memory constraints.** Gradient accumulation and activation recomputation allow full-parameter updates without changing the intended averaged loss. FP32 master weights preserve small cumulative optimizer updates when the computation copy is BF16. Numerical rounding, reduction order, and the effective batch still differ from the unknown original implementation.

## 6. Divergences, reasons, and expected implications

**D01 - Continuation instead of training from scratch.** The starting point already contains extensive prior knowledge. Recreating its initialization and entire historical corpus is outside this experiment's resources and source availability. Interpretation: observed changes concern additional BBC exposure, not the complete origin of Base-model memorization.

**D02 - Narrow, English BBC corpus and much smaller token budget.** The date/parity selection is the user's experimental intervention. It intentionally cannot reproduce the original domain mixture or scale. Interpretation: stronger specialization, forgetting, and different memorization behavior are possible; original benchmark capabilities should not be assumed unchanged.

**D03 - Three repeated passes.** This defines an observable exposure curve with manageable compute and saved checkpoints. It is not an inferred Base pre-training epoch count. Repetition and duplicates alter effective exposure. We report epoch-0 and per-epoch diagnostics separately rather than equating our endpoint with the historical model's training.

**D04 - Effective batch 16 rather than the published global batch.** At 4096 tokens, a 1280-sequence update contains 5,242,880 tokens, more than our 3,941,989-token corpus. Three passes would supply about 2.26 such full-token batches. Matching that global batch would yield only two complete updates and a remainder if accumulated across epochs. We choose enough updates to make a useful small-corpus continuation experiment. Gradient noise and optimization behavior therefore differ materially.

**D05 - Physical microbatch 1, accumulation, and epoch remainders.** One sequence fits the A100 with our numerical state. FP32 gradient buffers reduce repeated BF16 accumulation rounding. Averaging the final three-sequence window by its actual size preserves an example-average update, but creates smaller batches at epoch boundaries. Microbatch-dependent behavior and finite-precision reductions prevent exact equivalence to a large physical batch.

**D06 - Scaled learning-rate trajectory and reset optimizer.** The checkpoint does not include the historical optimizer or scheduler state. We initialize new moments and use the short schedule in section 4. This avoids a warmup longer than the entire experiment and an unsupported assumption that a from-scratch rate is suitable for continuation. It also makes direct replication of the original trajectory impossible.

**D07 - CPU offloading and precision choices.** CPU FP32 masters/moments and GPU FP32 accumulation are explicit, tested choices. They avoid model/optimizer quantization and preserve small updates. CPU transfers and stock CPU kernels increase time. They do not establish the original optimizer precision or reproduce its device/reduction order.

**D08 - Reference-code epsilon.** The paper's idealized masking description and G's numerical implementation differ at the lower endpoint. We choose the executable reference formula and record that choice. Very-low-mask events are therefore slightly different from an exact unrestricted uniform rate.

**D09 - Document boundaries, order, and tail.** We insert one EOS, omit an added BOS, concatenate in frozen source order, then shuffle blocks. We retain the final 1637-token block without padding so no original text is silently discarded or synthetic targets learned. The exact historical separator, order, shuffle/repacking, and corpus-tail policies are unavailable. Our extra short tail is one block per epoch in addition to stochastic cropping.

**D10 - Per-occurrence random crop at microbatch 1.** A cropped full block uses an integer length from 1 to 4096; the already-short final block is cropped only within its available length. The released example applies a common crop to a physical batch; our virtual batch may contain different lengths. We normalize each example by its actual length and average examples. The resulting length/sample weighting may differ from the distributed implementation.

**D11 - Declared optimizer defaults and grouping.** Betas, epsilon, clipping, and decay on all parameters are not asserted as historical settings. These assumptions fix otherwise ambiguous implementation choices. They can affect learning, particularly norm/bias regularization and small gradients. The run records them so a later sensitivity experiment can change one factor at a time.

**D12 - Different machine and software stack.** One A100, CPU offload, modern PyTorch/Transformers, and recomputation replace the original distributed environment. This changes throughput and floating-point reduction order. No claim of bitwise reproduction is made. The original architecture and objective are retained.

**D13 - Diagnostic evaluation and model-only checkpointing.** The lightweight evaluation below is tailored to this study. Checkpoints support later extraction analysis but omit the large training state to respect Drive capacity. These choices affect observability and recovery, not the intended uninterrupted optimizer update rule.

**D14 - Exact-overlap filtering only for controls.** The requested training membership remains intact; only contaminated controls are excluded from the diagnostic stream. This is a study-specific evaluation decision. It is insufficient to remove near-duplicate or historical pre-training contamination and must not be represented as a complete deduplication solution.

## 7. Unknown original details and our handling

These are bounded claims about the reviewed public material, not proof that the information could never be recovered. A new author disclosure or training-code release could resolve them. We keep known divergences separate from missing information.

| ID | Question mark | Explicit handling in this experiment |
|---|---|---|
| U01 | Exact original document identities, versions, and full filtering outputs | Freeze the BBC revision; do not claim corpus replication. |
| U02 | Original article-level exposure, sampling/repetition rates, and exact domain schedule | Record our block visits and duplicates; do not infer historical exposure from an epoch count. |
| U03 | Presence of older/reprinted BBC wording in historical training | Use the untouched Base reference and flag this limitation; dates alone are insufficient. |
| U04 | Full original deduplication implementation and retained overlap | Perform explicit exact-overlap audit; report that near-duplicate audit remains outstanding. |
| U05 | Exact document separators, BOS/EOS choices, and boundary reset policy | Use one EOS, no added BOS, and ordinary contiguous bidirectional blocks. |
| U06 | Exact sample order, shuffle algorithm, repacking cadence, worker sharding, and final-tail handling | Freeze packing, log seed/permutation hashes, retain the short tail. |
| U07 | Adam beta coefficients | Set (0.9, 0.999) as an implementation assumption. |
| U08 | Adam epsilon and related numerical variants | Set 1e-8; use stock torch AdamW. |
| U09 | Which parameter groups receive weight decay | Apply 0.1 to every trainable parameter. |
| U10 | Gradient-clipping policy and threshold | Use global FP32 norm clipping at 1.0. |
| U11 | Training model/master-weight, gradient, and moment dtypes | Declare our BF16 compute and FP32 accumulation/masters/moments explicitly. Inference dtype in the paper does not resolve training-state precision. |
| U12 | Loss scaling, mixed-precision underflow handling, stochastic rounding, and optimizer fusion | Use no GradScaler for BF16, no stochastic rounding, no fused/foreach CPU optimizer; abort nonfinite updates. |
| U13 | Original loss-reduction details for variable lengths, per-rank averaging, and zero-mask edge cases | Follow G's normalization per sequence; average the actual virtual batch; keep zero-mask contributions zero. |
| U14 | Correlation/stratification of mask rates across ranks or batches | Independent fresh per-example draws from the declared stream. |
| U15 | Training seeds, full RNG states, deterministic-kernel settings | Seed our experiment, record versions and hashes; deterministic CUDA equivalence is not claimed. |
| U16 | Exact distributed parallelism, accumulation, gradient-reduction order, kernels, and activation-recomputation placement | Use one-GPU non-reentrant whole-layer checkpointing and FP32 buffers; document hardware and versions. |
| U17 | Original optimizer/scheduler state corresponding to the released weights | Initialize a new optimizer; label this continuation rather than exact resumption. |
| U18 | Every operational restart, data replay, skipped update, recovery choice, and full intermediate training log | Record our own events/failures; do not reconstruct an unavailable historical trajectory. |
| U19 | Whether the public reference snippet exactly matches every original production-training detail | Treat G as behavioral guidance, not a complete original training-system release. |
| U20 | A published, validated recipe for continuing Base on this particular small BBC corpus | Our batch, schedule, and three-epoch budget are declared experimental choices; no quality guarantee is inferred. |

The checkpoint's architecture and tokenizer configuration are available and preserved; these are not counted as unknown merely because the original training framework is missing. Original pre-training and SFT settings are not interchangeable.

## 8. Diagnostics and later memorization analysis

Before training and after each epoch, measure the same masked-diffusion objective on eight evenly spaced full blocks from the training stream and eight from the clean odd-control stream. Use four fixed mask draws per block and matched seeds across groups. Evaluation uses no gradient, model evaluation mode, and isolated RNG state so it does not alter subsequent training noise.

These are descriptive fixed-noise training-objective diagnostics. They are not an exact log-likelihood, an extraction score, a calibrated membership-inference attack, or a complete memorization test. The small number of mask draws is chosen for economical monitoring and is not asserted to reproduce the paper's likelihood-evaluation procedure. Controls never affect gradients, schedule, stopping, or model selection in this fixed run.

The following later analysis choices remain to be specified before making scientific memorization claims: prefix/suffix lengths, extraction sample selection, decoding/remasking settings and seeds, generated length, exact-match and near-match criteria, duplicate/boilerplate handling, statistical aggregation, and uncertainty estimates. Save all three epoch models so that this analysis can compare exposure levels against the same untouched base reference. Do not report diagnostic loss improvements as verbatim memorization rates.

## 9. Model sizes and Drive retention

Each of the 8,015,581,184 parameters requires two bytes in the saved BF16 model. Tensor payload is exactly **16,031,162,368 bytes = 16.031 GB = 14.930 GiB**. The existing base shards occupy 16,031,197,112 bytes, excluding tokenizer/configuration/custom-code files; this independently verifies the payload calculation. Fine-tuned snapshots retain the same shapes and dtype, so they have approximately the same size. Shard headers, metadata, tokenizer, and custom code add a small overhead.

| Retained object | Approximate decimal size | Approximate binary size |
|---|---:|---:|
| One BF16 model | 16.03 GB | 14.93 GiB |
| Epochs 1, 2, and 3 model snapshots together | 48.09 GB | 44.79 GiB |
| Original base cache plus all three snapshots | 64.12 GB | 59.72 GiB |
| One CPU FP32 master copy | 32.06 GB | 29.86 GiB |
| Two FP32 Adam moment tensors together | 64.12 GB | 59.72 GiB |
| Master copy plus optimizer moments | 96.19 GB | 89.58 GiB |
| BF16 model plus full master/moment state | 112.22 GB | 104.51 GiB |

**Retention policy:** keep the original cache and three separate epoch model-only snapshots. Save the tokenizer, configuration, needed custom model code, run/epoch metadata, and shard hashes. Do not write FP32 master weights or optimizer moments to Drive. Log observed sizes after each save.

Model-only checkpoints are suitable for loading the BF16 model for inference and later analysis. They do not permit exact training continuation: moments, sub-BF16 master-weight information, accumulation buffers, and full RNG/scheduler state are absent. An interruption may require restarting this fixed experiment from Base; loading an intermediate model would be a different continuation run and must receive a new run identifier.

On the inspected mount, about 102 GiB was available before writing these snapshots. Approximately 45 GiB for the three model snapshots fits that reported capacity; retaining resumable FP32 states alongside them does not. Recheck free space before training and each save, and do not delete unrelated Drive content. Reported mount capacity is an operational check, not an independent billing/quota guarantee. Other datasets, logs, and user files consume additional space.

Checkpoint directories are considered valid only after all shards and supporting files have been written, their headers/shapes/dtypes checked, hashes recorded, and a `COMPLETE.json` marker produced. A partially written directory is not a completed checkpoint. All large writes occur on Google Drive from Colab; no weights or dataset are downloaded to the Windows workspace.

## 10. Feasibility evidence and runtime estimate

The selected numerical setup completed one warmup and two timed optimizer updates, each accumulating 16 real 4096-token blocks. All 48 sample losses and update gradient norms were finite. Master parameters changed. A separate small-tensor test compared the gradient-accumulation mechanism to full-batch autograd across two windows.

| Quantity | Measured result |
|---|---:|
| Trainable parameters | 8,015,581,184 |
| Mean steady-state update | 72.29 seconds, including all 16 microbatches and CPU transfers |
| Throughput | About 907 tokens/second |
| Peak allocated GPU memory | 52.7 GiB |
| Peak reserved GPU memory | 71.3 GiB |
| Peak process CPU RAM | 125.0 GiB |
| Steady-state three-epoch projection | About 3 hours 38 minutes |

This projection excludes setup, data preparation, diagnostics, model saving, and interruptions. Smaller final accumulation windows add some optimizer overhead relative to the steady-state token-rate projection. Random short examples slightly change the compute mix. Budget roughly 4-5 hours plus evaluation/saving variability; this is a forecast, not a completed-run measurement or guarantee. The production status/events will provide actual times.

Earlier 1024/4096 BF16-moment and quantized-optimizer benchmarks measured different numerical setups. Their faster estimates do not describe this FP32-offloaded run. The original benchmark state is discarded; production starts from the original frozen model.

## 11. Execution records, validation, and failure handling

The production run records model/dataset revisions, environment versions, source and configuration hashes, method-document hash, block-permutation hashes, selected counts, actual processed tokens, random-crop counts, loss, gradient norm, learning rate, update/epoch number, timings, diagnostic results, and checkpoint manifests. Sources, configuration, and documentation are copied into the run directory before any optimizer update.

Implementation checks cover the scheduler's exact endpoints, averaged gradients for a full and a remainder accumulation window, zero-mask loss behavior, independent FP32 masters, and checkpoint indexing/header validation. Data preparation must reproduce the frozen selected-token count. Any mismatch is an error rather than silently changing membership or data volume.

The trainer aborts on nonfinite losses/gradients, missing or corrupt inputs, mismatched parameter counts, insufficient reported save space, or checkpoint-validation failure. Errors and tracebacks are written into the run status. Existing completed run directories are not overwritten. No automatic restart from a model-only snapshot is allowed under the same run identifier.

## 12. Artifact map and reproducibility

All large artifacts live under `/content/drive/MyDrive/LLaDA-ft/`:

- `hf-cache/`: original pinned model, tokenizer, and dataset caches.
- `datasets/manifest.json`: original pinned date/parity selection metadata.
- `datasets/bbc_even_after_2025_02.jsonl`: frozen selected article membership and text.
- `datasets/train_tokens.npy`: complete selected token stream.
- `datasets/bbc_odd_controls.jsonl`: odd records and exact-overlap flags.
- `datasets/control_tokens_clean.npy`: diagnostic control token stream.
- `datasets/training_data_manifest.json`: counts, duplicate audit, token-file hashes.
- `docs/LLADA_BBC_METHOD.md`: this protocol; the run also captures its own immutable copy.
- `code/`: preparation, training, validation, and benchmark source.
- `runs/bbc_fp32_3ep_v1/`: frozen configuration/sources/methods, status, events, and diagnostics.
- `checkpoints/bbc_fp32_3ep_v1/epoch-001/`, `epoch-002/`, `epoch-003/`: completed model-only snapshots.
- `logs/`: preparation, benchmark, and production process logs.

The Windows workspace contains only small code, configuration, documentation, and scalar-result files. The existing runtime is reused; no new runtime is provisioned. A reader reproducing the experiment must obtain the pinned original model and dataset, create the recorded environment, run preparation/validation, and start the production trainer with the frozen configuration. Exact CUDA bit patterns are not promised even with the same seed.

Any later change to membership, numerical precision, objective, batching, schedule, exposure budget, or checkpoint-resume behavior requires an updated protocol and a new run identifier. New evidence about the authors' procedure should be entered as a resolved unknown, rather than retroactively describing an assumption as a confirmed match.
