# Autoregressive comparison to the completed LLaDA BBC experiment

Updated: 27 October 2026 (Europe/Zurich). Scope: the matched continuation protocol, resource probes and saved-run verification. Mistral and OLMo production runs completed; Llama production runs have not been started.

## Reference experiment

The primary reference is successful LLaDA run `bbc_fp32_colab_20261026_084128_323830`, completed after three epochs and 183 updates. Its protocol, source hashes, configuration and environment are preserved in Drive. The October 22 notebook run also completed, but is a separate replicate; do not combine the two as a single exposure history.

The intended comparison changes the predictive objective from masked diffusion to autoregressive next-token prediction while preserving all feasible continuation settings. It cannot literally change only the objective: the requested models have different released weights, tokenizers, parameter counts, attention architectures and original training histories. These are explicit comparison factors, not implementation details to conceal.

## Exact model identities

Use the original base releases, not chat/instruction variants or later model generations. Resolve and record a full immutable revision before any tokenizer/weight download.

| Requested model | Repository | Selection rationale |
|---|---|---|
| Llama 2 7B | `meta-llama/Llama-2-7b-hf` | Official original Llama 2 base, Hugging Face format |
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B` | Official original Llama 3 base; not Llama 3.1/3.2 or Instruct |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | First original Mistral 7B base; v0.3 would be another experimental model |
| OLMo-7B | `allenai/OLMo-7B-hf` | Authors' Transformers-format conversion of the original OLMo-7B; not OLMo-1.7, OLMo-2 or Instruct |

The OLMo authors identify `OLMo-7B-hf` as the conversion of the original release and recommend it with modern Transformers. Its card's statement that it does not directly support their fine-tuning processes refers to the original OLMo tooling; actual Transformers backward/optimizer feasibility is tested here rather than assumed from that sentence. Sources: [original card](https://huggingface.co/allenai/OLMo-7B/blob/main/README.md), [converted card](https://huggingface.co/allenai/OLMo-7B-hf/blob/main/README.md).

Llama downloads require a Hugging Face login with approved access. Initial anonymous requests returned HTTP 401 for both repositories. The user authenticates directly in Colab; tokens are never placed in scripts, reports or this workspace. Authentication is not a numerical training difference.

## Corpus identity and tokenization

Use the frozen `RealTimeData/bbc_news_alltime` revision `8dd1ecdc92ac43f9c04a3da3e945537dbb08179b`, configurations `2025-03`, `2025-04`, `2025-05`, `2025-06`, sorted in that order. Date-filter publication dates to at least 2025-03-01, assign one global zero-based index in original source order, and select even indices. This is the already completed LLaDA corpus, not an expanded snapshot including any new monthly data.

Use exactly the saved `datasets/bbc_even_after_2025_02.jsonl`: 5,322 rows, indices `0,2,...,10642`. Keep all training duplicates. Omit the same seven empty contents from tokenization, leaving 5,315 nonempty articles. Use only the original article content, with no title, instruction, chat template, BOS, extra textual delimiter, deduplication or normalization of the training text.

The frozen JSONL currently has SHA-256 `8f22ff46d5dfc7d85a2c58d598c72252b754a335514e3bd02ca485a18b87e91e`. Each probe checks the complete index list, dates and empty-row count and records this file hash. The additional reference-tokenization audit checks that this text reproduces the canonical LLaDA token file. All AR models must use this same file hash.

Tokenize each nonempty article with that model's native tokenizer and `add_special_tokens=False`, append exactly one native EOS, concatenate in the frozen article order, then pack into 4096-token blocks. Keep the final short block unpadded. Save a distinct token-file hash and tokenizer/model revision for every model. Reusing LLaDA token IDs in another vocabulary would encode different text and is prohibited.

Consequences of native tokenization:

- Article identity, ordering and nominal three-pass corpus exposure match; token counts, block boundaries, EOS IDs, tail lengths and examples seen together in an update differ.
- With microbatch 1 and accumulation 16, the number of optimizer updates is computed from each actual block count. Three epochs and the original 183 updates cannot both be held constant when tokenization changes. Preserve three epochs and accumulation 16; report the resulting update counts. Do not truncate or repeat data to force 183.
- Equal 4096-token context and effective batch do not imply equal bytes/characters of context per model. Weight-decay applications and the number of optimizer steps can differ as a consequence. A separate fixed-update experiment would be a different protocol.
- Omitting BOS matches the LLaDA continuation procedure, even for a native tokenizer whose default would insert BOS. It is not a claim about the AR model's original pretraining recipe.

## Settings to preserve in full continuation

| Setting | Matched value or policy |
|---|---|
| Parameters | All released model parameters trainable; no LoRA, quantization or frozen layers |
| Model computation | BF16 GPU parameters/computation |
| Accumulation | FP32 GPU buffers, with each BF16 parameter gradient added then released |
| CPU optimizer | Independent FP32 master weights, stock PyTorch AdamW and FP32 moments |
| Optimizer implementation | Exact `CpuMasterAdamW` class extracted from the reviewed LLaDA trainer; no altered update rule |
| AdamW | betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.1` for every trainable parameter including norms/biases |
| Clipping | One CPU FP32 global gradient norm, limit 1.0, after the accumulation window |
| Sequence length | 4096, with unpadded final short block; OLMo context issue below |
| Batch | Microbatch 1; accumulate 16; final window each epoch averaged over its actual size |
| Epochs | Three; no control-driven stopping or tuning |
| Length variation | Independent 1% crop per block occurrence to a uniformly sampled integer length 1 through current block length |
| Packing and shuffle | Frozen source-order packing; shuffled blocks per epoch with NumPy generator seed `42+epoch` |
| Crop RNG | Separate NumPy generator seed `42+10000+epoch` |
| Model RNG | PyTorch seed 42; no claim of cross-model identical RNG trajectories |
| LR peak/final | `1e-5` / `1e-6` |
| LR schedule | First `ceil(0.05*total_updates)` warmup, steady peak, final `ceil(0.10*total_updates)` linear decay to final rate; count actual model-specific updates |
| Memory strategy | Whole decoder-layer activation checkpointing, non-reentrant; KV cache disabled |
| Host execution | 12 CPU threads; same Colab A100 80 GB and high host RAM |
| TF32 | Enabled as in LLaDA reference |
| Durable output | BF16 model-only epoch snapshots, tokenizer/configuration, method/config/code/data hashes and upload receipts |
| Storage mechanism | Serialize/validate on Colab local disk; resumable Drive API uploads; server size/MD5/SHA checks; completion marker last |
| Resume | No exact optimizer-state resume; new run identifiers required for separately specified restarts |

The learning rate and AdamW settings are matched to the completed LLaDA continuation experiment, not to each AR model's original pretraining optimizer. Replacing them with the original AR model's defaults would add a confound. For example, the original OLMo card reports a different beta2 and epsilon; those values are not used here. [OLMo pretraining hyperparameters](https://huggingface.co/allenai/OLMo-7B/blob/main/README.md).

## Required objective and attention differences

LLaDA used bidirectional attention, independently sampled corruption per occurrence and inverse-mask-probability weighted, unshifted reconstruction loss. AR uses the released causal attention and uncorrupted input with teacher-forced shifted targets:

`loss(block) = mean(CE(logits[0:T-1].float(), token_ids[1:T]))`.

Each micro-example's mean loss is averaged over the actual accumulation-window size, preserving LLaDA's sequence-mean then window-mean convention. Do not flatten together variable-length examples and change weighting to a token-weighted batch mean. EOS participates as an ordinary next-token target; predictions may cross article EOS boundaries within the concatenated stream, matching the lack of document-isolation masking in the LLaDA continuation.

The first token of each independent block has no preceding input and is not an AR target. Thus an uncropped block has `T-1` targets, unlike LLaDA's same-position reconstruction opportunities across `T` positions. A length-1 crop has no AR target; use a differentiable zero contribution and preserve its occurrence and denominator. Do not silently redraw it or change the crop minimum to 2. Adding BOS or a preceding-context token to recover every first-token target would change the specified packing/context policy.

No diffusion mask draws, mask ID, inverse-probability weights or random noising occur for AR. Preserve each native released architecture, RoPE settings, normalization, grouped-query/full-head pattern and vocabulary. Do not convert an AR model to bidirectional attention to imitate LLaDA; that would change the model and predictive task.

Hugging Face native AR modules use causal PyTorch SDPA during the probes; the original LLaDA custom module has its own attention implementation. At lengths at most 4096, Mistral's native 4096 sliding window does not exclude any additional in-block causal history. Floating-point kernel arithmetic can differ, so bitwise equivalence is not claimed. [Mistral configuration](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json).

## OLMo context and source-precision limitations

The original OLMo-7B release has `max_position_embeddings=2048`, RoPE theta 10000 and no configured scaling. Matching LLaDA's 4096 therefore extends OLMo beyond its original pretraining context. The primary probe attempts actual 4096-token forward/backward and CPU-master updates with the native configuration unchanged. A computational success would demonstrate resource/implementation feasibility, not validate the quality of unscaled RoPE extrapolation.

OLMo at 2048 would respect its original context but would change block size, effective tokens per update, update counts and retained context relative to LLaDA. RoPE scaling or replacing OLMo with a later 4096-context generation would introduce still other changes. Report these alternatives explicitly rather than silently selecting one. This document does not authorize a full OLMo run under an undisclosed context change. [OLMo native configuration](https://huggingface.co/allenai/OLMo-7B-hf/blob/main/config.json).

The converted OLMo configuration advertises FP32 source weights. Actual shard header dtypes are audited separately. Under the common BF16-model/FP32-master procedure, native FP32 weights are rounded on BF16 model loading and FP32 masters start from that rounded model, just as the reference initializes masters from the loaded BF16 model. Preserving FP32 source masters instead would be another initialization protocol. Record this distinction if source tensors are FP32.

The same distinction applies to a source stored in FP16: loading it as BF16 rounds its mantissa before FP32 masters are initialized. The shard-dtype audit below records actual source tensors for each model; preserving a native FP16 or FP32 master initialization would change the common initialization procedure. No original-precision master weights are silently substituted for the tested BF16-load convention.

## Controls and interpretation

Preserve the same odd-index diagnostic membership as LLaDA: exclude the seven empty rows and the 1,969 exact normalized train overlaps only from diagnostic controls, yielding 3,345 clean nonempty controls. Keep all within-training duplicates. Tokenize controls with each model's native tokenizer. Near-duplicate independence is not established.

The intended production monitoring is deterministic AR CE on eight evenly spaced full training blocks and eight clean-control blocks, baseline and after each epoch, with isolated RNG state. Four repeated identical mask draws would be redundant without a diffusion corruption process, so AR diagnostics need only one deterministic pass per block. Diagnostic block identities change with tokenization; equal diagnostic article windows would require a separate documented evaluation design. Controls must not determine stopping, hyperparameters or model choice.

Native-token CE and LLaDA's fixed-noise reconstruction objective are not numerically comparable across these models. This continuation comparison needs a later shared string-level memorization/extraction or membership evaluation, with the same article identities, exposure accounting, prompt/continuation definitions and duplicate controls. None of the feasibility results demonstrates memorization by itself.

The publication-date cutoff does not prove the wording was absent from any base model's pretraining data: updated/reprinted news and shared passages may predate the listed article dates. Original pretraining corpora and exposure histories differ across the models. Include an untouched native Base baseline for each model in any memorization evaluation; no objective-only causal claim about whole model families is justified by this design.

## Feasibility probe and evidence boundary

Each available model is loaded from its pinned original weights and independently tested at 4096 with every parameter trainable, BF16 GPU weights, non-reentrant decoder checkpointing and the exact FP32 CPU-master optimizer. The probe performs two full 16-example updates and one actual-size final-window update including the real unpadded tail. Every step requires finite loss and gradient norm, gradients for every parameter and a changed FP32 master. FP32 optimizer-state tensors and a differentiable zero-loss one-token crop are checked. Report actual peaks, not an inference-only load or a LoRA proxy.

Probe-only deviations: constant learning rate `1e-5` instead of a complete production schedule; no random cropping for the full-length worst-case memory/timing measurement; only a few optimizer updates; no production epoch, diagnosis sweep or trained model checkpoint upload. A separate single-token path tests the crop edge case. These probes do not demonstrate full-run stability, inference quality or durability of a newly saved AR checkpoint. LLaDA's full-size Drive API transport tests provide prior evidence for that uploader, but new model serialization must still pass production validation.

All model downloads and token arrays are stored on the Colab local filesystem. Only small code, reports and metadata may be saved to Drive during feasibility work; no large file is copied to the Windows workspace and no probe weights are saved to Drive. A production storage plan must be based on current Drive quota and actual AR parameter counts. Retaining three snapshots for four additional models is a separate, substantially larger budget than one LLaDA run. Existing checkpoints must not be deleted to make room without the user's instruction.

Measured results and remaining questions are appended below. Machine-readable reports live in `ar_probe_results/` and are copied to Drive with the method document. When gated model access is unavailable, report the access failure rather than presenting a hardware estimate as a passed training probe.

## Measured feasibility results

All sizes below are actual measurements in the existing Colab A100-SXM4-80GB runtime, which exposes 79.25 GiB VRAM and approximately 167 GiB host RAM. “Passed” means the complete three-update probe and the one-token test passed, not merely that a model loaded.

| Model | 4096 full-parameter probe | Native train tokens | Updates in 3 epochs | Peak GPU allocated / reserved (GiB) | Peak process host RAM (GiB) |
|---|---|---:|---:|---:|---:|
| llama2_7b | Passed | 4,467,275 | 207 | 40.51 / 44.46 | 103.89 |
| llama3_8b | Passed | 3,832,940 | 177 | 52.87 / 59.41 | 125.89 |
| mistral7b_v01 | Passed | 4,302,687 | 198 | 43.35 / 46.75 | 111.34 |
| olmo7b | Passed | 3,930,266 | 180 | 42.29 / 46.90 | 105.96 |

### Pinned revisions

- `meta-llama/Llama-2-7b-hf`: `01c7f73d771dfac7d292323805ebc428287df4f9`.
- `meta-llama/Meta-Llama-3-8B`: `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`.
- `mistralai/Mistral-7B-v0.1`: `27d67f1b5f57dc0953326b2601d68371d40ea8da`.
- `allenai/OLMo-7B-hf`: `11fb3186a2e4f681edea621fa8b4345147a9db6a`.

### Timing and output size

| Model | Warm full update (seconds) | Approx. 3-epoch compute (hours) | One BF16 model (GB) |
|---|---:|---:|---:|
| llama2_7b | 59.8 | 3.44 | 13.48 |
| llama3_8b | 74.3 | 3.65 | 16.06 |
| mistral7b_v01 | 69.1 | 3.80 | 14.48 |
| olmo7b | 58.3 | 2.91 | 13.78 |

Compute projections use one warmed full-window measurement per model, scaled to the actual model-specific update count. They are rough estimates, not confidence intervals or total wall times. Initialization, native-token data preparation, diagnostics, local serialization, hashing, network upload and API retry time are extra. Final short accumulation windows can be faster. No trained AR weights were retained from these probes.

The probed models together require about 57.80 GB for one final BF16 snapshot each, or 173.39 GB for three epoch snapshots each, before rolling snapshots and headroom. Existing LLaDA checkpoints are additional.

At the auxiliary audit, Drive API reported 1722.28 GB (1604.00 GiB) free. This is a current account quota observation, not the mounted filesystem's reported capacity.

### Independent correctness/corpus audit

- `objective_matches_native_hf`: `True`.
- `causal_future_invariance`: `True`.
- `one_token_zero_loss`: `True`.
- `full_and_remainder_optimizer_equivalence`: `True`.
- `frozen_articles_reproduce_canonical_llada_token_sha256`: `f5e29db9e7b1459bf030cac1567f77f6595da59acdfaf010ab977b21fa0254df`.
- `frozen_controls_reproduce_canonical_llada_token_sha256`: `3a6968777b4ae68e4ed1f89b3309ab4e1133a43a2a659bf7fe561dc70416e27d`.

Source shard tensor dtypes, inspected without loading weight data:

- `mistral7b_v01`: `{'BF16': 7241732096}`.
- `olmo7b`: `{'F32': 6888095744}`.
- `llama2_7b`: `{'F16': 6738415616, 'F32': 2048}`.
- `llama3_8b`: `{'BF16': 8030261248}`.

Native-token control stream sizes:

- `mistral7b_v01`: 2,856,246 tokens across the same 3,345 clean odd-control articles.
- `olmo7b`: 2,609,034 tokens across the same 3,345 clean odd-control articles.
- `llama2_7b`: 2,965,409 tokens across the same 3,345 clean odd-control articles.
- `llama3_8b`: 2,545,774 tokens across the same 3,345 clean odd-control articles.

The independent objective test matches the native Hugging Face shifted CE on a tiny Llama architecture; changing future inputs leaves earlier logits unchanged. The optimizer test compares full-batch CE against accumulated single-example CE across 16- and 3-example windows. These CPU checks complement actual full-model GPU probes; they do not replace those probes.

### Remaining decisions / evidence limits

- Original OLMo at unscaled 4096 extrapolates beyond its original 2048 context. Computational feasibility does not establish the scientific effect of that extension.
- The final-only notebooks retain one final model per run, plus text records, without deleting existing user snapshots. The earlier three-epoch-snapshot storage proposal is historical.
- The final-only production trainer preserves the specified data/control/RNG/optimizer rules and validates native serialization. Mistral and OLMo production results are audited separately; Llama runs here remain feasibility probes until their notebooks are run.
- Few-update probes alone do not establish full-run stability, checkpoint loading/inference or shared memorization results. Saved-model loading/inference checks, where performed, are recorded in the separate saved-run audit report.
