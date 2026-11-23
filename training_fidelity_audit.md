# LLaDA Base: fidelity audit for BBC memorization experiment

Sources reviewed: [paper, section 2.2](https://arxiv.org/html/2502.09992v3#S2.SS2), [appendix B.1/B.2](https://arxiv.org/html/2502.09992v3#A2.SS1), and [authors' training guidelines](https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md).

The intended experiment is continued pre-training of LLaDA-8B-Base on raw BBC text. Three epochs is our exposure budget, not a published Base pre-training epoch count. The paper's three-epoch setting belongs to supervised instruction fine-tuning.

| Setting | Published information | Current timing probes / feasible refinement |
|---|---|---|
| Trainable parameters | Full model | Full-model probes match this; use no adapters or frozen layers in the main experiment. |
| Model/tokenizer | Original LLaDA architecture and vocabulary | Original Base checkpoint and tokenizer pinned by revision. |
| Context | 4096 tokens | Already benchmarked successfully. |
| Corpus construction | Concatenate documents and split into fixed-length sequences | Pack full article content plus EOS into blocks. The paper does not specify every document-boundary detail. |
| Masking | Per-example random mask rate, independent Bernoulli masks; inverse-rate weighted masked-token loss | Matches authors' reference code, including its numerical epsilon of 0.001. Fresh noise for each exposure. |
| Length variation | 1% of examples cropped to a random length in [1,4096] | Add to real training; omitted from fixed-shape timing probes. |
| Optimizer | AdamW | Use stock full-parameter AdamW; CPU offload can preserve FP32 master weights and moments. This is a numerical improvement, not a verified original-state-precision match. |
| Weight decay | 0.1 | Timing probes used optimizer defaults (0.01 for torch AdamW). Set 0.1 explicitly in real training. |
| Learning-rate schedule | Warmup-Stable-Decay; from-scratch peak 4e-4; final rate 1e-5 | Timing probes use constant 1e-5. For continuation, propose a short warmup to 1e-5, stable phase, final decay to 1e-6. This is an explicit adaptation. |
| Global batch | 1280 sequences, local batch 4 per GPU | Gradient accumulation can reproduce large effective batches. Propose effective batch 16 as a small-corpus adaptation, rather than batch 1. |
| Optimizer precision, betas, epsilon, clipping | Exact training details not specified in the reviewed paper | Declare assumptions: FP32 masters/moments if feasible, betas (0.9,0.999), epsilon 1e-8, clip norm 1.0. Do not label these as the authors' exact settings. |

## Why copying the batch and learning rate literally is unsuitable here

1280 x 4096 = 5,242,880 tokens per original optimizer update. Our selected half has 3,941,989 tokens, or 963 padded 4096-token blocks. Three passes total only about 2.26 original-sized token batches. Matching that batch would produce only two full updates and a remainder update if we accumulate across epochs. Flushing at each epoch boundary instead changes the effective batch. Neither choice recreates the original large-corpus optimization trajectory.

The original 2000-update warmup alone exceeds the entire number of updates in our small experiment. Its 4e-4 peak belongs to training from scratch, not automatically to continued training of the released checkpoint on BBC news. A smaller effective batch and a scaled schedule must be documented as experimental adaptations.

## Proposed main experiment

- Full model, original checkpoint/tokenizer, 4096-token packed raw text.
- Authors' masked-diffusion objective, including 1% length variation.
- AdamW with weight decay 0.1, preferably FP32 masters and moments with CPU offload if the probe succeeds.
- GPU microbatch 1, effective batch 16 through accumulation; this remains a substantial difference from original pre-training.
- Three epochs, evaluation at epoch 0 and after each epoch. Training membership fixed by the saved date-filtered even-index manifest.
- Proposed continuation learning rate: peak 1e-5, warmup over roughly 5% of optimizer updates, stable phase, decay over final 10% to 1e-6. These fractions are experiment choices.
- Audit duplicate content across even/odd groups before interpreting membership or memorization results. Preserve user-selected membership and flag contaminated evaluation examples.
- Record actual tokens, document exposures, updates, noise seeds, and all numerical/schedule deviations. Keep evaluation decoding settings fixed across checkpoints.

CPU offload does not change AdamW's update formula; BF16 forward/backward, gradient accumulation, reductions, and data/schedule adaptations still prevent bitwise or full-protocol replication. Matching more settings cannot remove the fundamental difference between continued training on a small news corpus and original 2.3T-token pre-training.

## Measured higher-precision feasibility

Successfully tested the full 8,015,581,184-parameter model with BF16 forward/backward, GPU FP32 gradient accumulation, CPU FP32 master weights and AdamW moments, weight decay 0.1, 4096-token sequences, and effective batch 16 (microbatch 1). No parameter quantization or adapters were used.

Three optimizer updates consumed 48 real packed corpus blocks: one warmup and two timed updates. All losses/gradient norms were finite, and FP32 master parameters changed. The accumulation hook was separately checked against full-batch autograd across two windows.

| Measurement | Result |
|---|---:|
| Mean steady-state optimizer update, including 16 microbatches and CPU transfers | 72.29 seconds |
| Throughput | 907 tokens/second |
| Peak GPU tensor allocation | 52.7 GiB |
| Peak GPU allocator reservation | 71.3 GiB |
| Peak process CPU RAM | 125.0 GiB |
| Projected one epoch over the selected half | 1.21 hours |
| Projected three epochs | 3.63 hours (about 3 hours 38 minutes) |

Allow 15-30% operational headroom, plus evaluation and checkpoint writes. This is a short-run throughput projection with constant learning rate and fixed sequence shape; the proposed training schedule and 1% length variation have not been used in a full run. Flushing remainder accumulation windows at epoch boundaries adds a small amount of optimizer overhead beyond the steady-state projection.

FP32 optimizer states and master weights occupy about 89.6 GiB together. Retaining them in a resumable checkpoint has a substantial storage cost; a BF16 model-only snapshot is about 14.9 GiB. No optimizer state or trained checkpoint was written by these probes. Raw measurements are saved in `benchmarks/full_fp32_master_cpu_adamw_4096_acc16_fp32grad.json` on Drive.
