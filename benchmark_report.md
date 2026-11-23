# LLaDA BBC training timing benchmark

Selected 5,322 of 10,643 date-eligible articles; 3,941,989 tokens including article EOS; 3,850 packed blocks.

All runs: A100 80 GiB, 1024 tokens, batch 1, accumulation 1, BF16 compute, whole-layer activation checkpointing, 2 warmups + 8 measured complete optimizer steps. Same article content and diffusion objective. No training checkpoint saved.

| Configuration | Status | Trainable params | Tokens/s | Peak allocated GiB | Hours / epoch | Observed step range, hours/epoch |
|---|---|---:|---:|---:|---:|---:|
| last2_adamw | ok | 436,228,096 | 7889 | 18.2 | 0.14 | 0.13–0.14 |
| last8_adamw | ok | 1,744,900,096 | 5205 | 25.0 | 0.21 | 0.21–0.21 |
| lora_qv_r8 | ok | 4,194,304 | 3220 | 16.9 | 0.34 | 0.34–0.34 |
| lora_qv_r32 | ok | 16,777,216 | 3214 | 17.0 | 0.34 | 0.34–0.34 |
| full_sgd | ok | 8,015,581,184 | 2334 | 30.1 | 0.47 | 0.47–0.47 |
| lora_all_r64 | ok | 167,772,160 | 2332 | 18.7 | 0.47 | 0.46–0.48 |
| lora_all_r16 | ok | 41,943,040 | 2323 | 17.3 | 0.47 | 0.46–0.50 |
| full_adamw_bf16 | ok | 8,015,581,184 | 2105 | 60.0 | 0.52 | 0.52–0.52 |
| qlora_nf4_r16 | ok | 41,943,040 | 1689 | 10.8 | 0.65 | 0.64–0.66 |
| full_adamw8bit | ok | 8,015,581,184 | 1415 | 45.3 | 0.77 | 0.77–0.78 |
| lora_int8_r16 | ok | 41,943,040 | 1018 | 16.1 | 1.08 | 1.05–1.11 |
| dora_all_r16 | ok | 43,384,832 | 975 | 17.8 | 1.12 | 1.09–1.30 |

Estimates cover training compute for ONE pass over the selected even-index half, not the full unfiltered source. Three epochs approximately triple training time. Loading, downloads, preprocessing, evaluation, checkpoint writes, and interruptions are excluded. Observed ranges are min/max of 8 steps, not statistical confidence intervals. Allow 15–30% operational headroom before adding checkpoint/evaluation time. Longer sequence lengths or different batches need fresh measurements.

Partial tuning trains only the final 2 or 8 blocks plus final norm; frozen embeddings and output head. LoRA/DoRA train adapters in the specified transformer-block projections. Full SGD uses no momentum. Full AdamW stores moments in BF16; this is a timing configuration, not equivalent to full FP32-master-weight AdamW. Full FP32-master-weight AdamW typically exceeds this GPU without offload/sharding. Quantized bases use BF16 compute and floating-point adapters.

Timing probes measure feasibility and speed; they do not rank model quality or determine learning rates. All configurations start from the same pinned base model in separate processes.

Sources: https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md ; https://huggingface.co/datasets/RealTimeData/bbc_news_alltime