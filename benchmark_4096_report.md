# Full fine-tuning at 4096 tokens

Hardware: NVIDIA A100-SXM4-80GB. Full 8,015,581,184-parameter LLaDA-8B-Base, BF16 weights/gradients and compute, batch 1, whole-layer non-reentrant activation checkpointing. Same masked diffusion objective as the earlier benchmark.

Each configuration completed 2 warmups, 8 timed optimizer updates, and an additional fully masked optimizer update to stress memory. All losses and gradient norms were finite.

| Optimizer | Seconds/step | Tokens/s | Peak allocated GiB | Peak reserved GiB | Minutes/epoch | Minutes/3 epochs |
|---|---:|---:|---:|---:|---:|---:|
| AdamW, BF16 moments | 1.571 | 2607 | 60.8 | 76.0 | 25.2 | 75.7 |
| AdamW, 8-bit states | 1.812 | 2261 | 46.1 | 61.3 | 29.1 | 87.2 |

Projection: 3,941,989 content-plus-EOS tokens across the selected even-index half of date-filtered BBC articles, packed into 963 blocks of 4096 tokens (final block padded; padding excluded from loss). One epoch is one pass over that selected half.

These are short-run compute projections. Downloads, setup, data preprocessing, evaluation, checkpoint writes and interruptions are excluded; allow 15-30% operational headroom plus saving/evaluation time. No trained checkpoint was saved. Peak reserved includes PyTorch caching allocator memory and is higher than live tensor allocation.

4096 is the original pre-training context length. These runs retain resource compromises in optimizer-state precision and batch size; matching context length does not reproduce the full original training setup. Larger batches require fresh memory tests.
