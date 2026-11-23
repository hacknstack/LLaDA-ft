# Fine-tuned sliding-window verification

This directory adapts the supplied `sliding window experiments` runner to load the five completed BBC fine-tuned checkpoints from local pod storage. It does not start the full extraction sweep.

## What changed

- A model path is required. Public base-model fallback is disabled, and an epoch-3 `COMPLETE.json` is required.
- The controller checks every checkpoint file against its saved size and SHA-256, plus the expected training run, update count, base revision and parameter count.
- Models and tokenizers are cached within each worker. Weights load directly onto that worker's GPU in BF16, without quantization.
- CSVs retain the original result columns and add `log_p_z`. Summaries record checkpoint and article provenance. Log probabilities remain useful when the ordinary probability underflows.
- Output directories use microseconds to avoid collisions. Normal CSV and summary files are produced; detailed sample traces are disabled.
- The original `probabilistic_extraction.py` and `get_log_likelihood.py` are copied without edits. `build_finetuned_sliding_runner.py` reproduces the wrapper changes and checks this identity.

## Small verification scope

The frozen training JSONL is checked against SHA-256 `8f22ff46d5dfc7d85a2c58d598c72252b754a335514e3bd02ca485a18b87e91e`. It contains the even filtered indices from the original March–June 2025 BBC selection. The negative example is an odd-index article with no normalized exact-text overlap with training. These controls can still share topics, passages or near-duplicate text; that exclusion is not a semantic deduplication guarantee.

Each model sees training article 1734 and clean odd-index control 1695. Each article supplies windows 0, 1 and 2, with a one-word stride, 800-character tokenization chunks and a 100-token window. Windows use each model's native tokenizer, so word starts correspond across models but the exact token spans can differ. No chat template, BOS or EOS is added.

LLaDA uses `path_sampling`, ten samples, temperature 1, full-vocabulary decoding, and both `low-confidence` and `threshold` (the current name for fast-dLLM). The confidence threshold is 0.9. Two masks are checked: the final 50 positions and the 50 even positions (one-based). The original estimator's FP64 calculations are preserved; model forwards use BF16. There are 24 LLaDA rows.

Llama 2 7B, Llama 3 8B, Mistral-7B v0.1 and the original OLMo-7B each use exact AR scoring of the final 50 tokens given the first 50, temperature 1 and full vocabulary. Each produces six rows. The original AR routine's native-logit softmax arithmetic is preserved. Its `exact` mode evaluates the AR product directly rather than using sampling; it is still subject to floating-point rounding. A separate teacher-forcing calculation checks suffix alignment for each AR model.

The extraction decision threshold is 0.001. The complete check expects 48 rows across 16 CSV files. This small selection verifies execution, checkpoint access and file structure; it cannot establish a memorization effect or estimate population extraction rates.

## Storage and execution

`verify_runpod.py` uses the private `researchdrive:` rclone configuration on the pod. Credentials are not part of this directory or the result bundle. Model weights download to `/workspace/finetuned-extraction/models`; they are not copied onto the Windows workspace or duplicated in Drive.

Results are uploaded to `LLaDA-ft/sliding-window-verification/<run>/` in Drive. The controller verifies a tiny write/read before downloading the checkpoints. Each worker runs on a separate GPU; the initial check uses five of the seven RTX 4090s. Reusing the cached weights for subsequent work avoids another full download.

On the configured pod:

```bash
cd /workspace/finetuned-extraction/code
/workspace/finetuned-extraction/venv/bin/python -u verify_runpod.py
```

A successful `verification_summary.json` requires all five workers to finish and all 48 rows to pass validation. Worker logs and provenance manifests accompany the normal results. Failed workers preserve their output for diagnosis.
