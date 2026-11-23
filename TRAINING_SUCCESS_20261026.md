# Verified LLaDA BBC training result

Checked on 26 October 2026 from a new Colab runtime using authenticated, read-only Drive API requests. No training was started and no Drive artifacts were modified.

The latest successful run is `bbc_fp32_colab_20261026_084128_323830`. Its API journal reports completion at 2026-10-26 14:54:56 UTC after three epochs and 183 optimizer updates (approximately 6 hours 12 minutes including initialization, diagnostics and checkpoint uploads).

| Epoch | Update | Saved weight bytes | Verified model folder |
|---|---:|---:|---|
| 1 | 61 | 16,031,197,080 | [epoch-001](https://drive.google.com/drive/folders/1HqywBKuxG1OoTrtl9mcxu-QmKQ6gerrMl) |
| 2 | 122 | 16,031,197,080 | [epoch-002](https://drive.google.com/drive/folders/1RUZqbexnmK33tb79qsW1RwvcutXx5U3m) |
| 3 | 183 | 16,031,197,080 | [epoch-003: final model](https://drive.google.com/drive/folders/1j87zoSGYh2m6VFsexiKNhCERKHN3wS9G) |

For every epoch, all 20 files are present, including nine safetensors shards, tokenizer/configuration/custom-code files and the completion marker. Current server file IDs, sizes, MD5 and SHA-256 match the recorded upload receipts; file SHA-256 and sizes also match the saved completion manifests. The model index references all nine existing shards and matches the manifest's 291 state tensors. The model-weight checksums differ from the saved untrained Base transport export. This audit checked cloud integrity without downloading model weights or performing an inference/load test.

| Fixed-noise diagnostic objective | Base | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|---:|
| Train | 2.551750 | 2.443219 | 2.416932 | 2.387241 |
| Clean odd controls | 2.595585 | 2.521869 | 2.509995 | 2.501128 |

These diagnostics support a functioning training run; they do not measure verbatim memorization or extraction. Snapshots are model-only, without the FP32 optimizer state needed for exact training resumption.

An older notebook run, `bbc_fp32_colab_20261022_203953_074010`, also completed all three epochs, finishing on 23 October UTC. Its final model is [here](https://drive.google.com/drive/folders/1iqTLby4zWHFmyZxr9iGeonxkzQRfbJn_). The original `bbc_fp32_3ep_v3_api` run has only a verified epoch-1 checkpoint and a step-80 periodic snapshot; its stale status stops at update 85 and is not evidence of current execution in this new runtime. No old checkpoints were deleted.

Full audit evidence, configuration, environment and diagnostic records: `drive_training_audit_20261026.json`.
