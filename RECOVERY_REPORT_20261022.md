# Checkpoint incident and recovery: 22 October 2026

The old run `bbc_fp32_3ep_v1` has no model checkpoint in Drive. Both mounted files and a direct Drive API inspection show its last persisted state as epoch 1, optimizer update 54 of 61 in that epoch, at 21 October 2026 21:14:09 UTC (23:14:09 Europe/Zurich). There is no persisted epoch-completion or checkpoint-start event. The old runtime has been replaced. Its in-memory trained weights and optimizer state are unavailable.

This evidence does not establish failure of a large checkpoint upload: no checkpoint attempt is recorded. The exact shutdown cause and any unpersisted activity are unknown. The old status file saying `running` is stale, not evidence of a surviving process.

A separate fault was confirmed while initializing a replacement: the mounted original-model cache lacked its first weight shard. An input-cache failure must not be confused with a failed export of trained weights. That initialization failed before any optimizer update and is retained as `bbc_fp32_3ep_v2_api`.

A 2,948,598,128-byte original-model shard was uploaded from local Colab disk to Drive with resumable API requests. Server file size, MD5 and SHA256 matched the local values. This took about 85.6 seconds after the local copy. Only the newly created test copy was deleted; its verification receipt remains on Drive. It was a storage test, not a trained checkpoint.

The replacement run is `bbc_fp32_3ep_v3_api`. It downloads the same pinned model revision onto local Colab disk, bypassing the damaged Drive cache. Model and tokenizer revisions, selection, numerical method, seeds, schedule and exposure budget remain unchanged. Fresh baseline diagnostic values matched the original run exactly: training objective 2.5517503088340163 and clean-control objective 2.595585035160184. These are monitoring losses, not memorization rates.

Before optimization, the replacement performs a complete Base-model serialization, validation, resumable API upload and remote checksum verification. This artifact is clearly labelled `base-transport-preflight`; it is not fine-tuned. A remote completion marker is uploaded only after all model files and sidecars verify. The run then saves periodic model snapshots every 20 updates (skipping those immediately before epoch boundaries) as well as all three epoch models. Only an older periodic snapshot whose file IDs exactly match the run's receipt may be removed after a newer verified copy exists.

Actual work executes through a blocking notebook-kernel subprocess rather than only an SSH background process. This is not a synthetic keepalive and does not guarantee that Colab will never expire. Independent API-owned state and upload receipts are under `LLaDA-ft/run-api-records/bbc_fp32_3ep_v3_api/`; model exports are under `LLaDA-ft/checkpoints/bbc_fp32_3ep_v3_api/`.

Checkpoints remain model-only. They preserve models for analysis but do not provide exact optimizer resumption. The method, incident, extra transfer time, space checks and these limitations are documented in `LLADA_BBC_METHOD_v3.md`. Previous run artifacts are preserved. No trained result from the interrupted run is presented as recovered.

## Complete checkpoint path verified

The complete v3 Base transport-preflight export finished at 22 October 2026 17:55:45 UTC (19:55:45 Europe/Zurich). It contains nine weight shards totaling 16,031,197,080 bytes and 20 files overall, including the last-uploaded completion marker. The independent upload receipt records matching server/local MD5 checksums and matching SHA256 checksums for every file. The delivered protocol and configuration hashes also match the immutable production run record.

This validates full-model local serialization, validation, resumable upload, and remote persistence. It is still the untrained Base model at epoch 0/update 0. The production training-start event followed at 17:55:49 UTC. The preflight snapshot is temporary and will be superseded by the first verified periodic trained snapshot; it is not an epoch-1 result. Original compute-only forecasts exclude these transfers and should not be used as an upper bound for total wall-clock time.

The first replacement optimizer update completed at 17:57:12 UTC (19:57:12 Europe/Zurich), with finite loss 2.3009927057862282, finite gradient norm, and a confirmed FP32 master-weight change. This is a start-of-run observation, not a claim that three epochs have completed. The stable [checkpoint parent folder](https://drive.google.com/drive/folders/1T0dBVPKa8rNV8CnnpdmBVus1cMYMt7GC) will contain the retained trained snapshots as they become verified.
