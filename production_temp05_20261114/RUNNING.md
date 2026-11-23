# Temperature 0.5 production experiment

Run: `temp0p5_validation300_20261114T115156Z`.

Drive destination: https://drive.google.com/drive/folders/1q3LQgLtuJ0fz3SPdMwatBg9hz20w1ss8

Pod: `pzibiqfkvlffyf`, four H100 80 GB GPUs, $17.70/hour supplied price. SSH: `root@64.247.201.47`, port 18231. Remote run directory: `/workspace/temperature-study/runs/temp0p5_validation300_20261114T115156Z`.

This runs the original pretrained GSAI-ML/LLaDA-8B-Base model at the pinned revision in config.json. The CSV-authoritative cohort contains 300 distinct document/window starts, each under all 12 notebook masks: 3,600 pairs per estimator/decoder. Only temperature 0.5 is included. Both low-confidence and threshold use 500 STS samples and 5,000 ordinary MC trials. Threshold is 0.9; windows are 100 native tokens, 50 masked positions, and 50 reveal steps. Native scalar BF16 forwards and original FP64 probability routines avoid the unresolved batched-logit differences on H100. No detailed step traces or paper wall-clock experiment are enabled.

Four separate CSVs each contain 3,600 rows when complete:

- `validation_temp0p5_low-confidence_STS500.csv`: exact 511-column Downloads reference header.
- `validation_temp0p5_threshold_STS500.csv`: exact 511-column Downloads reference header.
- `validation_temp0p5_low-confidence_MC5000.csv`: document, window_number, mask, avg_5000.
- `validation_temp0p5_threshold_MC5000.csv`: document, window_number, mask, p_z.

STS includes prefix averages at 1, 5, 10, 25, 50, 100, 250, and 500 samples, followed by 500 natural-log sample estimates. Mask strings retain the notebook/CSV labels. Output row order follows the original reference cohort. No scheme column is added. Each case has a stable independent seed recorded by the run configuration and jobs_for() implementation.

Eight real cases passed the smoke check and were reused by the full run. All headers, widths, row identities and STS average/log consistency passed validation. Drive uploads passed metadata size and MD5 readback checks.

Four persistent workers claim independent cases from a shared queue, append durable raw result journals, and never copy model weights to Windows. CSV snapshots update on Drive every five minutes. They are partial until status.json reports outcome complete and validation.json reports final true / completed 14400. The controller checks all 3,600 row identities per file before that final marker and verifies all final uploaded checksums.

After a successful verified save, the remote controller calls the authorized RunPod stop action. On failure it saves partial CSVs and an error report before stopping. A separate hidden Windows guardian (guardian.py) checks progress and pod status, provides a stop fallback, and retrieves final files from Drive after the pod reports stopped. If Drive upload fails, its fallback copies only the small run directory, including raw result journals, before stopping. It does not download model weights. The guardian initially ran as PID 33044; consult guardian.log for current status.

Final local downloads, after completion: `%USERPROFILE%\Downloads\temp0p5_validation300_20261114T115156Z\` and the sibling ZIP with the same run name. Drive is the primary durable destination; the remote shutdown does not depend on this chat staying open. The independent Windows guardian requires this Windows machine to stay running for its fallback and final local download.

Operational files: progress.json, production.log, drive_manifest.json, and shutdown_requested.json in the remote run directory. Local guardian logs are in this directory. Credentials remain outside the run directories and are excluded from outputs.

Stop API: POST https://api.runpod.io/v2/pods/{id}/action with action stop, as described by the official RunPod Python client: https://github.com/runpod/runpod-python/blob/main/runpod/api/ctl_commands.py . The key was read privately, and the pod identity was cross-checked against the supplied public IP and SSH port. No pod termination/deletion is configured.
