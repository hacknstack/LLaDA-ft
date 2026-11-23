# Running sliding-window experiment

Run: `bbc_12masks_20261101T171020Z`

Started 1 November 2026 at 18:13 CET (17:13 UTC). The approved cohort is 350 training articles and 350 held-out controls, with 5,582 shared window starts. All four AR passes have finished. LLaDA continues all twelve masks with ten threshold and ten low-confidence samples per mask on seven RTX 4090s.

Live readable status (updated by the pod uploader):
https://drive.google.com/file/d/1Te-_GBdqLBVUrpctp6pVTfDWMryrGdbI/view

Results folder:
https://drive.google.com/drive/folders/1o22PYUqHJv8y_m5kvDt5ICuet5Ilty3q

The initial launch was monitored for more than 15 minutes. Worker interruption recovery, uploader restart, real Drive output checks, ledger integrity and final assembly of all 28 CSVs for a complete pair passed. No failed calculations were recorded at handoff; the one retry was a controlled recovery test.

The job is detached from SSH and chat. Durable task state is on the pod, uploads are verified about every five minutes, worker/controller supervision is active, and an independent guard controls the budget deadline. Successful final upload is followed by a pod stop request. RUN_COMPLETE.json indicates full completion; RUN_PARTIAL.json indicates partial completion. Final CSVs include only complete training/control pairs across all five models. Immutable batches retain other valid partial work.

Budget origin: 1 November at 17:26:37 CET, $490 at $6.40/hour. Stop adding waves after 4 November 14:11:37 CET. Compute cutoff: 4 November 19:56:37 CET. Hard pod-stop deadline: 4 November 20:26:37 CET, allowing approximately $10 headroom from the original balance. External platform/API failures can affect shutdown, so the live status and completion markers are the authoritative progress records.

SSH: `ssh ynbsa87gmjxhoq-644117d1@ssh.runpod.io -i ~/.ssh/id_ed25519`
Pod run directory: `/workspace/finetuned-extraction/runs/bbc_12masks_20261101T171020Z`
Pod source: `/workspace/finetuned-extraction/code/production_*.py`

No model weights or full article corpus were downloaded to Windows for this run. Drive and RunPod credentials remain in private configuration folders and are excluded from code and result bundles.
