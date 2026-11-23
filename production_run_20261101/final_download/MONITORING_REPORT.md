# Initial production monitoring complete

Monitored 16.6 minutes from 2026-11-01 18:13:46 CET.

All four AR passes finished: 5,582 windows each. LLaDA has 592 / 133,968 valid calculations and all seven GPUs.
Completed shared training/control pairs: 1; failed calculations: 0. One intentional worker interruption recovered with the same seed.
Current estimated remaining time: 51.1 hours; estimate will update as more articles finish.

Verified a real Drive batch, automatic uploader restart, deterministic worker retry, ledger integrity, and generation of all 28 final CSVs on an isolated copy containing a completed pair. Final CSVs shared the same window set.

Verified Drive rows at this check: 22,706. Uploads continue every five minutes.

Automatic supervision is armed: worker retries, controller recovery, an independent budget guard, and pod stop after final upload or the deadline. RunPod API and CLI credentials passed read-only checks.

No detailed sample traces are produced. Probability helpers and approved document/window selection remain unchanged.
