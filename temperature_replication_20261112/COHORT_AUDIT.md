# Temperature replication: cohort audit

The PDF appendix lists match all three original CSV cohorts exactly.

| Cohort | PDF windows | CSV windows | Missing | Extra |
|---|---:|---:|---:|---:|
| validation_300 | 300 | 300 | 0 | 0 |
| exact_1000 | 1000 | 1000 | 0 | 0 |
| runtime_100 | 100 | 100 | 0 | 0 |

Window indices remain zero-based. The notebook masks, including its both-edges and middle-block boundaries, are authoritative.
The frozen cohorts will be reused at T=0.5 and T=2, plus a fresh T=1 threshold exact/STS comparison.
No new probability-based cohort selection is planned.

The application CSV has 1,899 window starts. All four downloaded AR CSVs contain finite probabilities at all 1,899 starts; this does not explain the paper's 1,889-start count. Preserve the full original cohort and record any new tokenizer validity exclusions explicitly.
