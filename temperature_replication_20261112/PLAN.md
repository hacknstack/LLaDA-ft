# Temperature replication plan

Status: bounded GPU benchmarking only; the complete experimental suite has not started.

The original pretrained checkpoints and frozen CSV cohorts in Downloads are authoritative.
The three explicit PDF window lists match all 300, 1,000, and 100 original window identities.
Use notebook mask definitions and labels, including both-edges 1-24/75-100 and middle 25-74.
Use 100-token, word-aligned windows, 800-character tokenizer chunks, no special tokens,
native BF16 model forwards, full vocabulary, and the original FP64 confidence estimators.
Temperature changes token sampling, while confidence ranking and threshold 0.9 remain untempered.

| Case | Cohort | Temperatures | Decoder / estimator | Samples |
|---|---:|---|---|---:|
| Correctness / convergence | 3,600 window-mask pairs | 0.5, 2 | Both confidence decoders; STS and MC | 500 STS / 5,000 MC |
| Exact validation | 12,000 reduced-mask pairs | 0.5, 2 | Both confidence decoders; exact DP and STS | exact / 500 STS |
| Threshold regression | Same 12,000 reduced-mask pairs | 1 | Threshold; exact DP and STS | exact / 500 STS |
| Runtime | Deferred to A100 | -- | Excluded from this pod by user instruction | -- |
| Remasking comparison | 3,600 pairs | 0.5, 2 | Reuse confidence STS; full random paths; deterministic greedy-confidence | 500 / 500 / exact path |
| Geometry / LLaDA application | 22,788 pairs | 0.5, 2 | Both confidence decoders; STS | 10 |
| AR baselines | CSV-derived shared starts | 0.5, 2 | Four AR base models; exact full-vocabulary suffix scoring | exact |
| Post-release controls | 15,384 pairs | 0.5, 2 | Both confidence decoders; STS | 10 |

No 5-million-trial study. No wall-clock study on this pod: the user requires A100 measurements.
Observed MIG durations are exclusively for resource planning and implementation optimization.
No re-selection of cohorts based on the new temperatures.
The AR comparison reuses the 10-sample LLaDA application rows, rather than launching a second copy.
Convergence prefix averages share their 500-trajectory source.

Export each temperature, experiment, decoder, estimator, model, and timing budget separately.
Preserve the corresponding example headers from Downloads: four-column probability CSVs,
eight prefix-average columns plus 500 individual log estimates for STS convergence, and interleaved
individual log estimates for convergence. Keep budget-planning durations and seeds in metadata;
do not publish MIG timing observations as paper runtime measurements.
Do not add a scheme or temperature column to the requested CSV layouts. Keep log-probability metadata
so that probability underflow does not destroy the underlying result. Upload incrementally to Drive;
weights and large working files stay on the pod. A refreshed Drive authorization may be needed.

Benchmark controls: eight explicit MIG UUIDs, persistent model replicas, CPU threads 2/3/4,
singleton forwards for exact confidence comparisons, and bounded representative tasks. A native-logit
cache may reuse identical model inputs across temperatures/decoders only after a bitwise result check.
Random paths must evaluate the full requested sample count and all 50 reveal steps: current helper
defaults of 128 paths and 25 steps are unsuitable for this replication and are explicitly disabled.

Budget forecasts must distinguish measured timings from sample-count extrapolation and include
uncertainty for configuration heterogeneity. New temperatures can make the old MC reference cohort
unresolved; retain zero-hit rows and omit undefined relative errors rather than replacing probabilities.
