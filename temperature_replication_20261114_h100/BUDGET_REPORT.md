# Four-H100 300-window MC versus STS budget

At $17.70 per pod-hour, the conditional batched forecast is **213 hours (8.9 days), $3,767**. This is for the 300-window CSV cohort, all 12 notebook masks, both low-confidence and threshold, both temperatures, 500 STS and 5,000 MC samples per estimate. There are eight result CSVs, 28,800 aggregate rows, 7.2 million requested STS samples and 72 million requested MC trials. Duplicate-state reuse and original early exits remain intact.

| Temperature | Pod-hours | Cost |
|---|---:|---:|
| 0.5 | 2.6 | $46 |
| 2 | 210.2 | $3,720 |
| Total | 212.8 | $3,767 |

Sensitivity range: 128?340 hours, $2,260?$6,026. This is not a confidence interval. Source/window variation and MC5000 extrapolation remain important uncertainties.

Four full H100 80 GB GPUs were verified. The pinned base model was downloaded to the pod. The matched T2 four-case probe took 39.27 seconds with batching versus 191.43 on the prior MIG configuration; a larger STS125/MC500 probe took 114.07 seconds. An eight-case STS32/MC128 probe took 55.28 seconds. The forecast conservatively uses a 4x per-GPU transfer gain against the previous optimized MIG forecast, accounts for four rather than eight GPUs, and retains a 10% utilization reserve. T0.5 uses eight full-count case probes totalling 18.78 seconds, with two cases per method/decoder group. It covers fewer sources than the older source-stratified pilot and is a weaker cohort-wide estimate.

**Numerical validation is unresolved for the fast configuration.** Native BF16 batching on these H100s changes logits relative to scalar inference. The mathematical estimator, model dtype, masks, seeds, decoding rules and sample counts are unchanged, but seeded paths and STS estimates are not bit-identical. Disabling BF16 reduced-precision reduction did not remove the differences. The old pod's bit-identity guarantee cannot be carried over. These isolated probe copies record differences instead of rejecting the run; production code was not changed to accept them. A full launch should first validate batched inference numerically. The central forecast is conditional on that validation.

If scalar-preserving inference is required, the measured matched-case throughput instead projects approximately **545 hours / $9,645**. Concurrent singleton inference streams were checked and did not materially improve the scalar broker. This transfer estimate also has substantial short-probe uncertainty.

No exact/STS comparison, geometry, negative-control, random/greedy, AR, five-million-trial, or paper wall-clock study is included. All timing here is for budgeting and optimization; the paper runtime study remains reserved for A100. No full experiment was launched. All probe processes were verified exited. The pod itself remains running and billed. Model weights and source texts remain on the pod.
