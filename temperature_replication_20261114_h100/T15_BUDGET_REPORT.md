# Temperature 1.5: bounded H100 evaluation

For 300 windows, 12 masks, both low-confidence and threshold, 500 STS / 5,000 MC samples, the conditional forecast is **132 hours (5.5 days), $2,330** at $17.70/pod-hour. Planning sensitivity: 79?211 hours / $1,398?$3,728. These are heuristic forecasts, not confidence bounds.

Relative to the earlier T2 central forecast, T1.5 is approximately 1.6x faster, or 37% less time. It remains approximately 50x slower than the earlier T0.5 prediction. These comparisons concern T1.5 alone, not a combined multi-temperature run.

## Measured probes

- 48 matched cases per temperature: one selected window per mask, all four method/decoder combinations, STS16 / MC64. T1.5: 109.05s; T2: 132.77s (1.22x speedup).
- Four matched larger cases per temperature, covering all method/decoder combinations, STS500 / MC500. T1.5: 66.84s; T2: 210.20s (3.14x speedup).
- Four T1.5 cases at the requested STS500 / MC5000 counts: 82.69s in total. There is no directly measured T2 full-count counterpart.

The difference between short- and large-probe ratios shows why a single throughput multiplier or linear temperature interpolation is inadequate. Low-confidence MC and STS improve substantially in the larger cases. Threshold MC showed little benefit in the broad short probe (T1.5/T2 duration ratio approximately 1.046).

## Forecast construction

For each method/decoder group, transfer the prior T2 cohort-wide cost using the average of (1) its broad short-probe T1.5/T2 duration ratio and (2) its primary full-count ratio. For STS, both primary temperatures were measured at 500 samples. For MC, T1.5 was measured at 5,000; T2 is extrapolated from the new MC500 case using the earlier group-specific scaling exponent. Summing all groups yields the central estimate. This gives equal influence to the wider input coverage and the larger sample counts. It is a conservative planning heuristic; it is not a statistical fit. The 10% utilization reserve inherited from the T2 estimate remains.

The machine was busy across all four GPUs during the initial probes. Recorded durations include each case's broker/queue waiting and overlap, so per-group task durations are not isolated GPU execution times. They are used as matched scheduling observations. Small-probe concurrency and source coverage remain limitations. The CSV and JSON contain every ratio and transferred cost, making the forecast reviewable.

## Unresolved numerical validation and status

The same mathematical algorithms, model dtype, sample counts and mask definitions were used. Native H100 BF16 batches still change logits, sampled paths and STS estimates relative to scalar inference; the fast forecast is conditional on numerical validation before full launch. This evaluation does not resolve that issue or provide a T1.5 scalar-preserving forecast.

All probe processes exited. The full 300-window experiment has not started. The pod remains running and billed. These are budget/optimization observations only; the paper wall-clock study remains excluded and reserved for A100. No paper runtime results were produced.
