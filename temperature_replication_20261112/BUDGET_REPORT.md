# Temperature replication: measured budget plan

**Status: the benchmark is complete; the full experiment has not started.**

At the supplied $5.66/hour pod price, the central estimate is **$3,577**, or **632 pod-hours (26.3 continuous days)**.
The planning sensitivity range is $2,146–$5,724. It reflects configuration variation and extrapolation, not a statistical confidence interval.

**The paper wall-clock study is excluded and reserved for A100 hardware.** The five-million-trial study is also excluded. MIG durations are budget and optimization observations only; they are not A100 runtime results.

| Experiment | Estimated pod-hours | Estimated cost |
|---|---:|---:|
| STS versus MC | 422.8 | $2,393 |
| Geometry and LLaDA AR comparison | 36.0 | $204 |
| Post-release controls | 35.2 | $199 |
| Exact versus STS (including T1 threshold) | 60.0 | $340 |
| Random remasking | 77.2 | $437 |
| Greedy-confidence remasking | 0.8 | $5 |
| Four AR baselines | 0.0 | $0 |

The plan retains the original 300-window correctness/remasking cohort, 1,000-window exact cohort, 1,899 application starts, and 15,384 negative-control window-mask pairs. Both temperatures are 0.5 and 2; threshold exact/STS additionally gets a fresh temperature-1 pass. Correctness uses 500 STS / 5,000 MC; exact validation uses 500 STS with 10 masked tokens; application and controls use 10 STS; AR remains exact.

planned_csv_outputs.csv lists all 38 planned scientific CSVs and their exact header templates from Downloads (331,080 aggregate rows). Decoders and temperatures are separated by filename. AR mask fields retain the original CSV convention. No scientific CSV results are claimed complete by this budget package.

The PDF 300-, 1,000-, and 100-window index lists exactly match the supplied CSVs. The CSVs and notebook mask boundaries are authoritative. All 1,899 application starts have 100 valid native tokens under all four AR tokenizers. All four AR models passed 80-window/two-temperature probes.

The pod has eight 23.625-GiB MIG slices across four RTX PRO 6000 cards. Explicit MIG UUID assignment exposes all eight to separate persistent workers. A repeated calibration batch took 54.9 seconds alone and 54.6 seconds with eight workers simultaneously. CPU thread counts 2/3/4 had similar throughput; two threads per worker leaves CPU headroom. The forecast allows 10% scheduling/utilization overhead.

Exact-state native-logit reuse reduced the checked five-exact/five-STS bundle from 394 to 130 seconds with identical probabilities/log probabilities; that cache benefit is included in measured exact-bundle costs. State prefetch batching preserved the checked individual STS log weights and MC outcomes. The forecast uses conservative 1.35x T=2 STS and 1.5x T=2 MC gains. Only batch shapes 1/2/16 are used by that prototype; other tested shapes sometimes changed native logits and are not assumed interchangeable. The bounded lookahead holds at most 16 projected states instead of thousands.

Cross-case allocation now serves four independent estimator contexts from each resident model, combining ready inference requests across contexts. The all-eight-slice probe improved throughput by 17.6% with bit-identical checked results. The forecast discounts this incremental gain to 1.132x and applies it only to T=2 confidence sampled workloads, retaining the 10% utilization reserve. See ALLOCATION_PLAN.md for validation scope and remaining orchestration work.

Random reveal paths are independent of temperature, so the same 500 complete 50-step paths can be scored at both temperatures from shared forwards. The paired prototype matched both separate runs exactly and measured approximately 2x speedup; the forecast uses 1.9x. No shortened-path or reduced-sample approximation is used. AR similarly shares each forward between temperatures.

The projected bill before the new prefetch and paired-random optimizations was $5,631; exact-state caching was already included in that baseline. The extrapolation details, sample-count scaling assumptions, and measured case counts are in budget_estimate.json. MC5000 costs are forecasts from shorter measured cases and observed scaling; no long full-suite run was performed.

A smaller alternative, **not adopted**, would use 100 rather than 300 windows for correctness and remasking while retaining their sample counts, all masks, and the other cohorts. Its proportional central estimate is approximately **$1,687**. A fixed, source-balanced subset would need to be chosen before launch. This would change the original study size.

All benchmark processes were checked stopped. The pod itself remains running and continues to be billed. No automatic pod-stop command has been issued. Full-run orchestration and renewable Drive upload authorization still need preparation before the complete suite is launched. Model weights stay on the pod.
