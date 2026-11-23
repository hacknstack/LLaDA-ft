# RunPod GPU comparison — 1 November 2026

The RTX 4090 was faster and cheaper for the measured pipeline on these two RunPod instances. This is an inference-only hardware pilot, not the full extraction experiment.

| Measurement | A100 SXM4 80 GB | RTX 4090 24 GB |
|---|---:|---:|
| Quoted GPU price/hour | $1.79 | $0.89 |
| Total for 16 evaluations | 125.26 s | 94.50 s |
| low-confidence: mean seconds/evaluation | 8.45 | 6.18 |
| threshold: mean seconds/evaluation | 7.20 | 5.63 |
| Peak allocated GPU memory | 15.43 GiB | 15.43 GiB |
| Peak reserved GPU memory | 15.48 GiB | 15.48 GiB |

Across this case mix, the RTX delivered **1.33x throughput** and **2.67x as many evaluations per GPU dollar**. Prices are the user's quoted rental rates; installation/download time, idle time and storage charges are excluded from these steady evaluation measurements.

Both pods ran the same unchanged `probabilistic_extraction.py` and `get_log_likelihood.py`, verified by source SHA-256. The instrumentation code hash, public model revision and selected input token hashes also match. Both use PyTorch 2.8.0+cu128 and Transformers 4.46.3, four CPU threads, BF16 model weights, FP64 estimator arithmetic, model batch size one, full decoding, temperature 1, ten trajectories and state caching. Threshold is 0.9. The environment sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` for the estimator's existing deterministic-evaluation requirement; deterministic execution was not disabled to make the benchmark run.

The pilot uses the pinned **untrained LLaDA Base** (`0f2787f2d87eac5eed8a087d5ecd24277e6255b2`). No authenticated Colab runtime was available for Drive access, so the fine-tuned checkpoint has not been timed. Input articles are from the frozen BBC dataset revision and original global even/odd membership rule, with normalized training overlaps excluded from controls. One training article (filtered index 1734) and one clean odd control (1695) supply one 100-token window each. Suffix-50 and alternating-50 masks, both remasking schemes, and two matched timing repetitions give sixteen evaluations per GPU. These are not all twelve masks or a representative sample of the whole corpus.

The probabilities pass the estimator's numerical checks, but BF16 execution on different GPU architectures does not yield identical logits or sampled trajectories even with the same seed. The GPU reports record different probabilities and forward-call counts. This is not a cross-hardware bitwise-equivalence result. For the scientific sweep, use a consistent GPU/backend policy, retain seeds/configuration/provenance, and recheck performance and numerical behavior on the fine-tuned model before deciding the final subset or mixing hardware types.

The A100 evaluated 2,486 model states and the RTX 2,700; the RTX's advantage therefore was not caused by evaluating fewer states. Host CPU and GPU platform differences are part of these measured pod results; this is not a claim that every RTX 4090 setup universally outperforms every A100.

Both pilot processes completed. No production sweep, code adaptation to fine-tuned checkpoints, Google Drive result upload, pod stop or pod termination was performed. No large file was copied to Windows. Raw per-evaluation results and source hashes are in the neighboring JSON files.
