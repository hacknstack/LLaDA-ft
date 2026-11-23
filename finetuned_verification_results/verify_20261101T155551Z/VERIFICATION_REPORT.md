# Fine-tuned extraction verification — 1 November 2026

All five saved epoch-3 fine-tuned models passed the small end-to-end check on the seven-RTX-4090 pod. Every checkpoint file was checked against its saved SHA-256 before loading. The final Drive result upload passed a file-hash comparison against the pod's outputs.

| Model | Scored windows | Peak allocated GPU memory | Scoring and output time |
|---|---:|---:|---:|
| LLaDA 8B | 24 | 15.43 GiB | 229.76 s |
| Llama 2 7B | 6 | 12.59 GiB | 0.65 s |
| Llama 3 8B | 6 | 15.01 GiB | 0.68 s |
| Mistral 7B v0.1 | 6 | 13.53 GiB | 0.74 s |
| OLMo 7B | 6 | 12.87 GiB | 0.64 s |

Times exclude checkpoint downloads and model loading. Memory is the PyTorch allocator peak; the CUDA process also uses memory outside that allocator. These measurements cover the small verification workload, not a full sweep or a throughput benchmark.

The check used frozen training article 1734 and excluded odd-index article 1695, which has no normalized exact-text duplicate in training. Each supplied word-sliding windows 0, 1 and 2, using each model's native tokenizer. LLaDA tested two 50-position masks with ten path samples each under both `low-confidence` and `threshold`. Each AR model used exact suffix scoring. See `code/README.md` in the ZIP for the full method and changes.

All 48 rows have no recorded error. Probabilities and log probabilities passed consistency checks. All four AR models passed a separate teacher-forcing check of the suffix alignment. There are 16 normal CSV files and no detailed sampling traces.

At extraction threshold 0.001, Mistral's three training windows passed the threshold and its three excluded windows did not. All other tested windows were below the threshold. These are overlapping windows from only two articles, so this result does not establish a population memorization rate or a model ranking.

The probability estimator source is unchanged from the supplied experiment: SHA-256 `3fa3d3991237c6960a2aab950f71efb0df2c279436d96769f63f914b9a1705f4`. Checkpoint, dataset and environment provenance are included in the manifests and summaries.

Results: [Google Drive folder](https://drive.google.com/drive/folders/1WWfNZtxW2KYL9VmGDkUwHyGrCRqmIbZs).

The full extraction sweep has not started. Checkpoint weights remain cached on the pod at `/workspace/finetuned-extraction/models`. No model weights were written to Windows or duplicated in Drive.
