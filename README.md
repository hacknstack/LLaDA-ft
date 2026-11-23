# LLaDA fine-tuning and extraction experiments

Research code, Colab notebooks, training protocols, corpus audits, and recorded
results for continued pretraining and probabilistic extraction experiments.

## Experiments

- Full-parameter BBC-news continued pretraining of LLaDA-8B-Base, Llama 2 7B,
  Llama 3 8B, Mistral-7B-v0.1, and OLMo-7B.
- Sliding-window extraction with LLaDA low-confidence and threshold decoding,
  Monte Carlo and path-sampling estimators, and exact autoregressive comparisons.
- Temperature-replication measurements and controlled article-exposure studies.

## Main locations

| Location | Contents |
|---|---|
| Root `*.ipynb` | Model-specific Colab training notebooks |
| `train_llada_bbc.py`, `train_ar_bbc.py` | Earlier BBC training implementations |
| `LLADA_BBC_METHOD_v3.md`, `AR_FINAL_ONLY_NOTEBOOK_METHOD.md` | Earlier training methods and limitations |
| `sliding window experiments/` | Original extraction implementation and sample texts |
| `finetuned_sliding_windows/` | Fine-tuned-model extraction and production utilities |
| `temperature_replication_20261112/` | Paper-cohort audit, pilots, and budget planning |
| `temperature_replication_20261114_h100/` | H100 temperature comparisons and estimates |
| `exposure_protocol_20261123/` | Disjoint-article audit and controlled-exposure setup |
| `production_temp05_20261114/` | Temperature-0.5 extraction controller and shutdown guardian |

Historical configurations and status files describe their individual runs. They
are not a live indication that a training or extraction job is currently running.
Some preparation artifacts and final protocol revisions reside on the remote
runtime or Google Drive; consult the corresponding setup reports and manifests.

## Execution

This is an experiment workspace, not a single installable Python package. Use the
notebook or setup instructions associated with the experiment you intend to run.
Dependencies, native model/tokenizer revisions, datasets, and resource assumptions
are recorded by the individual protocols and environment reports.

Many historical utilities contain machine-specific local paths, SSH endpoints,
or Drive folder IDs. Review those settings before executing them. In particular,
some production controllers can launch paid GPU jobs or stop a specified pod.

Model weights, token arrays, dataset caches, credentials, temporary logs, and
duplicate ZIP exports are excluded from Git. Existing small text/CSV/JSON research
artifacts are retained. Keep authentication credentials in the external private
locations expected by the relevant scripts; never commit tokens or private keys.

Git preserves file bytes rather than normalizing line endings, because several
research manifests record checksums of frozen input/configuration files.
