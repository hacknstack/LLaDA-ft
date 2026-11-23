"""Prepare pinned model and deterministic BBC subset on Colab Google Drive."""
import os
from pathlib import Path

ROOT = Path('/content/drive/MyDrive/LLaDA-ft')
os.environ['HF_HOME'] = str(ROOT / 'hf-cache')
os.environ['HF_DATASETS_CACHE'] = str(ROOT / 'hf-cache' / 'datasets')
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'

import json
import datetime
import requests
from huggingface_hub import HfApi, snapshot_download


def main():
    api = HfApi()
    model_id = 'GSAI-ML/LLaDA-8B-Base'
    dataset_id = 'RealTimeData/bbc_news_alltime'
    model_rev = api.model_info(model_id).sha
    data_rev = api.dataset_info(dataset_id).sha
    info = requests.get('https://datasets-server.huggingface.co/splits', params={'dataset': dataset_id}, timeout=120)
    info.raise_for_status()
    configs = sorted({r['config'] for r in info.json()['splits'] if r['config'] >= '2025-03'})
    print('CONFIGS', configs, flush=True)
    (ROOT/'datasets').mkdir(exist_ok=True)
    manifest = dict(model_id=model_id, model_revision=model_rev, dataset_id=dataset_id,
                    dataset_revision=data_rev, configs=configs,
                    cutoff='2025-03-01', parity='global even indices AFTER date filtering; monthly configs sorted, original row order')
    (ROOT/'datasets'/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Downloading model to Drive', model_rev, flush=True)
    model_path = snapshot_download(model_id, revision=model_rev, allow_patterns=['*.json','*.py','*.safetensors','*.model','*.txt','*.tiktoken'], max_workers=4)
    manifest['model_path'] = model_path
    (ROOT/'datasets'/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('MODEL_READY', model_path, flush=True)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    output = ROOT/'datasets'/'bbc_even_after_2025_02.jsonl'
    stats, filtered_i, selected, tokens, empty_selected = [], 0, 0, 0, 0
    sample_ids = []
    with output.open('w') as out:
        for config in configs:
            ds = load_dataset(dataset_id, config, split='train', revision=data_rev)
            eligible = kept = 0
            for source_i, row in enumerate(ds):
                date = str(row.get('published_date') or '')[:10]
                try:
                    date = datetime.date.fromisoformat(date).isoformat()
                except ValueError:
                    continue
                if date < '2025-03-01':
                    continue
                idx = filtered_i
                filtered_i += 1
                eligible += 1
                if idx % 2:
                    continue
                content = row.get('content') or ''
                record = dict(filtered_index=idx, source_config=config, source_row=source_i,
                              published_date=date, title=row.get('title'), link=row.get('link'), content=content)
                out.write(json.dumps(record, ensure_ascii=False)+'\n')
                selected += 1
                kept += 1
                if not content.strip():
                    empty_selected += 1
                    continue
                ids = tokenizer.encode(content, add_special_tokens=False) + [tokenizer.eos_token_id]
                tokens += len(ids)
                if len(sample_ids) < 128 * 2048:
                    sample_ids.extend(ids)
            stats.append(dict(config=config, source_rows=len(ds), eligible=eligible, selected=kept))
            print('DATA_PROGRESS', stats[-1], flush=True)
    import numpy as np
    np.save(ROOT/'datasets'/'benchmark_tokens.npy', np.asarray(sample_ids[:128*2048], dtype=np.int32))
    manifest.update(months=stats, eligible_articles=filtered_i, selected_articles=selected,
                    selected_empty_articles=empty_selected, total_tokens_with_eos=tokens,
                    packing='Concatenate article content + EOS, split into 1024-token blocks; final block padded, padding excluded from loss',
                    blocks_1024=(tokens+1023)//1024, sample_token_count=min(len(sample_ids),128*2048))
    (ROOT/'datasets'/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('PREPARATION_COMPLETE',json.dumps(manifest),flush=True)


if __name__ == '__main__':
    main()
