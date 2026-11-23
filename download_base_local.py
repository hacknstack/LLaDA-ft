"""Fetch the exact same pinned checkpoint onto Colab's local POSIX filesystem."""
import os
os.environ['HF_HOME']='/content/hf-cache-local'
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download
from pathlib import Path
import json

revision='0f2787f2d87eac5eed8a087d5ecd24277e6255b2'
p=Path(snapshot_download('GSAI-ML/LLaDA-8B-Base',revision=revision,local_dir='/content/llada-base-model',
    allow_patterns=['*.json','*.py','*.safetensors','*.model','*.txt','*.tiktoken'],max_workers=4))
index=json.loads((p/'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    assert (p/name).is_file(),name
assert sum((p/n).stat().st_size for n in set(index['weight_map'].values()))==16031197112
print('PINNED_LOCAL_BASE_READY',str(p),revision,flush=True)
