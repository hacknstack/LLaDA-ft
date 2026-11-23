"""Run real training as blocking notebook-kernel work; no synthetic keepalive."""
import subprocess
from pathlib import Path

log=Path('/content/drive/MyDrive/LLaDA-ft/logs/bbc_fp32_3ep_v3_api.log')
with log.open('a') as output:
    print('Starting checkpoint preflight and training; inspect the API status journal for progress.',flush=True)
    subprocess.run(['/content/llada-bench-venv/bin/python','-u','/content/train_llada_bbc.py',
        '--config','/content/llada_training_config_v3.json'],stdout=output,stderr=subprocess.STDOUT,check=True)
print('Training process finished successfully.',flush=True)
