"""Recover complete but unrenamed HF blobs on the Drive FUSE mount."""
from pathlib import Path
import hashlib
import json
root=Path('/content/drive/MyDrive/LLaDA-ft')
manifest=json.loads((root/'datasets/manifest.json').read_text())
path=Path(manifest['model_path'])
for file in path.glob('*.safetensors'):
    if file.exists():
        continue
    target=file.parent / file.readlink()
    incomplete=target.with_name(target.name+'.incomplete')
    if not incomplete.exists():
        raise RuntimeError(f'Missing download: {file.name}')
    print('Verifying SHA256',file.name,flush=True)
    with incomplete.open('rb') as f:
        checksum=hashlib.file_digest(f,'sha256').hexdigest()
    assert checksum==target.name, f'Incomplete or corrupt file {file.name}'
    incomplete.rename(target)
    assert file.exists()
    print('Recovered',file.name,flush=True)
print('CACHE_VERIFIED',flush=True)
