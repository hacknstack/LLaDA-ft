"""Offline notebook checks; no model downloads, CUDA or Drive access required."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
from unittest.mock import patch

import nbformat

HERE = Path(__file__).resolve().parent
notebook = nbformat.read(HERE / 'LLaDA_BBC_3epochs_Colab.ipynb', as_version=4)
nbformat.validate(notebook)
scripts = {}
for cell in notebook.cells:
    if cell.cell_type != 'code':
        continue
    source = cell.source
    if source.startswith('%%writefile '):
        header, source = source.split('\n', 1)
        scripts[Path(header.split(' ', 1)[1]).name] = source
    ast.parse(source)
    assert not cell.outputs and cell.execution_count is None

# Account for overlapping uploads with exactly the trainer's retention/cadence.
epochs_kept = 0
rolling = 1  # initial Base transport preflight
peak = 1
exports = 1
for update in range(1, 184):
    step_in_epoch = (update - 1) % 61 + 1
    periodic = update % 20 == 0 and 61 - step_in_epoch > 5
    end_epoch = step_in_epoch == 61
    if periodic or end_epoch:
        peak = max(peak, epochs_kept + rolling + 1)
        exports += 1
        rolling = int(periodic)
        epochs_kept += int(end_epoch)
assert (peak, exports, epochs_kept, rolling) == (4, 10, 3, 0)
assert "(config['epochs']+1)*payload+2*2**30" in scripts['train_llada_bbc.py']
assert "(config['epochs']+1)*8015581184*2+2*2**30" in scripts['notebook_drive_preflight.py']

# Exercise real selection/filtering across month boundaries with an offline source.
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    (root / 'datasets').mkdir()
    shared = types.ModuleType('benchmark_llada')
    shared.ROOT = root
    datasets = types.ModuleType('datasets')
    loads = []
    sizes = {'2025-03': 2697, '2025-04': 2571, '2025-05': 2704, '2025-06': 2671}

    def load_dataset(dataset, config, **kwargs):
        assert kwargs['revision'] == '8dd1ecdc92ac43f9c04a3da3e945537dbb08179b'
        loads.append(config)
        offset = sum(size for month, size in sizes.items() if month < config)
        rows = [dict(published_date='invalid', content='excluded'),
                dict(published_date='2025-02-28', content='excluded')]
        rows.extend(dict(published_date=config + '-01',
                         content='' if offset+i in range(0, 14, 2) else f'article {offset+i}')
                    for i in range(sizes[config]))
        return rows

    datasets.load_dataset = load_dataset
    toy_files = {name: dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
                 for name, content in [('train_tokens.npy', b'train'), ('control_tokens_clean.npy', b'control')]}
    tree = ast.parse(scripts['notebook_prepare_data.py'])
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'TOKEN_FILES':
            node.value = ast.parse(repr(toy_files), mode='eval').body
    ast.fix_missing_locations(tree)

    def materialize_tokens(*args, **kwargs):
        for name, data in [('train_tokens.npy', b'train'), ('control_tokens_clean.npy', b'control')]:
            (root / 'datasets' / name).write_bytes(data)
        (root / 'datasets/training_data_manifest.json').write_text(json.dumps(dict(
            model_revision='0f2787f2d87eac5eed8a087d5ecd24277e6255b2',
            dataset_revision='8dd1ecdc92ac43f9c04a3da3e945537dbb08179b',
            train_tokens=3941989, train_blocks_4096=963, token_files=toy_files)))

    def prepare():
        with patch.dict(sys.modules, {'benchmark_llada': shared, 'datasets': datasets}), \
             patch('subprocess.run', side_effect=materialize_tokens), \
             contextlib.redirect_stdout(io.StringIO()):
            exec(compile(tree, 'notebook_prepare_data.py', 'exec'), {'__name__': '__main__'})

    prepare()
    rows = [json.loads(line) for line in (root / 'datasets/bbc_even_after_2025_02.jsonl').read_text().splitlines()]
    assert [row['filtered_index'] for row in rows] == list(range(0, 10643, 2))
    assert sum(not row['content'].strip() for row in rows) == 7
    april_first = next(row for row in rows if row['source_config'] == '2025-04')
    assert april_first['filtered_index'] == 2698 and april_first['source_row'] == 3
    assert len(loads) == 4
    prepare()
    assert len(loads) == 4, 'Verified cache should not redownload/reselect.'
    (root / 'datasets/train_tokens.npy').write_bytes(b'corrupt')
    prepare()
    assert len(loads) == 8, 'Corrupt cache must rebuild and reverify.'

# Use the embedded uploader, with a minimal MediaFileUpload double. SDK networking
# is deliberately excluded; the real failure/order/ownership checks execute.
uploader = types.ModuleType('drive_checkpoint_upload')
exec(scripts['drive_checkpoint_upload.py'], uploader.__dict__)
google_api = types.ModuleType('googleapiclient')
google_http = types.ModuleType('googleapiclient.http')
class MediaFileUpload:
    def __init__(self, path, **kwargs):
        assert Path(path).is_file()
google_http.MediaFileUpload = MediaFileUpload
with patch.dict(sys.modules, {'drive_checkpoint_upload': uploader,
                              'googleapiclient': google_api, 'googleapiclient.http': google_http}):
    exec(scripts['validate_drive_checkpoint_upload.py'], {'__name__': '__main__'})

# Check that final verification actually rejects incomplete runs and changed files.
with tempfile.TemporaryDirectory() as temp:
    config_path = Path(temp) / 'config.json'
    config_path.write_text(json.dumps({'run_name': 'offline_test'}))
    source = scripts['notebook_verify_final.py'].replace('/content/llada_training_config.json', str(config_path).replace('\\', '/'))
    status = {'status': 'complete'}
    changed = False

    class FakeStore:
        def __init__(self): self.service = self
        def folder(self, *args, **kwargs): return 'folder'
        def children(self, parent, name): return [{'id': name}]
        def files(self): return self
        def get_media(self, fileId): self.id = fileId; self.media = True; return self
        def get(self, fileId, **kwargs): self.id = fileId; self.media = False; return self
        def execute(self, **kwargs):
            if not self.media:
                return dict(name='COMPLETE.json', size='2', md5Checksum='changed' if changed else 'expected')
            if self.id == 'status_api.json':
                return json.dumps(status).encode()
            if self.id.endswith('_upload_receipt.json'):
                epoch = int(self.id.split('-')[1].split('_')[0])
                return json.dumps(dict(verified=True, epoch=epoch, folder_url='test', files={
                    'COMPLETE.json': dict(id=f'marker-{epoch}', size='2', md5Checksum='expected')})).encode()
            return json.dumps(dict(epoch=int(self.id.split('-')[1]), run='offline_test')).encode()

    fake_module = types.ModuleType('drive_checkpoint_upload')
    fake_module.DriveCheckpointStore = FakeStore
    def final_check():
        with patch.dict(sys.modules, {'drive_checkpoint_upload': fake_module}), contextlib.redirect_stdout(io.StringIO()):
            exec(source, {'__name__': '__main__'})
    final_check()
    changed = True
    try:
        final_check()
    except AssertionError:
        pass
    else:
        raise AssertionError('Cloud checksum mismatch was accepted')
    changed = False
    status['status'] = 'running'
    try:
        final_check()
    except AssertionError:
        pass
    else:
        raise AssertionError('Incomplete run was accepted')

print('NOTEBOOK_VALIDATED: format, syntax, clean outputs, frozen global parity, empty retention,')
print('verified-cache reuse, corrupt-cache rebuild, upload gates, final completion/checksum gates.')
