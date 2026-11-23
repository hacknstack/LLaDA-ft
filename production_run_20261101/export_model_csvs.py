import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'final_download'
OUT = ROOT / 'model_csvs'
MASKS = ['1,3,5,7', '1-10,21-30', '1-24,75-100', '1-25,51-75',
         '1-50', '2,4,6,8', '3,4,7,8', '6-10,16-20', '11-20,31-40',
         '25-74', '26-50,76-100', '51-100']


def main():
    manifest = json.loads((SOURCE / 'RUN_COMPLETE.json').read_text())
    assert manifest['status'] == 'complete'
    OUT.mkdir(exist_ok=True)
    by_model = {}
    task_ids = set()
    for entry in manifest['files']:
        path = SOURCE / entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], path
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == entry['rows']
        for row in rows:
            assert row['task_id'] not in task_ids
            task_ids.add(row['task_id'])
            assert math.isfinite(float(row['p_z'])) and 0 <= float(row['p_z']) <= 1
            assert row['p_z'] == row['probability']
            by_model.setdefault(row['model'], []).append(row)
    assert len(task_ids) == manifest['expected_total'] == 156296
    reference = None
    checks = {}
    for model, rows in sorted(by_model.items()):
        rows.sort(key=lambda r: (r['group'] != 'train', int(r['filtered_index']),
                                 int(r['window_index']), int(r['mask_id']), r['scheme']))
        keys = {(r['group'], r['filtered_index'], r['window_index']) for r in rows}
        reference = keys if reference is None else reference
        assert keys == reference and len(keys) == 5582
        assert len({(r['group'], r['filtered_index']) for r in rows}) == 700
        if model == 'llada':
            counts = Counter((int(r['mask_id']), r['scheme']) for r in rows)
            assert counts == Counter({(m, s): 5582 for m in range(1, 13)
                                      for s in ('threshold', 'low-confidence')})
            assert all(r['num_samples'] == '10' for r in rows)
        else:
            assert len(rows) == 5582
            assert all(r['scheme'] == 'exact' and r['mask_id'] == '12' for r in rows)
        fields = ['document', 'window_number', 'mask', 'p_z']
        partitions = ({f'llada_{scheme}': [r for r in rows if r['scheme'] == scheme]
                       for scheme in ('threshold', 'low-confidence')}
                      if model == 'llada' else {model: rows})
        for name, partition in partitions.items():
            assert len(partition) == (66984 if model == 'llada' else 5582)
            path = OUT / f'{name}.csv'
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in partition:
                    writer.writerow(dict(document=f"{row['group']}_{row['filtered_index']}",
                                         window_number=row['window_index'],
                                         mask=MASKS[int(row['mask_id']) - 1], p_z=row['p_z']))
            checks[name] = {'rows': len(partition), 'windows': len(keys),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / 'llada.csv').unlink(missing_ok=True)
    shutil.copy2(SOURCE / 'documents.csv', OUT / 'document_manifest.csv')
    (OUT / 'README.txt').write_text(
        'Completed run: bbc_12masks_20261101T171020Z\n'
        '700 articles: 350 training and 350 excluded controls. 5,582 common windows.\n'
        'All six result CSVs have exactly: document,window_number,mask,p_z.\n'
        'LLaDA has separate threshold and low-confidence files; 10 draws per estimate.\n'
        'Each LLaDA file has 66,984 rows; each AR model has 5,582 rows.\n'
        'document = group_filteredIndex, e.g. train_100; the index refers to the frozen\n'
        'March-June 2025 filtered dataset, not the selected-cohort row number.\n'
        'document_manifest.csv maps group + filtered_index to source metadata and hashes.\n'
        'window_number retains the original ZERO-BASED word-start window_index.\n'
        'Windows were sampled from each article; do not renumber them consecutively.\n'
        'mask strings are the user-supplied labels in notebook order; abbreviated labels\n'
        'represent the original full 50-position masks, not truncated literal masks.\n'
        'AR always uses mask 12 (51-100): exact suffix probability conditioned on 1-50.\n'
        'p_z is copied verbatim from verified source CSVs; no rounding or averaging.\n'
        'Zero p_z may reflect floating-point underflow; original Drive results retain log_p_z.\n'
        'All source files passed completion-manifest SHA256 checks. Unique tasks and\n'
        'identical window identities across all five models were verified.\n', encoding='utf-8')
    (OUT / 'validation.json').write_text(json.dumps(checks, indent=2) + '\n')
    archive = Path.home() / 'Downloads' / 'finetuned_sliding_window_results.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:
        for path in sorted(OUT.iterdir()):
            bundle.write(path, path.name)
    print(json.dumps({'models': checks, 'zip': str(archive), 'bytes': archive.stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
