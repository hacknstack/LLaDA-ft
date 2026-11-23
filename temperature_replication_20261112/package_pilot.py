import csv
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DOWNLOADS = Path.home() / 'Downloads'

FILES = {
    'validation': 'interesting_300_windows_by_mask.csv',
    'exact': 'interesting_1000_windows_long_format.csv',
    'runtime': 'ASTRA_path_sampling_low-confidence_samples125_logs_times.csv',
    'application': 'Application_path_sampling_low-confidence_samples100_pz.csv',
    'negative': 'Negative_path_sampling_low-confidence_samples100_pz.csv',
}


def main():
    inputs = {}
    for label, filename in FILES.items():
        with (DOWNLOADS / filename).open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        inputs[label] = [{k: r[k] for k in ('document', 'window_number', 'mask')}
                         | {'selection_p_z': r.get('p_z')} for r in rows]
    for label, filename in [('validation_reference', 'ASTRA_LowConfSTS_samples500_estimators_with_logs.csv'),
                            ('exact_reference', 'ASTRA_LowConfexact_pz.csv')]:
        with (DOWNLOADS / filename).open(newline='', encoding='utf-8-sig') as handle:
            inputs[label] = [{k: r[k] for k in ('document', 'window_number', 'mask')}
                             | {'reference_p_z': r.get('p_z', r.get('avg_500'))}
                            for r in csv.DictReader(handle)]
    (ROOT / 'input_cohorts.json').write_text(json.dumps(inputs), encoding='utf-8')
    documents = {r['document'] for rows in inputs.values() for r in rows}
    with ZipFile(ROOT / 'pilot_source.zip', 'w', ZIP_DEFLATED) as archive:
        for filename in ['probabilistic_extraction.py', 'get_log_likelihood.py', 'check_temperature_consistency.py']:
            archive.write(WORKSPACE / 'finetuned_sliding_windows' / filename, 'code/' + filename)
        for path in ROOT.glob('*.py'):
            archive.write(path, 'code/' + path.name)
        for name in documents:
            path = WORKSPACE / 'sliding window experiments' / 'texts' / (name + '.txt')
            assert path.exists(), path
            archive.write(path, 'texts/' + path.name)
        archive.write(ROOT / 'input_cohorts.json', 'input_cohorts.json')
        archive.write(ROOT / 'cohort_audit.json', 'cohort_audit.json')
    print('PILOT_PACKAGE_BYTES', (ROOT / 'pilot_source.zip').stat().st_size)


if __name__ == '__main__':
    main()
