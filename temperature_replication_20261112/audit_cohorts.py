import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DOWNLOADS = Path.home() / 'Downloads'
NAMES = {
    'MIT License': 'MITLicense', 'The Zen of Python': 'TheZenOfPython',
    'United States Declaration of Independence': 'DeclarationTranscript',
    '3-Clause BSD License': '3-ClauseBSDLicense', 'ISC License': 'ISCLicense',
    'The Unlicense': 'Unlicense', 'Gettysburg Address': 'GettysburgAddress',
    'Apache License': 'ApacheLicense', 'United States Bill of Rights': 'BillOfRights',
    'Psalm 23': 'Psalm23', 'Sonnet 18': 'Sonnet18', 'Book of Genesis': 'GenesisNumbered',
    'Lorem ipsum': 'lorem ipsum', 'United States Constitution': 'ConstitutionIntro',
}


def csv_keys(name):
    with (DOWNLOADS / name).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    return {(r['document'], int(r['window_number'])) for r in rows}, rows


def main():
    text = '\n'.join((WORKSPACE / f'tmp/pdfs/paper_page_{i:02}.txt').read_text(encoding='utf-8')
                     for i in range(22, 28))
    starts = [text.index('A.  300-window selection'), text.index('B.  1000-window selection'),
              text.index('C.  100-window selection'), len(text)]
    pattern = re.compile(r'([a-n])\)[ \t]+([^\n(]+?)\s+\((\d+) windows?\)\.\s*:\s*\{([^}]+)\}', re.S)
    report = {}
    datasets = [('validation_300', 'interesting_300_windows.csv', 300),
                ('exact_1000', 'interesting_1000_windows_long_format.csv', 1000),
                ('runtime_100', 'ASTRA_path_sampling_low-confidence_samples125_logs_times.csv', 100)]
    for i, (label, filename, expected) in enumerate(datasets):
        matches = pattern.findall(text[starts[i]:starts[i + 1]])
        assert len(matches) == 14, (label, len(matches))
        pdf_keys = set()
        documents = []
        for _, title, count, numbers in matches:
            title = re.sub(r'\s+', ' ', title).strip()
            document = NAMES[title]
            indices = list(map(int, re.findall(r'\d+', numbers)))
            assert len(indices) == int(count), (document, len(indices), count)
            pdf_keys.update((document, n) for n in indices)
            documents.append({'document': document, 'count': len(indices), 'indices': indices})
        actual, rows = csv_keys(filename)
        assert len(pdf_keys) == expected
        report[label] = dict(pdf_windows=len(pdf_keys), csv_windows=len(actual), csv_rows=len(rows),
                             exact_match=pdf_keys == actual,
                             missing_from_csv=sorted(pdf_keys - actual), extra_in_csv=sorted(actual - pdf_keys),
                             documents=documents, source_csv=filename,
                             csv_sha256=hashlib.sha256((DOWNLOADS / filename).read_bytes()).hexdigest())
    report['all_match'] = all(v['exact_match'] for v in report.values())
    (ROOT / 'cohort_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'documents'} if isinstance(v, dict) else v
                      for k, v in report.items()}, indent=2))


if __name__ == '__main__':
    main()
