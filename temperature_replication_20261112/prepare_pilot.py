import hashlib
import json
import re
from pathlib import Path

ROOT = Path('/workspace/temperature-study')
MASKS = [list(range(1, 101, 2)), [i for i in range(1, 101) if ((i-1)//10)%2 == 0],
         [i for i in range(1, 101) if not 24 < i < 75],
         [i for i in range(1, 101) if i % 50 in range(1, 26)], list(range(1, 51)),
         list(range(2, 101, 2)), [i for i in range(1, 101) if i % 4 in (0, 3)],
         [i for i in range(1, 101) if (i-1)%10 in range(5, 10)],
         [i for i in range(1, 101) if (i-1)%20 in range(10, 20)], list(range(25, 75)),
         [i for i in range(1, 101) if (i-1)%50 in range(25, 50)], list(range(51, 101))]
LABELS = ['1,3,5,7', '1-10,21-30', '1-24,75-100', '1-25,51-75', '1-50', '2,4,6,8',
          '3,4,7,8', '6-10,16-20', '11-20,31-40', '25-74', '26-50,76-100', '51-100']


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/llada-base', trust_remote_code=True, local_files_only=True)
    cohorts = json.loads((ROOT/'input_cohorts.json').read_text())
    all_keys = {(r['document'], int(r['window_number'])) for rows in cohorts.values() for r in rows}
    documents = {}
    for name in sorted({key[0] for key in all_keys}):
        path = ROOT/'texts'/(name+'.txt')
        text = path.read_text(encoding='utf-8')
        words = [m.start() for m in re.finditer(r'\S+', text)]
        documents[name] = (text, [0] + words[1:])
    tokens = {}
    invalid = []
    for name, index in sorted(all_keys):
        text, starts = documents[name]
        ids = tokenizer(text[starts[index]:starts[index]+800], add_special_tokens=False)['input_ids']
        if len(ids) < 100:
            invalid.append([name, index, len(ids)])
        else:
            tokens[f'{name}:{index}'] = ids[:100]
    selected = {}
    for cohort, reference in [('validation', 'validation_reference'), ('exact', 'exact_reference')]:
        rows = cohorts[reference]
        choices = []
        for label in LABELS:
            pool = sorted([r for r in rows if r['mask'] == label],
                          key=lambda r: (float(r['reference_p_z']), r['document'], int(r['window_number'])))
            for fraction in (0.2, 0.8):
                row = dict(pool[round((len(pool)-1)*fraction)])
                row['mask_id'] = LABELS.index(label)+1
                row['selection_fraction'] = fraction
                choices.append(row)
        selected[cohort] = choices
    for cohort in ('runtime', 'application', 'negative'):
        rows = cohorts[cohort]
        choices = []
        for i, label in enumerate(LABELS):
            pool = [r for r in rows if r['mask'] == label]
            row = dict(min(pool, key=lambda r: hashlib.sha256((cohort+r['document']+r['window_number']).encode()).hexdigest()))
            row['mask_id'] = i+1
            choices.append(row)
        selected[cohort] = choices
    (ROOT/'pilot_inputs.json').write_text(json.dumps(dict(tokens=tokens, selected=selected, masks=MASKS, labels=LABELS)))
    summary = dict(unique_windows=len(all_keys), valid_windows=len(tokens), invalid_windows=invalid,
                   selection_counts={k: len(v) for k,v in selected.items()},
                   source_hashes={name: hashlib.sha256((ROOT/'texts'/(name+'.txt')).read_bytes()).hexdigest() for name in documents})
    (ROOT/'results/input_preflight.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
