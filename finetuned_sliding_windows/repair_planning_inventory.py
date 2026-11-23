"""Correct the original runner's leading-whitespace word-start convention."""
import json,time
from tokenizers import Tokenizer
from planning_inventory import ROOT,OUT,KEYS,corpus,count_windows

while not (OUT/'corpus_inventory.json').exists():time.sleep(10)
docs,_=corpus();leading=[r for r in docs if r['content'][0].isspace()]
models={}
for key in KEYS:
    file=OUT/('window_counts_'+key+'.json');data=json.loads(file.read_text())
    lookup={(r['group'],r['filtered_index']):r for r in data['rows']}
    tok=Tokenizer.from_file(str(ROOT/'models'/key/'tokenizer.json'))
    for r in leading:lookup[(r['group'],r['filtered_index'])]['windows']=count_windows(tok,r['content'])
    file.write_text(json.dumps(data));models[key]=lookup
path=OUT/'corpus_inventory.json';data=json.loads(path.read_text())
for d in data['documents']:
    ident=(d['group'],d['filtered_index'])
    d['windows_by_model']={k:models[k][ident]['windows'] for k in KEYS}
    d['common_windows']=min(d['windows_by_model'].values())
data['word_start_convention']='Original runner: initial character 0, followed by word_start_positions[1:]'
path.write_text(json.dumps(data,indent=2))
(OUT/'inventory_repaired.json').write_text(json.dumps(dict(status='complete',leading_whitespace_documents_checked=len(leading))))
print('INVENTORY_WORD_START_CONVENTION_VERIFIED',len(leading),flush=True)
