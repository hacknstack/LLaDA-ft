"""Adapt I/O/loading of the supplied runner; preserve estimator source verbatim."""
import ast
import hashlib
import re
from pathlib import Path
import shutil

root=Path(__file__).resolve().parent
original=root/'sliding window experiments'
target=root/'finetuned_sliding_windows';target.mkdir(exist_ok=True)
for name in ['probabilistic_extraction.py','get_log_likelihood.py']:
    shutil.copyfile(original/name,target/name)
source=(original/'sliding_window_extraction.py').read_text(encoding='utf-8')

def change(old,new):
    global source
    assert old in source,old[:100]
    source=source.replace(old,new)

change('import argparse\n','import argparse\nimport os\nimport hashlib\nfrom functools import lru_cache\nos.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")\n')
change("parser.add_argument('--model-name', type=str, default=None)",
       "parser.add_argument('--model-name', type=str, required=True, help='Verified local fine-tuned checkpoint directory')")
source=re.sub(r'^DEFAULT_\w+_MODEL = .*\n','',source,flags=re.MULTILINE)
source=re.sub(r'    if args.model_name is None:\n.*?        args.model_name = default_models\[args.model_family\]\n\n','',source,flags=re.DOTALL)
change('"Model: allenai/OLMo-7B-0724-hf."','f"Checkpoint: {model_name}."')
change('def _load_tokenizer(model_name: str):','@lru_cache(maxsize=1)\ndef _load_tokenizer(model_name: str):')
change('return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)',
       'return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)')
change('            use_fast=False,','            use_fast=False,\n            local_files_only=True,')
change('def _load_model(model_name: str, model_family: str, device: str):',
       '@lru_cache(maxsize=1)\ndef _load_model(model_name: str, model_family: str, device: str):')
change("            torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,\n        ).to(device).eval()",
       "            torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,\n            device_map={'': device}, low_cpu_mem_usage=True, local_files_only=True,\n        ).eval()")
change('    prompt_tokens = torch.tensor([prefix_ids], dtype=torch.long)',
       '    args._last_log_probability = None\n    prompt_tokens = torch.tensor([prefix_ids], dtype=torch.long)')
change("        return float(result['probability']), None",
       "        args._last_log_probability = result.get('log_probability')\n        return float(result['probability']), None")
change("        return float(result['probability']), result.get('verbose_steps')",
       "        args._last_log_probability = result.get('log_probability')\n        return float(result['probability']), result.get('verbose_steps')")
change("    if args.mode in {'exact', 'path_sampling'} or str(decoding_scheme).lower() == 'elbo':",
       "    args._last_log_probability = result.get('log_probability')\n    if args.mode in {'exact', 'path_sampling'} or str(decoding_scheme).lower() == 'elbo':")
change('    args = parse_args()','''    args = parse_args()
    checkpoint_path = Path(args.model_name).resolve()
    marker_path = checkpoint_path / 'COMPLETE.json'
    if not checkpoint_path.is_dir() or not marker_path.is_file():
        raise ValueError('A local fine-tuned checkpoint with COMPLETE.json is required; base fallback disabled')
    checkpoint = json.loads(marker_path.read_text())
    if checkpoint.get('epoch') != 3:
        raise ValueError('Expected a completed epoch-3 fine-tuned checkpoint')
    checkpoint_provenance = dict(path=str(checkpoint_path), run=checkpoint['run'],
        epoch=checkpoint['epoch'], update=checkpoint['update'],
        model_revision=checkpoint.get('model_revision'),
        complete_sha256=hashlib.sha256(marker_path.read_bytes()).hexdigest())''')
change("datetime.now().strftime('%Y%m%d_%H%M%S')","datetime.now().strftime('%Y%m%d_%H%M%S_%f')")
change("        p_z = float('nan')","        p_z = float('nan')\n        log_p_z = float('nan')")
change("            if verbose_file is not None:","            log_p_z = getattr(args, '_last_log_probability', None)\n            if verbose_file is not None:")
change("                'p_z': p_z,","                'p_z': p_z,\n                'log_p_z': log_p_z,")
change("                'p_z',","                'p_z',\n                'log_p_z',")
change("        'input_file': str(args.txt_path),","        'checkpoint': checkpoint_provenance,\n        'input_file': str(args.txt_path),")
change("    print('Top 5 windows by p_z (char_start, p_z):')","    print('Top 5 windows by p_z (char_start, p_z):')")
change("        print(f\"  {r['char_start']}, {float(r['p_z']):.8f}\")", "        print(f\"  {r['char_start']}, {float(r['p_z']):.8f}\")\n    return run_dir")
ast.parse(source)
(target/'sliding_window_extraction.py').write_text(source,encoding='utf-8')
assert all((original/n).read_bytes()==(target/n).read_bytes() for n in ['probabilistic_extraction.py','get_log_likelihood.py'])
print('FINE_TUNED_RUNNER_BUILT_ESTIMATORS_UNCHANGED')
