"""Small correctness, corpus-identity and source-precision audits; no training run."""
import os
os.environ.update(HF_HUB_CACHE='/content/ar-ft-cache/hub', HF_HUB_DISABLE_XET='1',
                  TOKENIZERS_PARALLELISM='false', USE_TF='0', USE_FLAX='0')
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from safetensors import safe_open
from ar_model_probe import ar_loss, ROOT, OUT
from ar_probe_optimizer import CpuMasterAdamW

torch.set_num_threads(12)
torch.manual_seed(12)
tiny = LlamaForCausalLM(LlamaConfig(vocab_size=31, hidden_size=32, intermediate_size=64,
                                  num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, max_position_embeddings=64))
ids = torch.randint(0, 31, (2, 17))
output = tiny(input_ids=ids, labels=ids, use_cache=False)
torch.testing.assert_close(ar_loss(output.logits, ids), output.loss, rtol=1e-6, atol=1e-7)
one_logits = torch.randn(1, 1, 31, requires_grad=True)
zero = ar_loss(one_logits, ids[:1, :1])
zero.backward()
assert zero.item() == 0 and torch.count_nonzero(one_logits.grad).item() == 0
before = tiny(ids, use_cache=False).logits[:, :8].detach()
changed = ids.clone()
changed[:, 8:] = (changed[:, 8:] + 1) % 31
after = tiny(changed, use_cache=False).logits[:, :8].detach()
torch.testing.assert_close(before, after, rtol=0, atol=0)

# Check full/remainder averaging through the exact master-weight implementation.
model = torch.nn.Linear(5, 3)
reference = torch.nn.Linear(5, 3)
reference.load_state_dict(model.state_dict())
config = dict(peak_learning_rate=1e-5, betas=[.9, .999], epsilon=1e-8, weight_decay=.1, gradient_clip_norm=1.)
opt = CpuMasterAdamW(model.parameters(), config)
ref = torch.optim.AdamW(reference.parameters(), lr=1e-5, betas=(.9,.999), eps=1e-8, weight_decay=.1, foreach=False, fused=False)
for count in (16, 3):
    x = torch.randn(count, 5)
    y = torch.randint(0, 3, (count,))
    ref.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(reference(x), y).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., foreach=False)
    ref.step()
    opt.begin()
    for i in range(count):
        (torch.nn.functional.cross_entropy(model(x[i:i+1]), y[i:i+1]) / count).backward()
    opt.step(1e-5)
    for p, r in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, r, rtol=1e-6, atol=1e-7)

report = dict(objective_matches_native_hf=True, causal_future_invariance=True,
              one_token_zero_loss=True, full_and_remainder_optimizer_equivalence=True,
              package_versions={p:importlib.metadata.version(p) for p in
                 ['torch','transformers','tokenizers','huggingface-hub','accelerate','safetensors']},
              source_sha256={name:hashlib.sha256(Path('/content',name).read_bytes()).hexdigest()
                 for name in ['ar_model_probe.py','ar_probe_optimizer.py','ar_probe_audits.py']})
print('AR_OBJECTIVE_AND_OPTIMIZER_CORRECTNESS_PASSED', flush=True)

rows = [json.loads(line) for line in (ROOT/'datasets/bbc_even_after_2025_02.jsonl').open()]
texts = [row['content'] for row in rows if row['content'].strip()]
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base',
    revision='0f2787f2d87eac5eed8a087d5ecd24277e6255b2', trust_remote_code=True)
encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False,
                    return_token_type_ids=False)['input_ids']
tokens = np.asarray([t for ids in encoded for t in ids+[tokenizer.eos_token_id]], dtype=np.int32)
serialized = io.BytesIO()
np.save(serialized, tokens)
hash_value = hashlib.sha256(serialized.getvalue()).hexdigest()
assert len(tokens)==3941989 and hash_value=='f5e29db9e7b1459bf030cac1567f77f6595da59acdfaf010ab977b21fa0254df'
report['frozen_articles_reproduce_canonical_llada_token_sha256'] = hash_value
report['frozen_article_file_sha256'] = hashlib.sha256((ROOT/'datasets/bbc_even_after_2025_02.jsonl').read_bytes()).hexdigest()
print('FROZEN_ARTICLES_REPRODUCE_CANONICAL_LLADA_TOKENS', flush=True)

controls = [json.loads(line) for line in (ROOT/'datasets/bbc_odd_controls.jsonl').open()]
clean = [row['content'] for row in controls if not row['empty'] and not row['exact_normalized_overlap_with_train']]
assert len(controls)==5321 and len(clean)==3345
control_encoded=tokenizer(clean,add_special_tokens=False,return_attention_mask=False,return_token_type_ids=False)['input_ids']
control_tokens=np.asarray([t for ids in control_encoded for t in ids+[tokenizer.eos_token_id]],dtype=np.int32)
serialized=io.BytesIO()
np.save(serialized,control_tokens)
control_hash=hashlib.sha256(serialized.getvalue()).hexdigest()
assert len(control_tokens)==2617971 and control_hash=='3a6968777b4ae68e4ed1f89b3309ab4e1133a43a2a659bf7fe561dc70416e27d'
report['frozen_controls_reproduce_canonical_llada_token_sha256']=control_hash
report['native_control_tokens'] = {}
report['source_weight_dtypes'] = {}
for path in sorted(OUT.glob('*_4096.json')):
    probe = json.loads(path.read_text())
    if probe.get('status')!='ok':
        continue
    model_path = Path(probe['model_path'])
    tok = AutoTokenizer.from_pretrained(model_path)
    encoded = tok(clean,add_special_tokens=False,return_attention_mask=False,return_token_type_ids=False)['input_ids']
    report['native_control_tokens'][probe['key']] = sum(len(ids)+1 for ids in encoded)
    dtypes = {}
    for shard in model_path.glob('*.safetensors'):
        with safe_open(shard,framework='pt',device='cpu') as f:
            for name in f.keys():
                tensor=f.get_slice(name)
                dtype=tensor.get_dtype()
                dtypes[dtype]=dtypes.get(dtype,0)+math.prod(tensor.get_shape())
    report['source_weight_dtypes'][probe['key']] = dtypes
    print('SOURCE_DTYPES',probe['key'],dtypes,flush=True)

import google.auth
from googleapiclient.discovery import build
credentials,_ = google.auth.default()
service=build('drive','v3',credentials=credentials,cache_discovery=False)
quota=service.about().get(fields='storageQuota').execute(num_retries=5)['storageQuota']
report['drive_storage_quota_bytes'] = quota
if quota.get('limit'):
    report['drive_free_bytes'] = int(quota['limit'])-int(quota.get('usage',0))
(OUT/'auxiliary_audits.json').write_text(json.dumps(report,indent=2))
print('AR_AUXILIARY_AUDITS_PASSED',json.dumps(report),flush=True)
