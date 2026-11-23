"""CPU checks for final-only saving, integrity failures and matched AR math."""
import os
os.environ.update(USE_TF='0',USE_FLAX='0')
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import uuid
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from ar_training_utils import ar_loss,learning_rate,validate_checkpoint
from train_ar_bbc import export_final
from drive_checkpoint_upload import DriveCheckpointStore

torch.set_num_threads(12)
torch.manual_seed(9)
model=LlamaForCausalLM(LlamaConfig(vocab_size=31,hidden_size=16,intermediate_size=32,
    num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64))
ids=torch.randint(0,31,(2,13))
out=model(input_ids=ids,labels=ids,use_cache=False)
torch.testing.assert_close(ar_loss(out.logits,ids),out.loss)
one_logits=torch.randn(1,1,31,requires_grad=True)
zero=ar_loss(one_logits,ids[:1,:1]);zero.backward()
assert zero.item()==0 and torch.count_nonzero(one_logits.grad)==0
for total in (177,180,198,207):
    config=dict(warmup_fraction=.05,decay_fraction=.1,peak_learning_rate=1e-5,final_learning_rate=1e-6)
    assert abs(learning_rate(total,total,config)-1e-6)<1e-15
    import math
    assert learning_rate(math.ceil(.05*total),total,config)==1e-5

class FakeService:
    def about(self):return self
    def get(self,**kwargs):return self
    def execute(self,**kwargs):return dict(storageQuota=dict(limit=str(10**12),usage='0'))

class Recorder(DriveCheckpointStore):
    def __init__(self,fail=False):self.service=FakeService();self.calls=[];self.fail=fail
    def ensure_path(self,parts):assert parts[-1]=='final';return 'final-folder'
    def upload(self,path,parent,progress=None):
        self.calls.append(path.name)
        if self.fail and path.name.endswith('.safetensors'):
            raise RuntimeError('Simulated upload failure')
        content=path.read_bytes()
        return dict(id=path.name,name=path.name,size=str(len(content)),
            md5Checksum=hashlib.md5(content).hexdigest(),verified_md5=hashlib.md5(content).hexdigest(),
            local_sha256=hashlib.sha256(content).hexdigest())

class Tokenizer:
    def save_pretrained(self,path):(path/'tokenizer_config.json').write_text('{}')

model=model.to(torch.bfloat16)
with tempfile.TemporaryDirectory() as temp:
    root=Path(temp)
    (root/'config.json').write_text('{}');(root/'METHODS.md').write_text('test protocol')
    config=dict(expected_parameters=sum(p.numel() for p in model.parameters()),
        run_name='unit_final_'+uuid.uuid4().hex,epochs=3,expected_total_updates=198,
        model_id='tiny-test',model_revision='test-revision')
    events=[]
    def emit(event,**kwargs):events.append(event)
    store=Recorder()
    result=export_final(model,Tokenizer(),config,root,dict(config_sha256='config',methods_sha256='methods'),store,'journal',emit)
    assert result.endswith('/final-folder')
    assert store.calls[-2:]==['COMPLETE.json','final_upload_receipt.json']
    assert events[-1]=='final_model_verified'
    assert not (Path('/content/ar-final-staging')/config['run_name']/'final').exists()
    # Failed upload must not claim completion or discard the only local trained model.
    config['run_name']='unit_failure_'+uuid.uuid4().hex
    store=Recorder(fail=True);events.clear()
    try:
        export_final(model,Tokenizer(),config,root,dict(config_sha256='config',methods_sha256='methods'),store,'journal',emit)
    except RuntimeError as error:
        assert 'Simulated' in str(error)
    else:raise AssertionError('Upload failure accepted')
    assert 'COMPLETE.json' not in store.calls and 'final_model_verified' not in events
    staging=Path('/content/ar-final-staging')/config['run_name']/'final'
    assert staging.is_dir()
    assert staging.resolve().is_relative_to(Path('/content/ar-final-staging').resolve())
    shutil.rmtree(staging)

tree=ast.parse(Path('/content/train_ar_bbc.py').read_text())
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
calls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='export_final']
assert len(calls)==1
for node in ast.walk(run):
    if isinstance(node,(ast.For,ast.While)):
        assert not any(n in calls for n in ast.walk(node)),'Final model export inside a training loop'
guards=[n for n in ast.walk(run) if isinstance(n,ast.If) and ast.unparse(n.test)=="config['save_final_model']"]
assert len(guards)==1 and any(n in calls for n in ast.walk(guards[0]))
print('FINAL_ONLY_CHECKS_PASSED: native AR loss, zero targets, schedules, tiny safetensors validation, marker-last, failed-save retention, one guarded final export outside training loops')
