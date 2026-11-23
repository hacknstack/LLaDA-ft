"""Exercise upload gates without performing network operations."""
from pathlib import Path
import tempfile
from drive_checkpoint_upload import DriveCheckpointStore


class Recorder(DriveCheckpointStore):
    def __init__(self,fail=False):
        self.calls=[];self.fail=fail
    def upload(self,p,parent,progress=None):
        self.calls.append(p.name)
        if self.fail and p.name=='model.bin':
            raise RuntimeError('Simulated interrupted shard upload')
        return dict(id=p.name,size=str(p.stat().st_size))


with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp)
    for name in ['model.bin','config.json','COMPLETE.json']:
        (p/name).write_text('{}')
    good=Recorder()
    result=good.upload_directory(p,'test-folder')
    assert result['verified'] and good.calls[-1]=='COMPLETE.json'
    bad=Recorder(fail=True)
    try:
        bad.upload_directory(p,'test-folder')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Interrupted upload was not rejected')
    assert 'COMPLETE.json' not in bad.calls

    class Existing(DriveCheckpointStore):
        def __init__(self): pass
        def children(self,parent,name=None):
            return [dict(id='existing',name='model.bin',mimeType='application/octet-stream',size='999',md5Checksum='wrong')]
    try:
        Existing().upload(p/'model.bin','test-folder')
    except RuntimeError as error:
        assert 'Refusing to overwrite' in str(error)
    else:
        raise AssertionError('Different existing file was overwritten')

    class Mismatch(DriveCheckpointStore):
        def __init__(self): self.service=self
        def children(self,*args): return []
        def files(self): return self
        def create(self,**kwargs): return self
        def next_chunk(self,**kwargs): return None,{'id':'test-id'}
        def get(self,**kwargs): return self
        def execute(self,**kwargs): return dict(id='test-id',size='2',md5Checksum='wrong')
    try:
        Mismatch().upload(p/'model.bin','test-folder')
    except RuntimeError as error:
        assert 'verification failed' in str(error)
    else:
        raise AssertionError('Checksum mismatch was accepted')

    class Unexpected(DriveCheckpointStore):
        def __init__(self): pass
        def children(self,*args): return [{'id':'unrelated'}]
    try:
        Unexpected().delete_own_periodic(dict(folder_id='test',files={'model':{'id':'ours'}}))
    except RuntimeError as error:
        assert 'unexpected children' in str(error)
    else:
        raise AssertionError('Cleanup of unexpected files was allowed')
print('DRIVE_UPLOAD_GATES_VALIDATED: marker-last, interruption, conflict, checksum mismatch, owned cleanup')
