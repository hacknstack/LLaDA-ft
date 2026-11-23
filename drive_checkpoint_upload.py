"""Resumable Drive API uploads, with server-side size and MD5 verification.

Upload session URLs stay in memory and are never printed or stored on Drive.
Existing named files are reused only when content checksums match.
"""
from pathlib import Path
import hashlib
import io
import json
import time

FOLDER='application/vnd.google-apps.folder'
FIELDS='id,name,size,md5Checksum,sha256Checksum,webViewLink,parents,appProperties'


def quoted(value):
    return str(value).replace('\\','\\\\').replace("'","\\'")


def hashes(path):
    md5=hashlib.md5()
    sha=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            md5.update(chunk);sha.update(chunk)
    return md5.hexdigest(),sha.hexdigest()


class DriveCheckpointStore:
    def __init__(self,service=None):
        if service is None:
            import google.auth
            from googleapiclient.discovery import build
            credentials,_=google.auth.default()
            service=build('drive','v3',credentials=credentials,cache_discovery=False)
        self.service=service

    def children(self,parent,name=None):
        query=f"'{quoted(parent)}' in parents and trashed=false"
        if name is not None:
            query+=f" and name='{quoted(name)}'"
        result=[]
        token=None
        while True:
            response=self.service.files().list(q=query,fields=f'nextPageToken,files({FIELDS},mimeType)',
                pageSize=1000,pageToken=token).execute(num_retries=5)
            result.extend(response.get('files',[]))
            token=response.get('nextPageToken')
            if not token:
                return result

    def folder(self,parent,name,create=True):
        matches=[f for f in self.children(parent,name) if f['mimeType']==FOLDER]
        if len(matches)>1:
            raise RuntimeError(f'Ambiguous Drive folder name: {name}')
        if matches:
            return matches[0]['id']
        if not create:
            return None
        return self.service.files().create(body=dict(name=name,mimeType=FOLDER,parents=[parent]),fields='id').execute(num_retries=5)['id']

    def ensure_path(self,parts):
        parent='root'
        for part in parts:
            parent=self.folder(parent,part)
        return parent

    def write_state(self,parent,name,data):
        """Overwrite only the small API-owned run-state mirror, not model artifacts."""
        from googleapiclient.http import MediaIoBaseUpload
        existing=self.children(parent,name)
        if len(existing)>1:
            raise RuntimeError('Ambiguous state mirror')
        media=MediaIoBaseUpload(io.BytesIO(json.dumps(data,indent=2).encode()),mimetype='application/json',resumable=False)
        if existing:
            if existing[0].get('appProperties',{}).get('codex_role')!='run_state':
                raise RuntimeError('Refusing to overwrite a state file not created by this uploader')
            return self.service.files().update(fileId=existing[0]['id'],media_body=media,fields=FIELDS).execute(num_retries=5)
        return self.service.files().create(body=dict(name=name,parents=[parent],appProperties={'codex_role':'run_state'}),
            media_body=media,fields=FIELDS).execute(num_retries=5)

    def delete_own_periodic(self,receipt):
        """Collect only a previous periodic folder whose children match our upload receipt."""
        folder=receipt['folder_id']
        expected={v['id'] for v in receipt['files'].values()}
        actual={v['id'] for v in self.children(folder)}
        if actual!=expected:
            raise RuntimeError('Periodic snapshot has unexpected children; automatic cleanup refused')
        for file_id in expected:
            self.service.files().delete(fileId=file_id).execute(num_retries=5)
        self.service.files().delete(fileId=folder).execute(num_retries=5)

    def upload(self,path,parent,progress=None):
        from googleapiclient.http import MediaFileUpload
        path=Path(path)
        size=path.stat().st_size
        md5,sha=hashes(path)
        existing=[f for f in self.children(parent,path.name) if f.get('mimeType')!=FOLDER]
        if len(existing)>1:
            raise RuntimeError(f'Ambiguous Drive file name: {path.name}')
        if existing:
            remote=existing[0]
            if int(remote.get('size',-1))==size and remote.get('md5Checksum')==md5:
                return dict(**remote,verified_md5=md5,local_sha256=sha,reused=True)
            raise RuntimeError(f'Refusing to overwrite different content: {path.name}')
        media=MediaFileUpload(str(path),mimetype='application/octet-stream',chunksize=64*1024*1024,resumable=True)
        request=self.service.files().create(body=dict(name=path.name,parents=[parent],appProperties={'sha256':sha}),
            media_body=media,fields=FIELDS)
        response=None
        try:
            while response is None:
                status,response=request.next_chunk(num_retries=8)
                if status is not None and progress:
                    progress(path.name,status.resumable_progress,size)
        except Exception as error:
            status=getattr(getattr(error,'resp',None),'status',None)
            raise RuntimeError(f'Upload failed for {path.name}: {type(error).__name__}, HTTP {status}; resumable URL withheld') from None
        remote=self.service.files().get(fileId=response['id'],fields=FIELDS).execute(num_retries=5)
        if int(remote.get('size',-1))!=size or remote.get('md5Checksum')!=md5:
            raise RuntimeError(f'Drive verification failed for {path.name}')
        if remote.get('sha256Checksum') and remote['sha256Checksum']!=sha:
            raise RuntimeError(f'Drive SHA256 mismatch for {path.name}')
        return dict(**remote,verified_md5=md5,local_sha256=sha,reused=False)

    def upload_directory(self,path,parent,progress=None):
        path=Path(path)
        if not (path/'COMPLETE.json').is_file():
            raise RuntimeError('Local checkpoint must be validated before upload')
        remote_files={}
        for p in sorted(path.iterdir()):
            if p.is_file() and p.name!='COMPLETE.json':
                remote_files[p.name]=self.upload(p,parent,progress)
        # Completion marker is uploaded only after every shard and sidecar is verified.
        remote_files['COMPLETE.json']=self.upload(path/'COMPLETE.json',parent,progress)
        return dict(folder_id=parent,folder_url=f'https://drive.google.com/drive/folders/{parent}',files=remote_files,
            verified=True,total_bytes=sum(int(v.get('size',0)) for v in remote_files.values()))


def inspect_project():
    store=DriveCheckpointStore()
    project=store.folder('root','LLaDA-ft',create=False)
    if not project:
        raise RuntimeError('Project folder not found via Drive API')
    print('PROJECT_FOLDER_ID',project)
    print('QUOTA',json.dumps(store.service.about().get(fields='storageQuota').execute()))
    def walk(parent,parts,depth):
        for item in store.children(parent):
            p=parts+[item['name']]
            if item['mimeType']==FOLDER:
                if depth:
                    walk(item['id'],p,depth-1)
            else:
                print('FILE',json.dumps(dict(path='/'.join(p),id=item['id'],size=item.get('size'),md5=item.get('md5Checksum'))))
                if item['name']=='status.json' and 'bbc_fp32_3ep_v1' in p:
                    raw=store.service.files().get_media(fileId=item['id']).execute(num_retries=5)
                    print('API_RUN_STATUS',raw.decode('utf-8'))
    for name in ['checkpoints','runs']:
        folder=store.folder(project,name,create=False)
        if folder:
            walk(folder,[name],3)
        else:
            print('ABSENT_FOLDER',name)


if __name__=='__main__':
    inspect_project()
