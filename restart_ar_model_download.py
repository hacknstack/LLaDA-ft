"""Replace only our own sequential download worker with bounded parallel I/O."""
from pathlib import Path
import os
import signal
import subprocess
for item in Path('/proc').iterdir():
    if not item.name.isdigit():continue
    try:args=(item/'cmdline').read_bytes().split(b'\0')
    except (OSError,PermissionError):continue
    if b'/content/test_saved_ar_models.py' in args and b'prepare' in args:
        os.kill(int(item.name),signal.SIGTERM)
        print('STOPPED_OWN_DOWNLOAD_WORKER',item.name)
with open('/content/ar_saved_models_download.log','a') as output:
    p=subprocess.Popen(['/content/ar-ft-venv/bin/python','-u','/content/test_saved_ar_models.py','prepare'],
                       stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
print('BOUNDED_PARALLEL_DOWNLOAD_STARTED',p.pid)
