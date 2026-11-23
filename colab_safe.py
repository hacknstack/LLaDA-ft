"""Run the existing Windows wrapper while redacting credential-bearing output."""
import os
from pathlib import Path
import re
import subprocess
import sys

wrapper=Path(os.environ['TEMP'])/'codex-colab-cli-source/colab_windows.py'
p=subprocess.run([sys.executable,str(wrapper),*sys.argv[1:]],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
s=p.stdout.decode('utf-8',errors='replace')
s=re.sub(r'(https?://[^\s?]+)\?[^\s]+',r'\1?[QUERY_REDACTED]',s)
s=re.sub(r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+','[JWT_REDACTED]',s)
sys.stdout.write(s)
raise SystemExit(p.returncode)
