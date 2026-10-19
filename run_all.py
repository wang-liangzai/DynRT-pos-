"""Rebuild features, fit models, verify, and summarize; resumes cached samples."""
import subprocess,sys
from pathlib import Path

root=Path(__file__).resolve().parent
for script in ('prepare.py','train.py','export_head.py','test_pipeline.py','make_report.py'):
    print('Running',script,flush=True)
    subprocess.run([sys.executable,str(root/script)],cwd=root,check=True)
