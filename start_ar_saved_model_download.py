"""Start actual Drive download work without occupying GPU memory."""
import subprocess
with open('/content/ar_saved_models_download.log','a') as output:
    process=subprocess.Popen(['/content/ar-ft-venv/bin/python','-u','/content/test_saved_ar_models.py','prepare'],
                             stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
print('DRIVE_MODEL_DOWNLOAD_STARTED',process.pid,flush=True)
