"""Check the notebook paths and input preparation without production training."""
import subprocess
python='/content/ar-ft-venv/bin/python'
subprocess.run([python,'-u','/content/validate_ar_final_only.py'],check=True)
for key in ['llama2_7b','llama3_8b']:
    subprocess.run([python,'-u','/content/prepare_ar_notebook_data.py','/content/'+key+'_final_only_config.json'],check=True)
print('BOTH_LLAMA_NOTEBOOK_INPUT_PATHS_VALIDATED',flush=True)
