"""Compare full BF16 and 8-bit-state AdamW at original LLaDA context length."""
import argparse
import json
import subprocess
import sys
from benchmark_llada import ROOT, worker

CONFIGS = [
    dict(name='full_adamw_bf16_4096',kind='full',optimizer='adamw'),
    dict(name='full_adamw8bit_4096',kind='full',optimizer='adam8'),
]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');args=parser.parse_args()
    if args.worker:
        worker(next(c for c in CONFIGS if c['name']==args.worker),sequence_length=4096,stress_mask=True)
    else:
        for c in CONFIGS:
            print('START',c['name'],flush=True)
            with (ROOT/'logs'/(c['name']+'.log')).open('w') as f:
                subprocess.run([sys.executable,__file__,'--worker',c['name']],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
            print((ROOT/'benchmarks'/(c['name']+'.json')).read_text(),flush=True)
        print('BENCHMARK_4096_COMPLETE',flush=True)
