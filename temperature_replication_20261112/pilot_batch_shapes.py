"""Measure transformer batching speed and whether native BF16 logits change."""
import json
import argparse
import os
import time
from pathlib import Path

ROOT=Path('/workspace/temperature-study')


def main():
    if (ROOT/'results/batch_shape_audit.json').exists():
        print('BATCH_SHAPE_AUDIT_ALREADY_COMPLETE',flush=True)
        return
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=4)
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=json.loads((ROOT/'mig_uuids.json').read_text())[args.gpu]
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from transformers import AutoModel
    import probabilistic_extraction as pe
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    data=json.loads((ROOT/'pilot_inputs.json').read_text())
    model=AutoModel.from_pretrained(ROOT/'models/llada-base',trust_remote_code=True,local_files_only=True,
          torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':0}).eval()
    model.config.use_cache=False
    inputs=[]
    for row in data['selected']['validation'][::3]:
        ids=data['tokens'][row['document']+':'+str(row['window_number'])]
        for revealed in (0,25,40,45,48,49):
            x=torch.tensor(ids,device='cuda')
            mask=data['masks'][row['mask_id']-1][revealed:]
            x[torch.tensor(mask,device='cuda')-1]=126336
            inputs.append(x)
    references=[]
    full_times=[]
    selected_records=[]
    records=[]
    with torch.inference_mode():
        for x in inputs:
            torch.cuda.synchronize();start=time.perf_counter()
            output=model(x.unsqueeze(0),attention_mask=None)
            torch.cuda.synchronize();full_times.append(time.perf_counter()-start)
            references.append(output.logits[0].detach().cpu());del output
        projection=pe._random_remasking_projection_layer(model)
        for i,x in enumerate(inputs):
            positions=torch.nonzero(x==126336,as_tuple=False).squeeze(-1)
            torch.cuda.synchronize();start=time.perf_counter()
            output=pe._random_remasking_forward_logits(model,x.unsqueeze(0),None,positions.unsqueeze(0),projection)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            native=output[0].detach().cpu();reference=references[i][positions.cpu()]
            delta=(native.float()-reference.float()).abs()
            selected_records.append(dict(masked_positions=len(positions),full_seconds=full_times[i],
                selected_seconds=elapsed,bit_identical=torch.equal(native,reference),
                max_abs=float(delta.max()),fraction_changed=float((native!=reference).float().mean())))
            del output
        for batch in (1,2,4,8,16,32,64):
            try:
                batch_inputs=torch.stack([inputs[i%len(inputs)] for i in range(batch)])
                # Warm kernels for this shape, then measure repeated forwards.
                del_output=model(batch_inputs,attention_mask=None);del del_output
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                output=None
                for _ in range(2):
                    del output
                    output=model(batch_inputs,attention_mask=None)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)/2
                logits=output.logits
                differences=[]
                for i in range(batch):
                    native=logits[i].detach().cpu();reference=references[i%len(inputs)]
                    delta=(native.float()-reference.float()).abs()
                    differences.append(dict(bit_identical=torch.equal(native,reference),max_abs=float(delta.max()),
                                            fraction_changed=float((native!=reference).float().mean())))
                records.append(dict(batch=batch,seconds=elapsed,seconds_per_state=elapsed/batch,
                                    peak_gib=torch.cuda.max_memory_allocated()/2**30,differences=differences))
                del output,logits,batch_inputs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache();records.append(dict(batch=batch,error='CUDA_OOM'))
    (ROOT/'results/batch_shape_audit.json').write_text(json.dumps(dict(batching=records,
                      selected_projection=selected_records),indent=2))
    print('BATCH_SHAPE_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
