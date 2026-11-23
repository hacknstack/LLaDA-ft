"""One model per GPU, with bounded lookahead and cross-case inference batching."""
import queue
import threading
import time
from collections import Counter
from concurrent.futures import Future
from types import SimpleNamespace

import torch


class LogitSlice:
    def __init__(self,positions,values,event=None):
        self.positions,self.values,self.event=positions,values,event

    def __getitem__(self,index):
        batch,positions,vocab=index
        assert batch==0 and vocab==slice(None) and positions.tolist()==self.positions
        stream=torch.cuda.current_stream()
        if self.event is not None:stream.wait_event(self.event)
        values=self.values.to('cuda:0')
        values.record_stream(stream)
        return values


class WithinCase(torch.nn.Module):
    """The already-tested baseline: batch pending states from one case only."""
    def __init__(self,base):
        super().__init__();self.base=base;self.cache={};self.pending=[];self.offset=0
        self.inputs=None;self.rows=self.calls=self.checks=0

    def prefetch(self,inputs,attention_mask):
        assert attention_mask is None
        self.cache.clear();unique={}
        for i,row in enumerate(inputs.detach().cpu()):unique.setdefault(row.numpy().tobytes(),i)
        self.pending=list(unique.items());self.inputs=inputs;self.offset=0

    def load_next(self):
        self.cache.clear();remaining=len(self.pending)-self.offset
        count=16 if remaining>=16 else (2 if remaining>=2 else 1)
        chosen=self.pending[self.offset:self.offset+count]
        indices=torch.tensor([i for _,i in chosen],device=self.inputs.device)
        outputs=self.base(self.inputs.index_select(0,indices),attention_mask=None)
        self.calls+=1;self.rows+=count
        for local,(key,i) in enumerate(chosen):
            positions=torch.nonzero(self.inputs[i]==126336,as_tuple=False).squeeze(-1)
            values=outputs.logits[local,positions,:].detach().cpu()
            if (self.rows-count+local)%97==0:
                direct=self.base(self.inputs[i:i+1],attention_mask=None).logits[0,positions,:].detach().cpu()
                assert torch.equal(values,direct),'Within-case native logits changed'
                self.checks+=1
            self.cache[key]=(positions.tolist(),values)
        del outputs
        self.offset+=count

    def forward(self,ids,attention_mask=None):
        key=ids[0].detach().cpu().numpy().tobytes()
        while key not in self.cache and self.offset<len(self.pending):self.load_next()
        if key in self.cache:
            positions,values=self.cache.pop(key)
            return SimpleNamespace(logits=LogitSlice(positions,values))
        self.calls+=1;self.rows+=1
        return self.base(ids,attention_mask=attention_mask)


class InferenceBroker:
    def __init__(self,model,max_wait_ms=1.,independent_stream=True,bulk=False):
        self.model=model;self.queue=queue.Queue();self.max_wait=max_wait_ms/1000
        self.independent_stream=independent_stream
        self.bulk=bulk
        self.rows=self.calls=self.checks=self.requests=self.deduplicated=0
        self.batches=Counter();self.failure=None
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def submit(self,ids,key):
        future=Future()
        if self.failure is not None:
            future.set_exception(self.failure)
            return future
        clone=ids.clone();event=torch.cuda.Event();event.record(torch.cuda.current_stream())
        self.queue.put((key,clone,event,future))
        return future

    def submit_many(self,inputs,chosen):
        if self.failure is not None:raise self.failure
        indices=torch.tensor([i for _,i in chosen],device=inputs.device)
        clone=inputs.index_select(0,indices)
        event=torch.cuda.Event();event.record(torch.cuda.current_stream())
        output={}
        for local,(key,_) in enumerate(chosen):
            future=Future();output[key]=future
            self.queue.put((key,clone[local:local+1],event,future))
        return output

    def run(self):
        stream=torch.cuda.Stream() if self.independent_stream else torch.cuda.default_stream()
        try:
            with torch.inference_mode(),torch.cuda.stream(stream),torch.autocast('cuda',enabled=False):
                stop=False
                while not stop:
                    first=self.queue.get()
                    if first is None:break
                    requests=[first];deadline=time.perf_counter()+self.max_wait
                    while len(requests)<16:
                        try:
                            value=self.queue.get(timeout=max(0,deadline-time.perf_counter()))
                            if value is None:stop=True;break
                            requests.append(value)
                        except queue.Empty:break
                    grouped={}
                    for request in requests:grouped.setdefault(request[0],[]).append(request)
                    self.requests+=len(requests);self.deduplicated+=len(requests)-len(grouped)
                    items=list(grouped.values());offset=0
                    while offset<len(items):
                        remaining=len(items)-offset
                        count=16 if remaining>=16 else (2 if remaining>=2 else 1)
                        chunk=items[offset:offset+count]
                        for duplicates in chunk:stream.wait_event(duplicates[0][2])
                        ids=torch.cat([duplicates[0][1] for duplicates in chunk],dim=0)
                        for duplicates in chunk:duplicates[0][1].record_stream(stream)
                        outputs=self.model(ids,attention_mask=None)
                        self.calls+=1;self.batches[count]+=1
                        for local,duplicates in enumerate(chunk):
                            positions=torch.nonzero(ids[local]==126336,as_tuple=False).squeeze(-1)
                            values=outputs.logits[local,positions,:].contiguous()
                            if self.rows%97==0:
                                direct=self.model(ids[local:local+1],attention_mask=None).logits[0,positions,:]
                                assert torch.equal(values,direct),'Cross-case native logits changed'
                                self.checks+=1
                            self.rows+=1
                            event=torch.cuda.Event();event.record(stream)
                            result=LogitSlice(positions.tolist(),values,event)
                            for request in duplicates:request[3].set_result(result)
                        del outputs,ids
                        offset+=count
        except BaseException as error:
            self.failure=error
            for request in locals().get('requests',[]):
                if not request[3].done():request[3].set_exception(error)
            while True:
                try:
                    request=self.queue.get_nowait()
                    if request is not None and not request[3].done():request[3].set_exception(error)
                except queue.Empty:break

    def stop(self):
        self.queue.put(None);self.thread.join(timeout=60)
        if self.failure is not None:raise self.failure
        assert not self.thread.is_alive()

    def stats(self):
        return dict(model_rows=self.rows,model_calls=self.calls,direct_logit_checks=self.checks,
                    model_batch_counts=dict(self.batches),requests=self.requests,deduplicated=self.deduplicated)


class CrossCase(torch.nn.Module):
    """Each algorithm context owns its RNG/state; model weights belong to the broker."""
    def __init__(self,broker,lookahead=16):
        super().__init__()
        self.register_parameter('device_anchor',torch.nn.Parameter(torch.zeros((),device='cuda'),requires_grad=False))
        self.broker=broker;self.lookahead=lookahead;self.cache={};self.inputs=None
        self.pending=[];self.offset=0
        self.eval()

    def prefetch(self,inputs,attention_mask):
        assert attention_mask is None
        self.cache.clear();unique={}
        for i,row in enumerate(inputs.detach().cpu()):unique.setdefault(row.numpy().tobytes(),i)
        self.inputs=inputs;self.pending=list(unique.items());self.offset=0

    def load_next(self):
        self.cache.clear()
        chosen=self.pending[self.offset:self.offset+self.lookahead]
        if self.broker.bulk:self.cache.update(self.broker.submit_many(self.inputs,chosen))
        else:
            for key,i in chosen:self.cache[key]=self.broker.submit(self.inputs[i:i+1],key)
        self.offset+=len(chosen)

    def forward(self,ids,attention_mask=None):
        assert attention_mask is None
        key=ids[0].detach().cpu().numpy().tobytes()
        while key not in self.cache and self.offset<len(self.pending):self.load_next()
        future=self.cache.pop(key,None)
        if future is None:future=self.broker.submit(ids,key)
        result=future.result(timeout=120)
        return SimpleNamespace(logits=result)
