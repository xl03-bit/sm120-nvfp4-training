"""Time the published workloads using the public training API and recipe names."""
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import capture_step, check_finite, prepare_training
from ._runtime import load_runtime


def run(a):
    if torch.cuda.mem_get_info()[0]<12*2**30:
        raise RuntimeError("The reference benchmark needs at least 12 GiB free before starting")
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(a.seed)
    total=a.warmup+a.rounds*a.steps
    if a.task=="453m":
        from ._workloads.synthetic import Network
        model=Network(1536,16,12)
        gen=torch.Generator(device="cuda").manual_seed(2026)
        sample=tuple(torch.randn(a.batch,a.seq,1536,device='cuda',dtype=torch.bfloat16,generator=gen) for _ in range(2))
        def batch(i):return sample
        def compute(model,x,y):return F.mse_loss(model(x).float(),y.float())
    else:
        from ._workloads import lm
        model=lm.LanguageModel(a.seq)
        data=lm.TextBatches(a)
        from ._workloads.token_loss import token_cross_entropy as loss
        def batch(i):return data.train[i]
        def compute(model,x,y):return loss(model(x),y)
    model,opt=prepare_training(model,profile=a.profile,precision=a.precision,remat=a.remat)
    rng=load_runtime()[2]
    rng.reset_stats(0)
    times=[];losses=[]
    hits=0
    original=torch._scaled_mm
    def probe(x,w,*args,**kwargs):
        nonlocal hits
        if x.dtype==w.dtype==torch.float4_e2m1fn_x2:hits+=1
        return original(x,w,*args,**kwargs)
    with patch.object(torch,'_scaled_mm',probe):
        graph=capture_step(model,compute,*batch(0),steps=total)
    with graph:
        native=hits+getattr(opt,'fused_mma_calls',0)
        layers=64 if a.task=='453m' else 97
        if native!=(3*layers if a.precision=='fp4' else 0) or graph.sr_calls!=(2*layers if a.precision=='fp4' else 0):
            raise RuntimeError('Unexpected training GEMM or SR count')
        for i in range(a.warmup):graph.step(*batch(i))
        for r in range(a.rounds):
            torch.cuda.synchronize()
            start=time.perf_counter()
            for j in range(a.steps):
                last=graph.step(*batch(a.warmup+r*a.steps+j))
            torch.cuda.synchronize()
            times.append(1000*(time.perf_counter()-start)/a.steps)
            losses.append(last.item())
            if not torch.isfinite(last).item():raise FloatingPointError('Nonfinite training loss')
        peak=torch.cuda.max_memory_allocated()/2**30
        reserved=torch.cuda.max_memory_reserved()/2**30
    check_finite(model)
    if any(s['step']!=total for s in opt.state.values()):raise RuntimeError('Optimizer step count mismatch')
    ms=statistics.median(times)
    report=dict(config=vars(a),torch=torch.__version__,gpu=torch.cuda.get_device_name(),rounds_ms=times,
                step_ms=ms,tokens_s=1000*a.batch*a.seq/ms,train_peak_GiB=peak,
                train_reserved_GiB=reserved,native_gemms=native,sr_calls=graph.sr_calls,losses=losses)
    if a.task=='fineweb':report['data']=data.provenance
    if a.output:Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)
