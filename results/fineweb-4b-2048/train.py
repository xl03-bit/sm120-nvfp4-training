"""FineWeb 4.06B full-parameter training on one 16 GB SM120 GPU."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nvfp4_training import prepare_training
from nvfp4_training._backend import nvfp4_fast_v2 as rng
from nvfp4_training._backend.engine6 import AdamW6
from nvfp4_training._backend.step6 import StaticStep
from nvfp4_training._workloads.lm import Block, LanguageModel
from nvfp4_training._workloads.token_loss import token_cross_entropy

class MatchedBatches:
    def __init__(self,args):
        path = Path(os.environ["NVFP4_FINEWEB"])
        with path.open('rb') as f:
            digest = hashlib.sha256()
            for chunk in iter(lambda:f.read(1<<20),b''):
                digest.update(chunk)
        if digest.hexdigest()!='771fa4a99b9fe0946ffb6e848b4ba5c6a9b0fe87860ebf03bc2c1c7e45f8178e':
            raise RuntimeError('FineWeb shard checksum mismatch')
        data=np.memmap(path,mode='r',offset=1024,dtype='<u2')
        assert len(data)==100_000_000
        boundary=95_000_000+int(np.flatnonzero(data[95_000_000:]==50256)[0])
        self.provenance = dict(sha256=digest.hexdigest(),split_token=boundary,windows_sha256={})
        def make(lo,hi,count,seed):
            rng=np.random.default_rng(seed)
            starts=rng.integers(lo,hi-args.seq,size=(count,args.batch))
            self.provenance['windows_sha256'][str(seed)] = hashlib.sha256(starts.astype('<i8').tobytes()).hexdigest()
            ids=starts[:,:,None]+np.arange(args.seq+1)
            tokens=torch.tensor(data[ids].astype('int64'),device='cuda')
            return [(row[:,:-1].contiguous(),row[:,1:].contiguous()) for row in tokens]
        self.train=make(0,boundary,args.warmup+args.steps*args.rounds,701+args.seed-1234)
        self.validation=make(boundary,len(data),args.val_batches,702)

def model_parts(width,depth,seq,gen):
    # Follow LanguageModel.forward, Block, weight tying, dimensions and initialization.
    # Each source FP32 block is short-lived; no parameters or states are offloaded during training.
    vocab=50257
    shell=LanguageModel.__new__(LanguageModel)
    nn.Module.__init__(shell)
    with torch.device('meta'):
        shell.tok=nn.Embedding(vocab,width)
        shell.pos=nn.Embedding(max(512,seq),width)
        shell.blocks=nn.Sequential()
        shell.ln=nn.LayerNorm(width)
        shell.head=nn.Linear(width,vocab,bias=False)
    for module in (shell.tok,shell.pos,shell.ln):module.to_empty(device='cuda')
    shell.head.weight=shell.tok.weight
    with torch.no_grad():
        shell.tok.weight.normal_(0,.02,generator=gen)
        shell.pos.weight.normal_(0,.02,generator=gen)
        shell.ln.weight.fill_(1);shell.ln.bias.zero_()
    def next_block():
        with torch.device('meta'):block=Block(width,width//64)
        block.to_empty(device='cuda')
        with torch.no_grad():
            for layer in (block.norm1,block.norm2):
                layer.weight.fill_(1);layer.bias.zero_()
            for layer in (block.qkv,block.proj,block.up,block.down):
                std=.02/math.sqrt(2*depth) if layer in (block.proj,block.down) else .02
                layer.weight.normal_(0,std,generator=gen);layer.bias.zero_()
        return block
    return shell,next_block


CONFIG = dict(parameters=4063797760, width=2560, layers=50, heads=40,
              profile="experimental_fp8_master", remat=True, seed=1234,
              batch=2, seq=512, steps=2048, val_batches=64, data_schedule=8192)
GIB = 2**30


def build():
    gen = torch.Generator(device="cuda").manual_seed(CONFIG["seed"])
    model, next_block = model_parts(CONFIG["width"], CONFIG["layers"], CONFIG["seq"], gen)
    options = dict(precision="fp4", profile=CONFIG["profile"], remat=True)
    model, outer = prepare_training(model, **options)
    cfg = outer.param_groups[0]
    states = dict(outer.state)
    for i in range(CONFIG["layers"]):
        block, local = prepare_training(next_block(), **options)
        del block.nvfp4_optimizer
        model.blocks.add_module(str(i), block)
        states.update(local.state)
    opt = AdamW6(model.parameters(), cfg["format"], lr=cfg["lr"], tiles=cfg["tiles"], warps=cfg["warps"])
    opt.param_groups[0].update({k: v for k, v in cfg.items() if k != "params"})
    opt.state.update(states)
    model.nvfp4_optimizer = opt
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == CONFIG["parameters"]
    assert all(p.is_cuda and p.requires_grad for p in model.parameters())
    assert model.tok.weight is model.head.weight
    return model, opt


def memory():
    free, total = torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                device_used=total-free)


def save_report(path, report):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, allow_nan=False, separators=(",", ":")), encoding="utf8")
    tmp.replace(path)


@torch.no_grad()
def evaluate(model, batches):
    scores = []
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        scores.append(F.cross_entropy(logits.flatten(0, 1).float(), y.flatten()).item())
        del logits
    if not all(map(math.isfinite, scores)):
        raise FloatingPointError("Nonfinite held-out cross-entropy")
    return dict(ce=sum(scores)/len(scores), per_batch=scores)


def checkpoint(path, model, opt, step, report):
    # StaticStep.close has committed step counters and the scale of each updated weight.
    assert len(opt.state) == len(list(model.parameters()))
    assert all(s["step"] == step for s in opt.state.values())
    scales = {n:p._v4_scale[3] for n,p in model.named_parameters() if hasattr(p, "_v4_scale")}
    snap = dict(config=CONFIG, step=step, model=model.state_dict(), optimizer=opt.state_dict(),
                scales=scales, sr=rng._SEED, cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(), report=report)
    tmp = path.with_suffix(".partial")
    try:
        torch.save(snap, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def restore(path, model, opt):
    snap = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if snap["config"] != CONFIG:
        raise ValueError("Checkpoint configuration differs")
    model.load_state_dict(snap["model"])
    opt.load_state_dict(snap["optimizer"])
    for name,p in model.named_parameters():
        if name in snap["scales"]:
            p._v4_scale = (p._version,p.data_ptr(),torch.cuda.current_stream().cuda_stream,snap["scales"][name].cuda())
    rng.reset_stats(snap["sr"])
    torch.set_rng_state(snap["cpu_rng"])
    torch.cuda.set_rng_state(snap["cuda_rng"])
    return snap["step"], snap["report"]


def run(output, resume):
    output.mkdir(parents=True, exist_ok=True)
    record = output/"metrics.json"
    ckpt = output/"checkpoint.pt"
    if not resume and (record.exists() or ckpt.exists()):
        raise FileExistsError("Run already exists; use --resume or a new output directory")
    if shutil.disk_usage(output).free < (11 if resume else 20)*GIB:
        raise OSError("Insufficient disk space for atomic checkpoint replacement")
    torch.manual_seed(CONFIG["seed"])
    free, total = torch.cuda.mem_get_info()
    if free < 14.4*GIB:
        raise RuntimeError(f"4.06B run needs 14.4 GiB free at startup; found {free/GIB:.3f}")
    if torch.cuda.get_device_capability() != (12,0) or str(torch.__version__) != "2.11.0+cu130":
        raise RuntimeError("Expected the archived SM120/cu130 environment")
    torch.cuda.reset_peak_memory_stats()
    batches = MatchedBatches(SimpleNamespace(batch=2, seq=512, steps=8192, warmup=0, rounds=1, seed=1234, val_batches=64))
    meta = dict(config=CONFIG, torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                initial_free_bytes=free, device_total_bytes=total, provenance=batches.provenance, status="running",
                initialization={}, captures=[], steps=[], validation=[])
    model,opt = build()
    fp8_weights=[p for p in model.parameters() if p.dtype==torch.float8_e4m3fn]
    initial_units=[p._v6_unit[0].clone() for p in fp8_weights]
    meta["master_scale"] = "adaptive power-of-two units; 512 consecutive FP8 weights per unit"
    meta["scale_blocks"] = sum(p._v6_unit.numel() for p in fp8_weights)
    meta["optimizer"] = {k:opt.param_groups[0][k] for k in ("lr","betas","eps","weight_decay","format","tiles","warps")}
    torch.cuda.synchronize()
    meta["initialization"] = memory()
    rng.reset_stats(0)
    start = 0
    if resume:
        start, saved = restore(ckpt,model,opt)
        if saved["provenance"] != meta["provenance"] or len(saved["steps"]) != start:
            raise RuntimeError("Checkpoint token stream or metrics mismatch")
        meta = saved
    def objective(x,y):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            return token_cross_entropy(model(x), y)
    def validate(step):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            bad = torch.stack([((p.view(torch.uint8)&127)==127).any() for p in fp8_weights]).any().item()
            grown = torch.stack([(p._v6_unit>initial).sum() for p,initial in zip(fp8_weights,initial_units)]).sum().item()
        if bad:
            raise FloatingPointError(f"Nonfinite FP8 master at validation update {step}")
        result = evaluate(model,batches.validation)
        meta["validation"].append(dict(step=step,grown_scale_blocks=grown,nonfinite_fp8_codes=0,**result,**memory()))
        print(f"VALIDATION step={step} ce={result['ce']:.5f} grown_blocks={grown}",flush=True)
    try:
        for base in range(start, CONFIG["steps"], 512):
            end = min(base+512, CONFIG["steps"])
            torch.cuda.reset_peak_memory_stats()
            native = 0
            original = torch._scaled_mm
            def traced(a,b,*args,**kwargs):
                nonlocal native
                native += int(a.dtype == b.dtype == torch.float4_e2m1fn_x2)
                return original(a,b,*args,**kwargs)
            t0 = time.perf_counter()
            with patch.object(torch, "_scaled_mm",traced):
                graph = StaticStep(model,opt,objective,*batches.train[base],end-base,stream_updates=True,from_start=base==0)
            native += getattr(opt,"fused_mma_calls",0)
            with graph:
                assert native==603 and graph.sr_calls==402, (native,graph.sr_calls)
                torch.cuda.synchronize()
                meta["captures"].append(dict(step=base,native_gemms=native,sr_calls=graph.sr_calls,
                                             seconds=time.perf_counter()-t0,**memory()))
                if base==0: validate(0)
                save_report(record,meta)
                for index in range(base,end):
                    torch.cuda.reset_peak_memory_stats()
                    tic = time.perf_counter()
                    value = graph.step(*batches.train[index]).item()
                    elapsed = time.perf_counter()-tic
                    step = index+1
                    if not math.isfinite(value):
                        raise FloatingPointError(f"Nonfinite training loss at update {step}: {value}")
                    meta["steps"].append(dict(step=step,ce=value,seconds=elapsed,**memory()))
                    if step in (5,64) or step%128==0: validate(step)
                    if step<=5 or step%32==0:
                        save_report(record,meta)
                        print(f"TRAIN step={step} loss={value:.5f} step_ms={elapsed*1000:.1f}",flush=True)
            del graph
            gc.collect()
            torch.cuda.empty_cache()
            assert rng._SEED == end*402, (end,rng._SEED)
            print(f"CHECKPOINT saving update {end}",flush=True)
            checkpoint(ckpt,model,opt,end,meta)
            print(f"CHECKPOINT committed update {end}",flush=True)
        assert len(meta["steps"])==CONFIG["steps"] and meta["validation"][-1]["step"]==CONFIG["steps"]
        from nvfp4_training import check_finite
        check_finite(model)
        meta["status"]="complete"
        meta["completed_updates"]=CONFIG["steps"]
        print("COMPLETE 2048 full-parameter updates",flush=True)
    except Exception as exc:
        meta["status"]="failed"
        meta["error"]=repr(exc)
        raise
    finally:
        save_report(record,meta)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    run(args.output,args.resume)
