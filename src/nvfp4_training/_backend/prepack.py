"""Validate optimizer-generated NVFP4 encodings against the exact current tensor scale."""
import torch
import triton
import triton.language as tl
from .pair import _encode

@triton.jit
def _repair(W,D,S,DT,ST,SCALE,TAG,HITS,M:tl.constexpr,K:tl.constexpr,R:tl.constexpr,C:tl.constexpr):
    pid=tl.program_id(0)
    scale=tl.load(SCALE)
    stale=scale!=tl.load(TAG)
    if pid==0:
        tl.store(HITS,tl.load(HITS)+1)
        tl.store(HITS+1,tl.load(HITS+1)+stale.to(tl.int32))
    if stale:
        inv=tl.div_rn(1.,scale)
        for tile in range(pid,(M//R)*(K//C),tl.num_programs(0)):
            r=tile//(K//C)*R
            c=tile%(K//C)*C
            off=(r+tl.arange(0,R))[:,None]*K+(c+tl.arange(0,C))[None,:]
            vals=tl.load(W+off).to(tl.float32).to(tl.bfloat16).to(tl.float32)
            _encode(vals,D,S,inv,r,c,K,R,C)
            _encode(tl.trans(vals),DT,ST,inv,c,r,M,C,R)

def buffers(p):
    m,k=p.shape
    def alloc(shape): return torch.empty(shape,device=p.device,dtype=torch.uint8)
    return (alloc((m,k//2)),alloc((m//128*32,k//64*16)),alloc((k,m//2)),
            alloc((k//128*32,m//64*16)),torch.full((),float('nan'),device=p.device),
            torch.zeros(2,device=p.device,dtype=torch.int32))

def read_pair(p,pts,cache):
    d,s,dt,st,tag,hits=cache
    m,k=p.shape
    grid=min((m//64)*(k//128),144)
    _repair[(grid,)](p,d,s,dt,st,pts,tag,hits,m,k,64,128,num_warps=4,enable_fp_fusion=False)
    return d,s,pts,dt,st
