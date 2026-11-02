"""Encode one orientation on demand to shorten transpose lifetime."""
import torch
import triton
import triton.language as tl
from .pair import _encode

@triton.jit
def _single(W,D,S,P,M:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,KP:tl.constexpr,
            T:tl.constexpr,R:tl.constexpr,C:tl.constexpr):
    r=tl.program_id(0)*R
    c=tl.program_id(1)*C
    rows=r+tl.arange(0,R); cols=c+tl.arange(0,C)
    vals=tl.load(W+rows[:,None]*K+cols[None,:],(rows[:,None]<M)&(cols[None,:]<K),other=0.)
    vals=vals.to(tl.float32).to(tl.bfloat16).to(tl.float32)
    inv=tl.div_rn(1.,tl.load(P))
    if T:
        _encode(tl.trans(vals),D,S,inv,c,r,MP,C,R)
    else:
        _encode(vals,D,S,inv,r,c,KP,R,C)

def single(w,pts,transpose):
    m,k=w.shape
    mp,kp=triton.cdiv(m,128)*128,triton.cdiv(k,128)*128
    a,b=(kp,mp) if transpose else (mp,kp)
    data=torch.empty((a,b//2),device=w.device,dtype=torch.uint8)
    scales=torch.empty((a//128*32,b//64*16),device=w.device,dtype=torch.uint8)
    _single[(mp//64,kp//128)](w,data,scales,pts,m,k,mp,kp,transpose,64,128,num_warps=4,enable_fp_fusion=False)
    return data,scales
