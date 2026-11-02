"""E4M3 projection weights with a power-of-two unit per 512 consecutive weights."""
import math
import torch
import triton
import triton.language as tl
from .pair import _encode

def unit_for(p):
    lo,hi=torch.aminmax(p.detach())
    maximum=max(abs(lo.item()),abs(hi.item()))
    if not math.isfinite(maximum):raise ValueError('Nonfinite initial weight')
    return 2.**math.ceil(math.log2(max(maximum,1e-20)/48.))

@triton.jit
def _decode(P,U,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    tl.store(Y+i,(tl.load(P+i,i<N,other=0.).to(tl.float32)*tl.load(U+tl.program_id(0))).to(tl.bfloat16),i<N)

def decoded(p):
    if p.dtype!=torch.float8_e4m3fn:return p.bfloat16()
    y=torch.empty_like(p,dtype=torch.bfloat16)
    _decode[(triton.cdiv(p.numel(),512),)](p,p._v6_unit,y,p.numel(),512)
    return y

@triton.jit
def round_e4m3(x,bits):
    raw=x.to(tl.uint32,bitcast=True)
    normal=((raw+(bits&0xfffff))&0xfff00000).to(tl.float32,bitcast=True)
    mag=tl.abs(x)*512.
    low=tl.floor(mag)
    sub=(low+((bits&0xfffff).to(tl.float32)/1048576.<mag-low).to(tl.float32))/512.
    val=tl.where(tl.abs(x)<.015625,tl.where(x<0.,-sub,sub),normal)
    # Preserve nonfinite updates for explicit tensor health checks.
    return tl.where(tl.abs(x)<=448.,val,float('nan')).to(tl.float8e4nv)

@triton.jit
def _scaled_pair(W,U,D,S,DT,ST,P,M:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,KP:tl.constexpr,
                 ROW:tl.constexpr,TRANS:tl.constexpr,R:tl.constexpr,C:tl.constexpr):
    r=tl.program_id(0)*R;c=tl.program_id(1)*C
    row=r+tl.arange(0,R);col=c+tl.arange(0,C)
    offset=row[:,None].to(tl.int64)*K+col[None,:]
    valid=(row[:,None]<M)&(col[None,:]<K)
    v=tl.load(W+offset,valid,other=0.).to(tl.float32)
    unit=tl.load(U+offset//512,valid,other=1.)
    v=(v*unit).to(tl.bfloat16).to(tl.float32)
    inv=tl.div_rn(1.,tl.load(P))
    if ROW:_encode(v,D,S,inv,r,c,KP,R,C)
    if TRANS:_encode(tl.trans(v),DT,ST,inv,c,r,MP,C,R)

def encode(p,scale,*,row=True,trans=True):
    m,k=p.shape;mp=triton.cdiv(m,128)*128;kp=triton.cdiv(k,128)*128
    def buf(shape):return torch.empty(shape,device=p.device,dtype=torch.uint8)
    d,s=(buf((mp,kp//2)),buf((mp//128*32,kp//64*16))) if row else (p,p)
    dt,st=(buf((kp,mp//2)),buf((kp//128*32,mp//64*16))) if trans else (p,p)
    _scaled_pair[(mp//64,kp//128)](p,p._v6_unit,d,s,dt,st,scale,m,k,mp,kp,row,trans,64,128,num_warps=4,enable_fp_fusion=False)
    return d,s,scale,dt,st
