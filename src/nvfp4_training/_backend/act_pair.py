"""Produce row FP4 and the separately RHT-encoded transpose from one input tile."""
import torch
import triton
import triton.language as tl
from .nvfp4_fast_v2 import Packed
from . import nvfp4_fast_v2 as rng
from .pair import _encode

@triton.jit
def _absmax(X,P,N:tl.constexpr,C:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    off=i if S0==C and S1==1 else (i//C)*S0+(i%C)*S1
    v=tl.load(X+off,i<N,other=0.).to(tl.bfloat16).to(tl.float32)
    tl.store(P+tl.program_id(0),tl.max(tl.abs(v),0))

@triton.jit
def _scale(P,S,Q,N:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B)
    hi=tl.maximum(1.e-12,tl.max(tl.load(P+i,i<N,other=0.),0))
    tl.store(S,hi*(1./2688))
    tl.store(Q,(hi*4.)*(1./2688))

@triton.jit
def _sr(z,D,S,inv,seed,r0,c0,PAD:tl.constexpr,R:tl.constexpr,C:tl.constexpr):
    row=r0+tl.arange(0,R)
    col=c0+tl.arange(0,C)
    groupcol=c0//16+tl.arange(0,C//16)
    z=tl.reshape(z,(R,C//16,16))
    maximum=tl.max(tl.abs(z),2)
    encoded=tl.minimum(448.,tl.maximum(.015625,tl.div_rn(maximum,6.)*inv)).to(tl.float8e4nv)
    idx=((row[:,None]//128)*(PAD//64)+groupcol[None,:]//4)*512
    idx+=(row[:,None]%32)*16+((row[:,None]%128)//32)*4+groupcol[None,:]%4
    tl.store(S+idx,encoded.to(tl.uint8,bitcast=True))
    z=tl.reshape(z*tl.div_rn(inv,encoded.to(tl.float32))[:,:,None],(R,C))
    mag=tl.minimum(6.,tl.abs(z))
    h=seed*0x9e3779b9+(row[:,None]*PAD+col[None,:]).to(tl.uint32)+0x85ebca6b
    h=(h^(h>>16))*0x7feb352d
    h=(h^(h>>15))*0x846ca68b
    u=((h^(h>>16))>>8).to(tl.float32)*(1./16777216)
    # Subtract the lower FP4 value before dividing. In the [2,4)
    # interval adding the code bias first can lose one bit at a tie.
    lo=tl.where(mag<2.,tl.floor(mag*2.)*.5,tl.where(mag<4.,tl.floor(mag),tl.floor(mag*.5)*2.))
    base=tl.where(mag<2.,(lo*2.).to(tl.int32),tl.where(mag<4.,lo.to(tl.int32)+2,(lo*.5).to(tl.int32)+4))
    prob=(mag-lo)*tl.where(mag<2.,2.,tl.where(mag<4.,1.,.5))
    code=base+(u<prob).to(tl.int32)
    code=code|tl.where(z<0.,8,0)
    a,b=tl.split(tl.reshape(code,(R,C//2,2)))
    tl.store(D+row[:,None]*(PAD//2)+c0//2+tl.arange(0,C//2)[None,:],(a|(b<<4)).to(tl.uint8))

@triton.jit
def _dual(X,D,S,Q,DT,ST,QT,SIGN,CLOCK,OFF,M:tl.constexpr,N:tl.constexpr,
          MP:tl.constexpr,NP:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,
          STOCH:tl.constexpr,DYNAMIC:tl.constexpr,R:tl.constexpr,C:tl.constexpr):
    r0=tl.program_id(0)*R;c0=tl.program_id(1)*C
    row=r0+tl.arange(0,R);col=c0+tl.arange(0,C)
    x=tl.load(X+row[:,None]*S0+col[None,:]*S1,(row[:,None]<M)&(col[None,:]<N),other=0.).to(tl.bfloat16).to(tl.float32)
    inv=tl.div_rn(1.,tl.load(Q))
    if STOCH:
        seed=OFF.to(tl.uint32)
        if DYNAMIC:seed+=tl.load(CLOCK).to(tl.uint32)
        _sr(x,D,S,inv,seed,r0,c0,NP,R,C)
    else:_encode(x,D,S,inv,r0,c0,NP,R,C)
    xt=tl.trans(x*tl.load(SIGN+row%16)[:,None])
    for j in tl.static_range(4):
        ix=tl.arange(0,R)
        other=tl.gather(xt,tl.broadcast_to((ix^(1<<j))[None,:],(C,R)),axis=1)
        xt=tl.where((ix&(1<<j))[None,:]!=0,other-xt,xt+other)
    xt*=.25
    invt=tl.div_rn(1.,tl.load(QT))
    if STOCH:_sr(xt,DT,ST,invt,seed+1,c0,r0,MP,C,R)
    else:_encode(xt,DT,ST,invt,c0,r0,MP,C,R)

def supported(x):
    m,n=x.shape
    return (m>0 and n>0 and min(x.stride())>=0 and
            triton.cdiv(m,128)*triton.cdiv(n,128)*16384<2**31 and
            (m-1)*x.stride(0)+(n-1)*x.stride(1)<2**31)

def dual(x,signs,*,sr=False,clock=None,offset=None,cfg=(32,64,4)):
    m,n=x.shape;mp=triton.cdiv(m,128)*128;np=triton.cdiv(n,128)*128
    if not supported(x):raise ValueError('Unsupported activation pair size/strides')
    r,c,warps=cfg
    buf=lambda shape:torch.empty(shape,device=x.device,dtype=torch.uint8)
    d,s=buf((mp,np//2)),buf((mp//128*32,np//64*16))
    dt,st=buf((np,mp//2)),buf((np//128*32,mp//64*16))
    q,qt=[torch.empty((),device=x.device,dtype=torch.float32) for _ in range(2)]
    blocks=triton.cdiv(x.numel(),2048)
    partial=torch.empty(blocks,device=x.device,dtype=torch.float32)
    _absmax[(blocks,)](x,partial,x.numel(),n,*x.stride(),2048)
    if blocks>32768:partial=partial.amax().reshape(1);blocks=1
    _scale[(1,)](partial,q,qt,blocks,triton.next_power_of_2(blocks))
    if offset is None:
        offset=0
        if sr:
            if clock is not None:offset=clock.sr_calls;clock.sr_calls+=2
            else:offset=rng._SEED;rng._SEED=(offset+2)&0xffffffff
    _dual[(mp//r,np//c)](x,d,s,q,dt,st,qt,signs,clock.seed if clock is not None else q,offset,
                          m,n,mp,np,*x.stride(),sr,clock is not None,r,c,num_warps=warps,enable_fp_fusion=False)
    rng._STATS['quant']+=2
    rng._STATS['rht_quant']+=1
    rng._STATS['sr_quant']+=int(sr)*2
    return Packed(d,s,q,x.dtype),Packed(dt,st,qt,x.dtype)
