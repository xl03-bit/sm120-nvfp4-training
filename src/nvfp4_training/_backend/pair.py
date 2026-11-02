"""Encode both contraction directions of an FP32 weight from one tiled load."""
import torch
import triton
import triton.language as tl

@triton.jit
def _encode(tile, data, scale, inv, r0, c0, K:tl.constexpr,
            R:tl.constexpr,C:tl.constexpr):
    rows = r0 + tl.arange(0,R)
    cols = c0//16 + tl.arange(0,C//16)
    groups = tl.reshape(tile,(R,C//16,16))
    bound = tl.max(tl.abs(groups),2)
    encoded = tl.minimum(448.,tl.maximum(.015625,tl.div_rn(bound,6.)*inv)).to(tl.float8e4nv)
    offset = ((rows[:,None]//128)*(K//64)+cols[None,:]//4)*512
    offset += (rows[:,None]%32)*16+((rows[:,None]%128)//32)*4+cols[None,:]%4
    tl.store(scale+offset,encoded.to(tl.uint8,bitcast=True))
    value = groups*tl.div_rn(inv,encoded.to(tl.float32))[:,:,None]
    value = tl.minimum(6.,tl.maximum(-6.,value))
    lo,hi = tl.split(tl.reshape(value,(R,C//2,2)))
    packed = tl.inline_asm_elementwise(
        asm="{ .reg .b8 byte; cvt.rn.satfinite.e2m1x2.f32 byte,$2,$1; cvt.u32.u8 $0,byte; }",
        constraints="=r,f,f",args=[lo,hi],dtype=tl.int32,is_pure=True,pack=1)
    tl.store(data+rows[:,None]*(K//2)+c0//2+tl.arange(0,C//2)[None,:],packed.to(tl.uint8))

@triton.jit
def _pair(W,D,S,DT,ST,P,M:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,KP:tl.constexpr,
          R:tl.constexpr,C:tl.constexpr):
    r0 = tl.program_id(0)*R
    c0 = tl.program_id(1)*C
    row = r0+tl.arange(0,R)
    col = c0+tl.arange(0,C)
    tile = tl.load(W+row[:,None]*K+col[None,:],(row[:,None]<M)&(col[None,:]<K),other=0.)
    tile = tile.to(tl.bfloat16).to(tl.float32)
    inv = tl.div_rn(1.,tl.load(P))
    _encode(tile,D,S,inv,r0,c0,KP,R,C)
    _encode(tl.trans(tile),DT,ST,inv,c0,r0,MP,C,R)

def pair(w,pts,cfg):
    m,k=w.shape
    mp,kp=triton.cdiv(m,128)*128,triton.cdiv(k,128)*128
    r,c,warps=cfg
    if mp%r or kp%c or mp*kp>=2**31:
        raise ValueError("Unsupported tile geometry or 32-bit index range")
    def buf(shape):
        return torch.empty(shape,device=w.device,dtype=torch.uint8)
    d,s=buf((mp,kp//2)),buf((mp//128*32,kp//64*16))
    dt,st=buf((kp,mp//2)),buf((kp//128*32,mp//64*16))
    _pair[(mp//r,kp//c)](w,d,s,dt,st,pts,m,k,mp,kp,r,c,num_warps=warps,enable_fp_fusion=False)
    return d,s,pts,dt,st
