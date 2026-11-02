"""Native NVFP4 Wgrad with an on-chip compact AdamW epilogue for aligned short K."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._layouts import BlockedLayout,SliceLayout
from .native_tile import grad_tile

@gluon.jit
def fused_update(A,B,AS,BS,SA,SB,P,M,V,MS,VS,TAIL,PART,CTRL,
                 N:gl.constexpr,T:gl.constexpr,lr,decay,b1,b2,eps,key,
                 MASTER:gl.constexpr,W:gl.constexpr,BLOCK:gl.constexpr):
    layout:gl.constexpr=BlockedLayout([1,4],[1,32],[W,1],[1,0])
    r0=gl.program_id(0)*16
    c0=gl.program_id(1)*BLOCK
    raw=grad_tile(A,B,AS,BS,r0,c0,T,BLOCK,W,layout).to(gl.bfloat16).to(gl.float32)
    alpha=(gl.load(SA)*gl.load(SB)).to(gl.bfloat16).to(gl.float32)
    g=(raw*alpha).to(gl.bfloat16).to(gl.float32)
    row=r0+gl.arange(0,16,layout=gl.constexpr(SliceLayout(1,layout)))
    col=c0+gl.arange(0,BLOCK,layout=gl.constexpr(SliceLayout(0,layout)))
    i=row[:,None]*N+col[None,:]
    group=row*(N//BLOCK)+gl.program_id(1)
    hp=gl.load(P+i)
    if MASTER==24:
        high=hp.to(gl.uint16,bitcast=True).to(gl.uint32)<<16
        tail=gl.load(TAIL+i).to(gl.int32)*256
        p=(high+tail.to(gl.uint32)).to(gl.float32,bitcast=True)
    else: p=hp.to(gl.float32)
    m=gl.load(M+i).to(gl.float32)*gl.load(MS+group)[:,None]
    v=gl.load(V+i).to(gl.float32)*gl.load(VS+group)[:,None]
    v=v*v
    m=b1*m+(1.-b1)*g
    v=b2*v+(1.-b2)*g*g
    root=gl.sqrt(v)
    c1=gl.load(CTRL);c2=gl.load(CTRL+1)
    t=gl.load(CTRL+3).to(gl.uint32,bitcast=True)
    rnd=i.to(gl.uint32)+t*0x9e3779b9+key.to(gl.uint32)
    rnd=(rnd^(rnd>>16))*0x7feb352d
    rnd=(rnd^(rnd>>15))*0x846ca68b
    rnd=rnd^(rnd>>16)
    decayed=gl.inline_asm_elementwise("mul.rn.f32 $0,$1,$2;",constraints="=f,f,f",args=[p,decay],dtype=gl.float32,is_pure=True,pack=1)
    updated=decayed-(lr*c1)*m/(root*c2+eps)
    rawbits=updated.to(gl.uint32,bitcast=True)
    if MASTER==24:
        hi=updated.to(gl.bfloat16)
        head=hi.to(gl.uint16,bitcast=True).to(gl.uint32)<<16
        d=(rawbits-head).to(gl.int32,bitcast=True)
        q=(d+127+((d>>8)&1))>>8
        gl.store(TAIL+i,gl.maximum(-128,gl.minimum(127,q)))
        gl.store(P+i,hi)
    else:
        normal=((rawbits+(rnd&8191))&0xffffe000).to(gl.float32,bitcast=True).to(gl.float16)
        mag=gl.abs(updated)*16777216.
        sub=gl.floor(mag)+(((rnd&65535).to(gl.float32)/65536.)<(mag-gl.floor(mag))).to(gl.float32)
        sbits=sub.to(gl.uint16)|((rawbits>>16)&32768).to(gl.uint16)
        half=gl.where(gl.abs(updated)<0.00006103515625,sbits.to(gl.float16,bitcast=True),normal)
        gl.store(P+i,half)
        hi=half.to(gl.float32).to(gl.bfloat16)
    ms=gl.maximum(gl.max(gl.abs(m),1)/127.,1.e-30)
    mc=gl.inline_asm_elementwise("cvt.rni.s32.f32 $0, $1;",constraints="=r,f",args=[m/ms[:,None]],dtype=gl.int32,is_pure=True,pack=1)
    gl.store(M+i,gl.minimum(127,gl.maximum(-127,mc)))
    gl.store(MS+group,ms)
    vs=gl.maximum(gl.max(root,1)/255.,1.e-30)
    z=gl.minimum(255.,root/vs[:,None])
    lo=gl.floor(z)
    prob=(z-lo)*(z+lo)/(2.*lo+1.)
    u=(rnd>>16).to(gl.float32)*(1./65536)
    vc=gl.maximum((root>0.).to(gl.float32),lo+(u<prob).to(gl.float32))
    gl.store(V+i,gl.minimum(255.,vc).to(gl.uint8))
    gl.store(VS+group,vs)
    gl.store(PART+group,gl.max(gl.abs(hi.to(gl.float32)),1))
