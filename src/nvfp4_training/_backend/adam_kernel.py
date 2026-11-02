"""Tiled AdamW updates. Groups contain independent 512-element state blocks."""
import triton
import triton.language as tl
from .pair import _encode
from .master8 import round_e4m3

@triton.jit
def _random_bits(index, step, key):
    z=index.to(tl.uint32)+tl.cast(step,tl.uint32)*0x9e3779b9+tl.cast(key,tl.uint32)
    z=(z^(z>>16))*0x7feb352d
    z=(z^(z>>15))*0x846ca68b
    return z^(z>>16)

@triton.jit
def _signed_tail(x,hi):
    upper=hi.to(tl.uint16,bitcast=True).to(tl.uint32)<<16
    delta=(x.to(tl.uint32,bitcast=True)-upper).to(tl.int32,bitcast=True)
    rounded=(delta+127+((delta>>8)&1))>>8
    return tl.maximum(-128,tl.minimum(127,rounded))

@triton.jit
def split_init(SRC,HI,TAIL,N,HAS_TAIL:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(SRC+i,i<N,other=0.)
    hi=x.to(tl.float16) if HI.dtype.element_ty==tl.float16 else x.to(tl.bfloat16)
    tl.store(HI+i,hi,i<N)
    if HAS_TAIL:
        tl.store(TAIL+i,_signed_tail(x,hi),i<N)

@triton.jit
def update(P,GRAD,M,V,MS,VS,TAIL,PART,SA,SB,EXTRA,CTRL,D,S,DT,ST,CURRENT,TAG,UNIT,SLOTS,TABLE,N,TILES,
           lr,decay,b1,b2,c1,c2,eps,t,key,
           PACKED_GRAD:tl.constexpr,ADD:tl.constexpr,AMAX:tl.constexpr,
           DYNAMIC:tl.constexpr,C:tl.constexpr,STRIDE:tl.constexpr,
           MASTER:tl.constexpr,VB:tl.constexpr,M4:tl.constexpr,NONLINEAR_M:tl.constexpr,FUSE:tl.constexpr,NROW:tl.constexpr,NCOL:tl.constexpr,G:tl.constexpr,B:tl.constexpr,EMBED:tl.constexpr):
    if FUSE:
        pid=tl.program_id(0)
        r0=pid//(NCOL//B)*G
        c0=pid%(NCOL//B)*B
        block=(r0+tl.arange(0,G))*(NCOL//B)+c0//B
    else:
        block=tl.program_id(0)*G+tl.arange(0,G)
    i=block[:,None]*B+tl.arange(0,B)[None,:]
    mask=i<N
    hp=tl.load(P+i,mask,other=0.)
    if MASTER==24:
        bits=hp.to(tl.uint16,bitcast=True).to(tl.uint32)<<16
        tail=tl.load(TAIL+i,mask,other=0).to(tl.int32)*256
        p=(bits+tail.to(tl.uint32)).to(tl.float32,bitcast=True)
    elif MASTER==8:
        tl.static_assert(B==512)
        unit=tl.load(UNIT+block,block<TILES,other=1.)
        p=hp.to(tl.float32)*unit[:,None]
    else:
        p=hp.to(tl.float32)
    if PACKED_GRAD:
        address=i if C==STRIDE else (i//C)*STRIDE+i%C
        alpha=(tl.load(SA)*tl.load(SB)).to(tl.bfloat16).to(tl.float32)
        g=(tl.load(GRAD+address,mask,other=0.).to(tl.float32)*alpha).to(tl.bfloat16).to(tl.float32)
    else:
        g=tl.load(GRAD+i,mask,other=0.).to(tl.float32)
    if ADD: g+=tl.load(EXTRA+i,mask,other=0.).to(tl.float32)
    if EMBED:
        row=tl.load(SLOTS+i//C,mask,other=EMBED)
        g+=tl.load(TABLE+row*C+i%C,mask&(row<EMBED),other=0.).to(tl.float32)
    if DYNAMIC:
        c1=tl.load(CTRL)
        c2=tl.load(CTRL+1)
        t=tl.load(CTRL+3).to(tl.int32,bitcast=True)
    if VB<16 or M4 or MASTER==16 or MASTER==17:
        bits_random=_random_bits(i,t,key)
    if M4:
        if VB==4:
            data=tl.load(M+i,mask,other=7).to(tl.int32)
            codes=tl.reshape((data&15)-7,(G,B//64,64))
        else:
            pair=block[:,None]*(B//2)+tl.arange(0,B//2)[None,:]
            data=tl.load(M+pair,pair<(N+1)//2,other=0).to(tl.int32)
            codes=tl.reshape(tl.join((data&15)-7,(data>>4)-7),(G,B//64,64))
        mi=block[:,None]*(B//64)+tl.arange(0,B//64)[None,:]
        ms=tl.load(MS+mi,mi<(N+63)//64,other=0.)
        mval=codes.to(tl.float32)
        if NONLINEAR_M: mval=mval*tl.abs(mval)
        m=tl.reshape(mval*ms[:,:,None],(G,B))
        m=tl.where(mask,m,0.)
    else:
        m=tl.load(M+i,mask,other=0.).to(tl.float32)*tl.load(MS+block,block<TILES,other=0.)[:,None]
    if VB==4:
        mi=block[:,None]*(B//64)+tl.arange(0,B//64)[None,:]
        if M4:
            vcode4=data>>4
        else:
            vp=block[:,None]*(B//2)+tl.arange(0,B//2)[None,:]
            vdata=tl.load(V+vp,vp<(N+1)//2,other=0).to(tl.int32)
            vcode4=tl.reshape(tl.join(vdata&15,vdata>>4),(G,B))
        vscale=tl.load(VS+mi,mi<(N+63)//64,other=0.)
        v=tl.reshape(tl.reshape(vcode4,(G,B//64,64)).to(tl.float32)*vscale[:,:,None],(G,B))
    else:
        v=tl.load(V+i,mask,other=0.).to(tl.float32)*tl.load(VS+block,block<TILES,other=0.)[:,None]
    if VB<16: v=v*v
    m=b1*m+(1.-b1)*g
    v=b2*v+(1.-b2)*g*g
    root=tl.sqrt(v)
    # Prevent ptxas from contracting weight decay and subtraction in the 2D tile.
    decayed=tl.inline_asm_elementwise("mul.rn.f32 $0,$1,$2;",constraints="=f,f,f",
                                     args=[p,decay],dtype=tl.float32,is_pure=True,pack=1)
    updated=decayed-(lr*c1)*m/(root*c2+eps)
    if MASTER==32:
        tl.store(P+i,updated,mask)
        hi=updated.to(tl.bfloat16)
    elif MASTER==24:
        hi=updated.to(tl.bfloat16)
        tl.store(P+i,hi,mask)
        tl.store(TAIL+i,_signed_tail(updated,hi),mask)
    elif MASTER==8:
        # One CTA owns each unit and all its encoded weights. Grow just the
        # blocks whose actual FP32 update would exceed E4M3's largest value.
        peak=tl.max(tl.where(mask,tl.abs(updated),0.),1)
        grow=(peak>448.*unit)&(peak<float("inf"))
        raised=tl.exp2(tl.floor(tl.log2(tl.maximum(peak,1.e-30)))-7.)
        unit=tl.where(grow,tl.maximum(unit,raised),unit)
        tl.store(UNIT+block,unit,(block<TILES)&grow)
        rnd_master=_random_bits(i,t,key+198743)
        encoded_master=round_e4m3(updated/unit[:,None],rnd_master)
        tl.store(P+i,encoded_master,mask)
        hi=(encoded_master.to(tl.float32)*unit[:,None]).to(tl.bfloat16)
    elif MASTER==17:
        # Binary16 has 10 fraction bits. For normals discard 13 FP32 bits
        # stochastically. Subnormals instead use the fixed 2**-24 quantum.
        raw=updated.to(tl.uint32,bitcast=True)
        normal=((raw+(bits_random&8191))&0xffffe000).to(tl.float32,bitcast=True).to(tl.float16)
        mag=tl.abs(updated)*16777216.
        sub=tl.floor(mag)+( ((bits_random&65535).to(tl.float32)/65536.) < (mag-tl.floor(mag)) ).to(tl.float32)
        sbits=sub.to(tl.uint16)|((raw>>16)&32768).to(tl.uint16)
        half=tl.where(tl.abs(updated)<0.00006103515625,sbits.to(tl.float16,bitcast=True),normal)
        tl.store(P+i,half,mask)
        hi=half.to(tl.float32).to(tl.bfloat16)
    else:
        # Unbiased rounding to adjacent BF16 numbers for the stored weight.
        hbits=(updated.to(tl.uint32,bitcast=True)+(bits_random&65535))>>16
        hi=hbits.to(tl.uint16).to(tl.bfloat16,bitcast=True)
        tl.store(P+i,hi,mask)
    if M4:
        vals=tl.reshape(m,(G,B//64,64))
        if NONLINEAR_M:
            ms=tl.maximum(tl.max(tl.abs(vals),2)/49.,1.e-30)
            norm=tl.minimum(7.,tl.sqrt(tl.abs(vals)/ms[:,:,None]))
            floor=tl.floor(norm)
            mprob=(norm-floor)*(norm+floor)/(2.*floor+1.)
        else:
            ms=tl.maximum(tl.max(tl.abs(vals),2)/7.,1.e-30)
            norm=tl.maximum(-7.,tl.minimum(7.,vals/ms[:,:,None]))
            floor=tl.floor(norm)
            mprob=norm-floor
        rbits=_random_bits(i,t,key+7131821)
        u=tl.reshape((rbits>>8).to(tl.float32)*(1./16777216),(G,B//64,64))
        mc=(floor+(u<mprob).to(tl.float32)).to(tl.int32)
        if NONLINEAR_M: mc=tl.where(vals<0.,-mc,mc)
        code=mc+7
        if VB!=4:
            even,odd=tl.split(tl.reshape(code,(G,B//2,2)))
            tl.store(M+pair,(even|(odd<<4)).to(tl.uint8),pair<(N+1)//2)
        tl.store(MS+mi,ms,mi<(N+63)//64)
    else:
        ms=tl.maximum(tl.max(tl.abs(m),1)/127.,1.e-30)
        code=tl.inline_asm_elementwise("cvt.rni.s32.f32 $0, $1;",constraints="=r,f",args=[m/ms[:,None]],dtype=tl.int32,is_pure=True,pack=1)
        tl.store(M+i,tl.minimum(127,tl.maximum(-127,code)),mask)
        tl.store(MS+block,ms,block<TILES)
    if VB==4:
        roots=tl.reshape(root,(G,B//64,64))
        scale4=tl.maximum(tl.max(roots,2)/15.,1.e-30)
        z4=tl.minimum(15.,roots/scale4[:,:,None])
        lo4=tl.floor(z4)
        prob4=(z4-lo4)*(z4+lo4)/(2.*lo4+1.)
        u4=tl.reshape((bits_random>>16).to(tl.float32)*(1./65536),(G,B//64,64))
        vc=tl.maximum((roots>0.).to(tl.float32),tl.minimum(15.,lo4+(u4<prob4).to(tl.float32))).to(tl.int32)
        if M4:
            tl.store(M+i,tl.reshape(code|(vc<<4),(G,B)).to(tl.uint8),mask)
        else:
            even,odd=tl.split(tl.reshape(vc,(G,B//2,2)))
            tl.store(V+vp,(even|(odd<<4)).to(tl.uint8),vp<(N+1)//2)
        tl.store(VS+mi,scale4,mi<(N+63)//64)
    elif VB==8:
        scale=tl.maximum(tl.max(root,1)/255.,1.e-30)
        z=tl.minimum(255.,root/scale[:,None])
        lo=tl.floor(z)
        # P(up) interpolates in variance space, not square-root space.
        prob=(z-lo)*(z+lo)/(2.*lo+1.)
        u=(bits_random>>16).to(tl.float32)*(1./65536)
        vcode=tl.maximum((root>0.).to(tl.float32),lo+(u<prob).to(tl.float32))
        tl.store(V+i,tl.minimum(255.,vcode).to(tl.uint8),mask)
    else:
        scale=tl.maximum(tl.max(v,1)/32768.,1.e-35)
        tl.store(V+i,v/scale[:,None],mask)
    if VB!=4: tl.store(VS+block,scale,block<TILES)
    if FUSE:
        current=tl.load(CURRENT)
        inv=tl.div_rn(1.,current)
        values=hi.to(tl.float32)
        _encode(values,D,S,inv,r0,c0,NCOL,G,B)
        _encode(tl.trans(values),DT,ST,inv,c0,r0,NROW,B,G)
        if pid==0: tl.store(TAG,current)
    if AMAX:
        tl.store(PART+block,tl.max(tl.where(mask,tl.abs(hi.to(tl.float32)),0.),1),block<TILES)
