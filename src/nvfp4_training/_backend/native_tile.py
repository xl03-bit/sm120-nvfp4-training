"""SM120 FP4 warp MMA probe. Fragment indexing follows NVIDIA's SM120 MMA traits."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._layouts import BlockedLayout,SliceLayout

@gluon.jit
def _scale_word(S,row,kbase,T:gl.constexpr):
    idx=((row//128)*(T//64)+kbase//64)*128+(row%32)*4+(row%128)//32
    return gl.load(S.to(gl.pointer_type(gl.uint32))+idx)

@gluon.jit
def _fragments(A,B,AS,BS,r0,c0,T:gl.constexpr,C:gl.constexpr,W:gl.constexpr):
    triples:gl.constexpr=BlockedLayout([1,2,2],[32,1,1],[W,1,1],[2,1,0])
    two:gl.constexpr=SliceLayout(2,triples)
    lanes:gl.constexpr=SliceLayout(1,two)
    threads=gl.arange(0,(C//8)*32,layout=lanes)
    lane=threads%32
    ng=threads//32
    row=r0+lane//4
    col=c0+ng*8+lane//4
    kword=lane%4
    au=A.to(gl.pointer_type(gl.uint32))
    bu=B.to(gl.pointer_type(gl.uint32))
    d0=gl.full(((C//8)*32,),0,gl.float32,layout=lanes)
    d1=d0;d2=d0;d3=d0
    for kb in range(T//64):
        k=kb*8+kword
        a0=gl.load(au+row*(T//8)+k)
        a1=gl.load(au+(row+8)*(T//8)+k)
        a2=gl.load(au+row*(T//8)+k+4)
        a3=gl.load(au+(row+8)*(T//8)+k+4)
        b0=gl.load(bu+col*(T//8)+k)
        b1=gl.load(bu+col*(T//8)+k+4)
        sa=_scale_word(AS,row+(lane%2)*8,kb*64,T)
        sb=_scale_word(BS,col,kb*64,T)
        d0,d1,d2,d3=gl.inline_asm_elementwise(
            asm="mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, {$14}, {0,0}, {$15}, {0,0};",
            constraints="=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
            args=[a0,a1,a2,a3,b0,b1,d0,d1,d2,d3,sa,sb],
            dtype=(gl.float32,gl.float32,gl.float32,gl.float32),is_pure=True,pack=1)
    return d0,d1,d2,d3

@gluon.jit
def grad_tile(A,B,AS,BS,r0,c0,T:gl.constexpr,C:gl.constexpr,W:gl.constexpr,OUTPUT:gl.constexpr):
    d0,d1,d2,d3=_fragments(A,B,AS,BS,r0,c0,T,C,W)
    merged=gl.join(gl.join(d0,d1),gl.join(d2,d3))
    shaped=gl.reshape(merged,((C//8),8,4,2,2))
    matrix=gl.reshape(gl.permute(shaped,(4,1,0,2,3)),(16,C))
    return gl.convert_layout(matrix,OUTPUT)

@gluon.jit
def native_raw(A,B,AS,BS,D,N:gl.constexpr,T:gl.constexpr,C:gl.constexpr,W:gl.constexpr):
    layout:gl.constexpr=BlockedLayout([1,4],[1,32],[W,1],[1,0])
    r0=gl.program_id(0)*16;c0=gl.program_id(1)*C
    vals=grad_tile(A,B,AS,BS,r0,c0,T,C,W,layout)
    ri=r0+gl.arange(0,16,layout=gl.constexpr(SliceLayout(1,layout)))
    ci=c0+gl.arange(0,C,layout=gl.constexpr(SliceLayout(0,layout)))
    gl.store(D+ri[:,None]*N+ci[None,:],vals.to(gl.bfloat16))


@gluon.jit
def raw_direct(A,B,AS,BS,D,N:gl.constexpr,T:gl.constexpr,C:gl.constexpr,W:gl.constexpr):
    triples:gl.constexpr=BlockedLayout([1,2,2],[32,1,1],[W,1,1],[2,1,0])
    lanes:gl.constexpr=SliceLayout(1,SliceLayout(2,triples))
    tid=gl.arange(0,C//8*32,layout=lanes)
    r0=gl.program_id(0)*16
    c0=gl.program_id(1)*C
    d0,d1,d2,d3=_fragments(A,B,AS,BS,r0,c0,T,C,W)
    idx=(r0+(tid%32)//4)*(N//2)+c0//2+(tid//32)*4+tid%4
    lo=d0.to(gl.bfloat16).to(gl.uint16,bitcast=True).to(gl.uint32)
    hi=d1.to(gl.bfloat16).to(gl.uint16,bitcast=True).to(gl.uint32)
    gl.store(D.to(gl.pointer_type(gl.uint32))+idx,lo|(hi<<16))
    lo=d2.to(gl.bfloat16).to(gl.uint16,bitcast=True).to(gl.uint32)
    hi=d3.to(gl.bfloat16).to(gl.uint16,bitcast=True).to(gl.uint32)
    gl.store(D.to(gl.pointer_type(gl.uint32))+idx+8*(N//2),lo|(hi<<16))
