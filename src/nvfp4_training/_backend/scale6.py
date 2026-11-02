"""Vectorized exact post-GEMM rounding; reuse the private raw buffer when row strides agree."""
import torch
import triton
import triton.language as tl
from .nvfp4_fast_v2 import _STATS

@triton.jit
def _rescale(R,Y,A,B,BIAS,N:tl.constexpr,C:tl.constexpr,STRIDE:tl.constexpr,
             HAS_BIAS:tl.constexpr,W:tl.constexpr):
    i=tl.program_id(0)*W+tl.arange(0,W)
    off=i if STRIDE==C else (i//C)*STRIDE+i%C
    x=tl.load(R+off,i<N,other=0.).to(tl.float32)
    alpha=tl.load(A)*tl.load(B)
    if R.dtype.element_ty==tl.bfloat16: alpha=alpha.to(tl.bfloat16).to(tl.float32)
    x=tl.inline_asm_elementwise("mul.rn.f32 $0,$1,$2;",constraints="=f,f,f",args=[x,alpha],dtype=tl.float32,is_pure=True,pack=1)
    if R.dtype.element_ty==tl.bfloat16: x=x.to(tl.bfloat16).to(tl.float32)
    if HAS_BIAS:
        bias=tl.load(BIAS+i%C).to(tl.float32)
        if R.dtype.element_ty==tl.bfloat16:bias=bias.to(tl.bfloat16).to(tl.float32)
        x=x+bias
        if R.dtype.element_ty==tl.bfloat16:x=x.to(tl.bfloat16).to(tl.float32)
    tl.store(Y+i,x,i<N)

def rescale(raw,sa,sb,rows,cols,dtype,bias=None):
    if raw.numel()>=2**31:
        from .nvfp4_fast_v2 import extension
        return extension().rescale(raw,sa,sb,rows,cols,dtype==torch.float32,bias)
    y=raw[:rows,:] if raw.dtype==dtype and raw.stride(0)==cols else torch.empty((rows,cols),device=raw.device,dtype=dtype)
    _rescale[(triton.cdiv(rows*cols,1024),)](raw,y,sa,sb,bias if bias is not None else sa,rows*cols,cols,raw.stride(0),bias is not None,1024,num_warps=4,enable_fp_fusion=False)
    return y

def matmul(a,bt,*,rows,cols,output_dtype,bias=None):
    if a.shape[1]!=bt.shape[1] or output_dtype not in (torch.bfloat16,torch.float32):
        raise ValueError('Incompatible contraction or output dtype')
    raw=torch._scaled_mm(a.data.view(torch.float4_e2m1fn_x2),bt.data.t().view(torch.float4_e2m1fn_x2),
                         a.scale.view(torch.float8_e4m3fn),bt.scale.view(torch.float8_e4m3fn),
                         out_dtype=a.dtype,use_fast_accum=False)
    _STATS['fp4_gemm']+=1
    return rescale(raw,a.pts,bt.pts,rows,cols,output_dtype,bias)
