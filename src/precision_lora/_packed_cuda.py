"""Small SM120 kernels compiled with the NVRTC shipped in the existing PyTorch.

No compiler installation. Source NVFP4 weights stay packed. BF16 dequantization
matches musubi's Comfy layout reference, including its BF16 scale rounding.
"""
import torch
import os
import ctypes as C
import ctypes.util
from pathlib import Path
import hashlib

SOURCE=r'''
__device__ float bf(unsigned short x) { return __uint_as_float((unsigned)x<<16); }
__device__ unsigned short rb(float x) {
    unsigned u=__float_as_uint(x);return (unsigned short)((u+0x7fff+((u>>16)&1))>>16);
}
__device__ float f8(unsigned char x) {
    int e=(x>>3)&15,m=x&7;
    float v=e ? (1.0f+m*0.125f)*__uint_as_float((unsigned)(e+120)<<23) : m*0.001953125f;
    return (x&128)?-v:v;
}
__device__ unsigned char rf8(float v) {
    if(v<=0.0f)return 0;
    if(v>=448.0f)return 126;
    int e=(int)((__float_as_uint(v)>>23)&255)-127;
    if(e < -6)return (unsigned char)__float2int_rn(v*512.0f);
    int m=__float2int_rn(v*__uint_as_float((unsigned)(130-e)<<23))-8;
    if(m==8){++e;m=0;}
    return (unsigned char)(((e+7)<<3)|m);
}
__device__ float f4(int c) {
    int a=c&7;
    float v=a<2 ? a*.5f : __uint_as_float((unsigned)((a/2+126)<<23)|((a&1)<<22));
    return (c&8)?-v:v;
}
__device__ unsigned mix(unsigned x) {x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return x;}
__device__ unsigned char rf4(float v,unsigned rng,int sr) {
    float a=fabsf(v);int c=0;
    if(a>=6)c=7;
    else {
        c=(a>=.5f)+(a>=1.f)+(a>=1.5f)+(a>=2.f)+(a>=3.f)+(a>=4.f);
        float lo=f4(c),hi=f4(c+1),p=(a-lo)/(hi-lo);
        if(sr ? ((rng>>8)*0x1p-24f < p) : (p>.5f || (p==.5f && (c&1))))++c;
    }
    return c | (signbit(v)?8:0);
}
__device__ unsigned pack_rne(float high,float low) {
    unsigned result;
    asm("{ .reg .b8 byte; cvt.rn.satfinite.e2m1x2.f32 byte,%1,%2; cvt.u32.u8 %0,byte; }" : "=r"(result) : "f"(high),"f"(low));
    return result;
}
__device__ long long swizzle(int r,int b,int k) {
    return (long long)(r/128)*((k+63)/64)*512+(b/4)*512+(r%32)*16+((r%128)/32)*4+b%4;
}
extern "C" __global__ void dequant(const unsigned char* q,const unsigned char* s,const float* g,unsigned short* y,int n,int k) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long long)n*k/2)return;
    int r=i/(k/2),col=(i%(k/2))*2;unsigned char p=q[i];
    float scale=bf(rb(g[0]*f8(s[swizzle(r,col/16,k)])));
    unsigned lo=rb(f4(p>>4)*scale),hi=rb(f4(p&15)*scale);
    ((unsigned*)y)[i]=lo|(hi<<16);
}
extern "C" __global__ void rotate(const unsigned short* x,const float* signs,float* y,float* amax,int rows,int cols) {
    __shared__ float temp[128];int j=threadIdx.x;long long i=(long long)blockIdx.x*128+j;
    if(i>=(long long)rows*cols)return;
    float v=bf(x[i])*signs[j];
    for(int d=1;d<32;d*=2){float b=__shfl_xor_sync(0xffffffff,v,d);v=(j&d)?b-v:v+b;}
    temp[j]=v;__syncthreads();
    float b=temp[j^32];v=(j&32)?b-v:v+b;__syncthreads();temp[j]=v;__syncthreads();
    b=temp[j^64];v=(j&64)?b-v:v+b;v*=0.08838834764831845f;
    y[i]=v;float maximum=fabsf(v);
    for(int d=1;d<32;d*=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    if(!(j&31))atomicMax((int*)amax,__float_as_int(maximum));
}
extern "C" __global__ void amax_bf16(const unsigned short* x,float* amax,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;float v=i<count?fabsf(bf(x[i])):0;
    for(int d=1;d<32;d*=2)v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,d));
    if(!(threadIdx.x&31))atomicMax((int*)amax,__float_as_int(v));
}
extern "C" __global__ void finish_scale(const float* amax,float* scale) {
    scale[0]=fmaxf(amax[0]/2688.0f,1.0e-30f);
}
extern "C" __global__ void quant(const void* x,const float* amax,unsigned char* q,unsigned char* s,float* global,int rows,int cols,unsigned seed,int sr,const unsigned* counter,int input_bf16) {
    if(counter)seed+=counter[0];
    int j=threadIdx.x;long long i=(long long)blockIdx.x*128+j;
    int r=i/cols,c=i%cols;
    float v=r<rows?(input_bf16?bf(((const unsigned short*)x)[i]):((const float*)x)[i]):0.0f,a=fabsf(v);
    for(int d=1;d<16;d*=2)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
    float g=fmaxf(amax[0]/2688.0f,1.0e-30f);
    unsigned char sc=rf8(a/(6.0f*g));float tot=f8(sc)*g;
    float normalized=tot==0?0:v/tot;
    if(sr){
        unsigned char code=rf4(normalized,mix((unsigned)i^seed),1);
        int other=__shfl_xor_sync(0xffffffff,(int)code,1);
        if(r<rows && !(c&1))q[i/2]=(code<<4)|other;
    }else{
        float other=__shfl_xor_sync(0xffffffff,normalized,1);
        if(r<rows && !(c&1))q[i/2]=pack_rne(normalized,other);
    }
    if(r<rows && !(c&15))s[swizzle(r,c/16,cols)]=sc;
    if(i==0)global[0]=g;
}
extern "C" __global__ void rope(const unsigned short* x,const unsigned short* rot,unsigned short* y,int rows,int heads,int dim,int pairs,int backward) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*heads*dim)return;
    int d=i%dim;
    if(d>=pairs*2){y[i]=x[i];return;}
    int base=i-d,p=d%pairs,row=i/(heads*dim),offset=(row*pairs+p)*4;
    float a=bf(x[base+p]),b=bf(x[base+p+pairs]);
    float r00=bf(rot[offset]),r01=bf(rot[offset+1]),r10=bf(rot[offset+2]),r11=bf(rot[offset+3]);
    float v=backward ? (d<pairs ? a*r00+b*r10 : a*r01+b*r11) : (d<pairs ? a*r00+b*r01 : a*r10+b*r11);
    y[i]=rb(v);
}
// Rebuild the backward layout straight from packed forward weights. No dense
// BF16/FP32 matrix, host transfer, or second persistent weight copy is needed.
extern "C" __global__ void transpose_rht_quant(const unsigned char* q,const unsigned char* scales,const float* global,
    const float* signs,const float* backward_global,unsigned char* qt,unsigned char* st,int n,int k,float* amax) {
    __shared__ float tile[32][129];
    int tid=threadIdx.x,nstart=blockIdx.y*128,kstart=blockIdx.x*32;
    int rn=tid/16,kb=tid%16;
    for(int r=rn;r<128;r+=16){
        int nn=nstart+r,kk=kstart+kb*2;
        unsigned char p=q[(long long)nn*(k/2)+kk/2];
        float sc=bf(rb(global[0]*f8(scales[swizzle(nn,kk/16,k)])));
        tile[kb*2][r]=bf(rb(f4(p>>4)*sc))*signs[r];
        tile[kb*2+1][r]=bf(rb(f4(p&15)*sc))*signs[r];
    }
    __syncthreads();
    int warp=tid/32,lane=tid%32;
    for(int kr=warp;kr<32;kr+=8){
        float v[4];
        #pragma unroll
        for(int h=0;h<4;++h)v[h]=tile[kr][lane+32*h];
        #pragma unroll
        for(int d=1;d<32;d*=2){
            #pragma unroll
            for(int h=0;h<4;++h){float b=__shfl_xor_sync(0xffffffff,v[h],d);v[h]=(lane&d)?b-v[h]:v[h]+b;}
        }
        float a=v[0],b=v[1],c=v[2],d=v[3];
        float p0=a+b,p1=a-b,p2=c+d,p3=c-d;
        v[0]=p0+p2;v[1]=p1+p3;v[2]=p0-p2;v[3]=p1-p3;
        if(amax){
            float maximum=0;
            for(int h=0;h<4;++h)maximum=fmaxf(maximum,fabsf(v[h]*0.08838834764831845f));
            for(int d=1;d<32;d*=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
            if(!lane)atomicMax((int*)amax,__float_as_int(maximum));
            continue;
        }
        float g=backward_global[0];
        #pragma unroll
        for(int h=0;h<4;++h){
            float value=v[h]*0.08838834764831845f,maximum=fabsf(value);
            for(int delta=1;delta<16;delta*=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,delta));
            unsigned char sc=rf8(maximum/(6.0f*g));float total=f8(sc)*g;
            float normalized=total==0?0:value/total;
            float other=__shfl_xor_sync(0xffffffff,normalized,1);
            int outrow=kstart+kr,col=nstart+h*32+lane;
            if(!(lane&1))qt[(long long)outrow*n/2+col/2]=pack_rne(normalized,other);
            if(!(lane&15))st[swizzle(outrow,col/16,n)]=sc;
        }
    }
}
'''

P=C.c_void_p;I=C.c_int;U=C.c_uint;S=C.c_size_t
_module=None;_functions={};_device=None
def check(code,op):
    if code:raise RuntimeError(f'{op} failed ({code})')
def init():
    global _module,_driver,_device
    if _module:
        if torch.cuda.current_device()!=_device:
            raise RuntimeError('Use one CUDA device per process with this packed NVFP4 backend')
        return
    torch.cuda.init()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError('Packed NVFP4 backend requires an SM120 GPU')
    if not hasattr(torch.nn.functional, 'scaled_mm'):
        raise RuntimeError('Packed NVFP4 requires torch.nn.functional.scaled_mm')
    libdir=Path(torch.__file__).parent/'lib'
    if os.name=='nt':
        libraries=list(libdir.glob('nvrtc64_*.dll'))
        if len(libraries)!=1:raise RuntimeError('Expected one NVRTC library in the PyTorch installation')
        nvrtc=C.CDLL(str(libraries[0]))
    else:
        library=ctypes.util.find_library('nvrtc')
        if not library:raise RuntimeError('CUDA 13 NVRTC library is required')
        nvrtc=C.CDLL(library)
    for name,args in [('nvrtcCreateProgram',[C.POINTER(P),C.c_char_p,C.c_char_p,I,P,P]),('nvrtcCompileProgram',[P,I,P]),
                     ('nvrtcGetProgramLogSize',[P,C.POINTER(S)]),('nvrtcGetProgramLog',[P,P]),
                     ('nvrtcGetPTXSize',[P,C.POINTER(S)]),('nvrtcGetPTX',[P,P]),('nvrtcDestroyProgram',[C.POINTER(P)])]:
        fn=getattr(nvrtc,name);fn.argtypes=args;fn.restype=I
    cache=Path(os.environ.get('PRECISION_LORA_KERNEL_CACHE', str(Path.home()/'.cache/precision_lora')));cache.mkdir(exist_ok=True)
    path=cache/(hashlib.sha256((SOURCE+'compute_120a'+str(torch.version.cuda)).encode()).hexdigest()+'.ptx')
    if path.exists():ptx=path.read_bytes()
    else:
        prog=P();check(nvrtc.nvrtcCreateProgram(C.byref(prog),SOURCE.encode(),b'nv4.cu',0,None,None),'create NVRTC program')
        opts=(C.c_char_p*3)(b'--gpu-architecture=compute_120a',b'--std=c++17',b'--fmad=false')
        err=nvrtc.nvrtcCompileProgram(prog,3,opts)
        if err:
            n=S();nvrtc.nvrtcGetProgramLogSize(prog,C.byref(n));buf=C.create_string_buffer(n.value);nvrtc.nvrtcGetProgramLog(prog,buf)
            raise RuntimeError(buf.value.decode())
        n=S();check(nvrtc.nvrtcGetPTXSize(prog,C.byref(n)),'PTX size');buf=C.create_string_buffer(n.value)
        check(nvrtc.nvrtcGetPTX(prog,buf),'get PTX');ptx=buf.raw;nvrtc.nvrtcDestroyProgram(C.byref(prog));path.write_bytes(ptx)
    _driver=C.WinDLL('nvcuda.dll') if os.name=='nt' else C.CDLL('libcuda.so.1')
    _driver.cuModuleLoadData.argtypes=[C.POINTER(P),P];_driver.cuModuleLoadData.restype=I
    _driver.cuModuleGetFunction.argtypes=[C.POINTER(P),P,C.c_char_p];_driver.cuModuleGetFunction.restype=I
    _driver.cuLaunchKernel.argtypes=[P,U,U,U,U,U,U,U,P,P,P];_driver.cuLaunchKernel.restype=I
    _module=P();buf=C.create_string_buffer(ptx);check(_driver.cuModuleLoadData(C.byref(_module),buf),'load PTX')
    _device=torch.cuda.current_device()
    for name in ('dequant','rotate','quant','rope','transpose_rht_quant','amax_bf16','finish_scale'):
        f=P();check(_driver.cuModuleGetFunction(C.byref(f),_module,name.encode()),'load kernel');_functions[name]=f

def launch(name,blocks,threads,*args):
    init();values=[P(x.data_ptr()) if isinstance(x,torch.Tensor) else P() if x is None else (U(x) if x>=2**31 else I(x)) for x in args]
    ptrs=(P*len(values))(*[C.cast(C.byref(x),P) for x in values])
    grid=(blocks,1) if isinstance(blocks,int) else blocks
    check(_driver.cuLaunchKernel(_functions[name],grid[0],grid[1],1,threads,1,1,0,P(torch.cuda.current_stream().cuda_stream),ptrs,None),name)

def dequant(q,s,g):
    n,k=q.shape[0],q.shape[1]*2
    out=torch.empty((n,k),device=q.device,dtype=torch.bfloat16)
    launch('dequant',(q.numel()+255)//256,256,q,s,g,out,n,k)
    return out

def quantize(x,signs=None,seed=0,sr=False):
    n,k=x.shape
    assert x.is_cuda and x.is_contiguous() and k%128==0 and x.dtype==torch.bfloat16
    if n%16:
        x=torch.nn.functional.pad(x,(0,0,0,16-n%16));n=x.shape[0]
    if signs is not None:
        work=torch.empty(x.shape,dtype=torch.float32,device=x.device);amax=torch.zeros((),device=x.device)
        launch('rotate',x.numel()//128,128,x,signs,work,amax,n,k)
    else:
        work=x;amax=torch.zeros((),device=x.device)
        launch('amax_bf16',(x.numel()+255)//256,256,x,amax,x.numel())
    q=torch.empty((n,k//2),dtype=torch.uint8,device=x.device)
    s=torch.zeros(((n+127)//128*128,k//16),dtype=torch.float8_e4m3fn,device=x.device)
    g=torch.empty((),device=x.device)
    launch('quant',(x.numel()+127)//128,128,work,amax,q,s,g,n,k,0 if torch.is_tensor(seed) else int(seed)&0xffffffff,int(sr),seed if torch.is_tensor(seed) else None,int(signs is None))
    return q,s,g

def matmul(a,b,dtype=torch.bfloat16):
    from torch.nn.functional import ScalingType as ST,SwizzleType as SW
    return torch.nn.functional.scaled_mm(a[0].view(torch.float4_e2m1fn_x2),b[0].view(torch.float4_e2m1fn_x2).t(),
        scale_a=[a[1].flatten(),a[2]],scale_b=[b[1].flatten(),b[2]],output_dtype=dtype,
        scale_recipe_a=[ST.BlockWise1x16,ST.TensorWise],scale_recipe_b=[ST.BlockWise1x16,ST.TensorWise],
        swizzle_a=[SW.SWIZZLE_32_4_4,SW.NO_SWIZZLE],swizzle_b=[SW.SWIZZLE_32_4_4,SW.NO_SWIZZLE])

def _rope(x,rotation,backward=False):
    assert x.ndim==4 and x.shape[0]==1 and x.dtype==torch.bfloat16 and rotation.dtype==torch.bfloat16
    x=x.contiguous();rotation=rotation.contiguous();out=torch.empty_like(x)
    _,rows,heads,dim=x.shape;pairs=rotation.shape[-3]
    launch('rope',(x.numel()+255)//256,256,x,rotation,out,rows,heads,dim,pairs,int(backward))
    return out

class _RopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,rotation):
        assert not rotation.requires_grad
        ctx.save_for_backward(rotation)
        return _rope(x,rotation)
    @staticmethod
    def backward(ctx,g):return _rope(g,ctx.saved_tensors[0],True),None

def rope(x,rotation):return _RopeFn.apply(x,rotation)

def transpose_rht_quant(source,signs,backward_scale):
    q,s,g=source;n,k=q.shape[0],q.shape[1]*2
    assert n%128==0 and k%128==0
    qt=torch.empty((k,n//2),device=q.device,dtype=torch.uint8)
    st=torch.empty((k,n//16),device=q.device,dtype=torch.float8_e4m3fn)
    launch('transpose_rht_quant',(k//32,n//128),256,q,s,g,signs,backward_scale,qt,st,n,k,None)
    return qt,st,backward_scale


def backward_scale(source, signs):
    q,s,g=source;n,k=q.shape[0],q.shape[1]*2
    assert n%128==0 and k%128==0
    amax=torch.zeros((),device=q.device)
    launch('transpose_rht_quant',(k//32,n//128),256,q,s,g,signs,None,None,None,n,k,amax)
    # Tensor / Python scalar may become reciprocal multiplication, differing
    # by one ULP from the quantizer's CUDA division. Use the same operation.
    scale=torch.empty((),device=q.device)
    launch('finish_scale',1,1,amax,scale)
    return scale
