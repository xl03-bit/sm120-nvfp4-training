// New implementation for the NVFP4 v2 adapter. No native FP4 .rs instruction.
// The stochastic quantizer uses adjacent E2M1 levels; RHT is H16 / 4.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace {
constexpr int THREADS = 256;
constexpr unsigned MASK = 0xffffffffu;

template<bool F32>
__device__ __forceinline__ float input_value(const void* p, int64_t i) {
    if constexpr (F32) {
        // Match the existing backend's float32 -> BF16 -> quantization path,
        // without materializing a full BF16 temporary tensor.
        return __bfloat162float(__float2bfloat16_rn(static_cast<const float*>(p)[i]));
    } else {
        return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
    }
}
__device__ __forceinline__ float block_max(float x) {
    __shared__ float warp_maxima[8];
    for (int d=16; d>0; d/=2) x=fmaxf(x, __shfl_down_sync(MASK,x,d));
    if ((threadIdx.x & 31)==0) warp_maxima[threadIdx.x/32]=x;
    __syncthreads();
    float a=(threadIdx.x<8)?warp_maxima[threadIdx.x]:0.0f;
    if (threadIdx.x<32) {
        for (int d=16; d>0; d/=2) a=fmaxf(a,__shfl_down_sync(MASK,a,d));
    }
    return a;
}
// Exact same counter-to-uniform mapping as the previously validated backend.
// Thus fixed-seed comparisons test numerical equivalence, not just "runs".
__device__ __forceinline__ float uniform24(uint32_t seed, uint32_t index) {
    uint32_t z=seed*0x9e3779b9u+index+0x85ebca6bu;
    z^=z>>16; z*=0x7feb352du; z^=z>>15; z*=0x846ca68bu; z^=z>>16;
    return static_cast<float>(z>>8)*(1.0f/16777216.0f);
}
__device__ __forceinline__ int sr_magnitude(float x,float u) {
    float lo,hi; int code;
    if (x<0.5f) {lo=0.f;hi=0.5f;code=0;}
    else if (x<1.f) {lo=0.5f;hi=1.f;code=1;}
    else if (x<1.5f) {lo=1.f;hi=1.5f;code=2;}
    else if (x<2.f) {lo=1.5f;hi=2.f;code=3;}
    else if (x<3.f) {lo=2.f;hi=3.f;code=4;}
    else if (x<4.f) {lo=3.f;hi=4.f;code=5;}
    else if (x<6.f) {lo=4.f;hi=6.f;code=6;}
    else return 7;
    return code+static_cast<int>(u<(x-lo)/(hi-lo));
}

template<bool F32>
__global__ void reduce_input(const void* x,float* partial,int64_t m,int64_t k,
    int64_t s0,int64_t s1,bool dense,bool final,float factor) {
    float a=0.f;
    for (int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<m*k; i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
        int64_t off=dense?i:(i/k)*s0+(i%k)*s1;
        a=fmaxf(a,fabsf(input_value<F32>(x,off)));
    }
    a=block_max(a);
    if (threadIdx.x==0) partial[blockIdx.x]=final?
        (factor*fmaxf(a,1.e-12f))*(1.f/(6.f*448.f)):a;
}
__global__ void finish_reduce(const float* partial,float* pts,int count,float factor) {
    float a=0.f;
    for (int i=threadIdx.x;i<count;i+=blockDim.x) a=fmaxf(a,partial[i]);
    a=block_max(a);
    if (threadIdx.x==0) pts[0]=(factor*fmaxf(a,1.e-12f))*(1.f/(6.f*448.f));
}

__device__ __forceinline__ int64_t scale_offset(int64_t row,int64_t col,
                                               int64_t padded_k) {
    // SWIZZLE_32_4_4. col is an ELEMENT column, not a scale column.
    return ((row/128)*(padded_k/64)+col/64)*512+
           (row%32)*16+((row%128)/32)*4+(col%64)/16;
}

template<bool SR,bool RHT>
__device__ __forceinline__ void emit_group(float value,int64_t row,int64_t col,
    int64_t kp,uint8_t* packed,uint8_t* scales,const float* pts,
    const float* signs,uint32_t seed) {
    const int lane=threadIdx.x&15;
    if constexpr (RHT) {
        value*=signs[lane];
        #pragma unroll
        for (int d=1;d<16;d*=2) {
            float other=__shfl_xor_sync(MASK,value,d,16);
            value=(lane&d)?other-value:value+other;
        }
        value*=0.25f;
    }
    float a=fabsf(value);
    #pragma unroll
    for (int d=8;d>0;d/=2) a=fmaxf(a,__shfl_xor_sync(MASK,a,d,16));
    float inv=1.f/pts[0];
    float encoded=fminf(fmaxf((a/6.f)*inv,0.015625f),448.f);
    __nv_fp8_e4m3 enc8(encoded);
    if (lane==0) scales[scale_offset(row,col,kp)]=*reinterpret_cast<uint8_t*>(&enc8);
    float v=fminf(fmaxf(value*(inv/static_cast<float>(enc8)),-6.f),6.f);
    if constexpr (SR) {
        int code=sr_magnitude(fabsf(v),uniform24(seed,static_cast<uint32_t>(row*kp+col)));
        code|=(v<0.f)?8:0;
        int next=__shfl_down_sync(MASK,code,1,16);
        if ((lane&1)==0) packed[row*(kp/2)+col/2]=static_cast<uint8_t>(code|(next<<4));
    } else {
        float next=__shfl_down_sync(MASK,v,1,16);
        if ((lane&1)==0) {
            auto p=__nv_cvt_float2_to_fp4x2(make_float2(v,next),__NV_E2M1,cudaRoundNearest);
            packed[row*(kp/2)+col/2]=static_cast<uint8_t>(p);
        }
    }
}

template<bool F32,bool SR,bool RHT>
__global__ void quant_rows(const void* x,uint8_t* packed,uint8_t* scales,
    const float* pts,const float* signs,int64_t m,int64_t k,int64_t mp,int64_t kp,
    int64_t s0,int64_t s1,uint32_t seed) {
    // A 16x128 tile per CTA. Constant-power-of-two indexing avoids a
    // per-element 64-bit division, while each half-warp processes 1x16.
    int64_t r0=static_cast<int64_t>(blockIdx.y)*16;
    int64_t c0=static_cast<int64_t>(blockIdx.x)*128;
    #pragma unroll 2
    for(int i=threadIdx.x;i<16*128;i+=THREADS) {
        int64_t row=r0+i/128, col=c0+i%128;
        float v=(row<m && col<k)?input_value<F32>(x,row*s0+col*s1):0.f;
        emit_group<SR,RHT>(v,row,col,kp,packed,scales,pts,signs,seed);
    }
}

template<bool F32,bool SR,bool RHT>
__global__ void quant_transpose(const void* x,uint8_t* packed,uint8_t* scales,
    const float* pts,const float* signs,int64_t m,int64_t k,int64_t mp,int64_t kp,
    int64_t s0,int64_t s1,uint32_t seed) {
    // Coalesced source reads for a transposed view (s0==1), then shared-memory
    // transpose. Transposition, BF16 cast, optional RHT/SR, packing and swizzle
    // all happen in this one kernel; no full transposed tensor is allocated.
    __shared__ float tile[32][65];
    int64_t r0=static_cast<int64_t>(blockIdx.y)*32;
    int64_t c0=static_cast<int64_t>(blockIdx.x)*64;
    #pragma unroll
    for (int i=threadIdx.x;i<32*64;i+=THREADS) {
        int rr=i%32, cc=i/32;
        int64_t r=r0+rr,c=c0+cc;
        tile[rr][cc]=(r<m && c<k)?input_value<F32>(x,r*s0+c*s1):0.f;
    }
    __syncthreads();
    #pragma unroll
    for (int i=threadIdx.x;i<32*64;i+=THREADS) {
        int rr=i/64,cc=i%64;
        emit_group<SR,RHT>(tile[rr][cc],r0+rr,c0+cc,kp,packed,scales,pts,signs,seed);
    }
}

// Read a contiguous FP32 weight tile once; independently compute 1x16
// scales for each contraction axis. Packed nibbles cannot be transposed.
__global__ void quant_weight_pair(const float* x, uint8_t* data, uint8_t* scale,
    uint8_t* transposed, uint8_t* transposed_scale, const float* pts,
    int m, int k, int mp, int kp) {
    __shared__ float tile[32][65];
    const int r0 = blockIdx.y * 32, c0 = blockIdx.x * 64;
    #pragma unroll
    for (int i = threadIdx.x; i < 32*64; i += THREADS) {
        int r = i/64, c = i%64;
        float v = (r0+r<m && c0+c<k) ?
            input_value<true>(x, static_cast<int64_t>(r0+r)*k+c0+c) : 0.f;
        tile[r][c] = v;
        emit_group<false,false>(v,r0+r,c0+c,kp,data,scale,pts,nullptr,0);
    }
    __syncthreads();
    #pragma unroll
    for (int i = threadIdx.x; i < 32*64; i += THREADS) {
        int r = i%32, c = i/32;
        emit_group<false,false>(tile[r][c],c0+c,r0+r,mp,transposed,
                                transposed_scale,pts,nullptr,0);
    }
}

template<bool F32,bool SR,bool RHT>
void launch_quant(const void* x,uint8_t* d,uint8_t* s,const float* p,const float* sg,
    int64_t m,int64_t k,int64_t mp,int64_t kp,int64_t s0,int64_t s1,
    uint32_t seed,cudaStream_t stream) {
    if (s0==1 && s1!=1) {
        dim3 grid(static_cast<unsigned>(kp/64),static_cast<unsigned>(mp/32));
        quant_transpose<F32,SR,RHT><<<grid,THREADS,0,stream>>>(x,d,s,p,sg,m,k,mp,kp,s0,s1,seed);
    } else {
        dim3 grid(static_cast<unsigned>(kp/128),static_cast<unsigned>(mp/16));
        quant_rows<F32,SR,RHT><<<grid,THREADS,0,stream>>>(x,d,s,p,sg,m,k,mp,kp,s0,s1,seed);
    }
}

template<bool F32>
void select_quant(const void* x,uint8_t* d,uint8_t* s,const float* p,const float* sg,
    int64_t m,int64_t k,int64_t mp,int64_t kp,int64_t s0,int64_t s1,
    uint32_t seed,bool sr,bool rht,cudaStream_t stream) {
    if(sr && rht) launch_quant<F32,true,true>(x,d,s,p,sg,m,k,mp,kp,s0,s1,seed,stream);
    else if(sr) launch_quant<F32,true,false>(x,d,s,p,sg,m,k,mp,kp,s0,s1,seed,stream);
    else if(rht) launch_quant<F32,false,true>(x,d,s,p,sg,m,k,mp,kp,s0,s1,seed,stream);
    else launch_quant<F32,false,false>(x,d,s,p,sg,m,k,mp,kp,s0,s1,seed,stream);
}

template<bool RAW32,bool OUT32>
__global__ void rescale_kernel(const void* raw,void* output,const float* sa,
    const float* sb,int64_t m,int64_t n,int64_t stride,const void* bias,
    bool bias32) {
    int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=m*n) return;
    const int64_t col=bias?i%n:0;
    int64_t offset=stride==n?i:(i/n)*stride+i%n;
    float alpha=sa[0]*sb[0];
    if constexpr(!RAW32) alpha=__bfloat162float(__float2bfloat16_rn(alpha));
    float v;
    if constexpr(RAW32) v=static_cast<const float*>(raw)[offset];
    else v=__bfloat162float(static_cast<const __nv_bfloat16*>(raw)[offset]);
    v=v*alpha;
    if constexpr(!RAW32) v=__bfloat162float(__float2bfloat16_rn(v));
    if(bias) {
        float b=bias32?static_cast<const float*>(bias)[col]:
                      __bfloat162float(static_cast<const __nv_bfloat16*>(bias)[col]);
        if constexpr(!RAW32) b=__bfloat162float(__float2bfloat16_rn(b));
        v=v+b;
        if constexpr(!RAW32) v=__bfloat162float(__float2bfloat16_rn(v));
    }
    if constexpr(OUT32) static_cast<float*>(output)[i]=v;
    else static_cast<__nv_bfloat16*>(output)[i]=__float2bfloat16_rn(v);
}
} // namespace

std::vector<torch::Tensor> quantize_cuda(torch::Tensor x,torch::Tensor signs,
    int64_t mp,int64_t kp,bool sr,bool rht,int64_t seed,
    c10::optional<torch::Tensor> given_scale, bool pair) {
    TORCH_CHECK(x.is_cuda() && x.dim()==2,"quantize requires a CUDA 2D tensor");
    TORCH_CHECK(x.scalar_type()==at::kFloat || x.scalar_type()==at::kBFloat16,
                "quantize supports float32 and bfloat16 only");
    TORCH_CHECK(mp>=x.size(0) && kp>=x.size(1) && mp%128==0 && kp%128==0,
                "padded dimensions must be >= original dimensions and multiples of 128");
    TORCH_CHECK(mp/16<=65535, "row count exceeds this revision's CUDA grid bound");
    TORCH_CHECK(x.numel()>0 && x.stride(0)>0 && x.stride(1)>0,"empty/broadcast/negative strides unsupported");
    TORCH_CHECK(signs.is_cuda() && signs.device()==x.device() && signs.scalar_type()==at::kFloat &&
                signs.numel()==16 && signs.is_contiguous(),"invalid RHT signs");
    const c10::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    auto fopts=x.options().dtype(at::kFloat);
    auto bytes=x.options().dtype(at::kByte);
    auto packed=torch::empty({mp,kp/2},bytes);
    auto scales=torch::empty({mp/128*32,kp/64*16},bytes);
    torch::Tensor pts;
    if(given_scale.has_value()) {
        pts=given_scale.value();
        TORCH_CHECK(pts.is_cuda() && pts.device()==x.device() && pts.scalar_type()==at::kFloat &&
                    pts.numel()==1 && pts.is_contiguous(),"invalid reused per-tensor scale");
    } else {
        pts=torch::empty({},fopts);
        int count=static_cast<int>(std::min<int64_t>((x.numel()+1023)/1024,1024));
        count=std::max(1,count);
        auto partial=count==1?pts:torch::empty({count},fopts);
        bool dense=x.is_contiguous() || (x.stride(0)==1 && x.stride(1)==x.size(0));
        float factor=rht?4.f:1.f;
        if(x.scalar_type()==at::kFloat)
            reduce_input<true><<<count,THREADS,0,stream>>>(x.data_ptr(),partial.data_ptr<float>(),
                x.size(0),x.size(1),x.stride(0),x.stride(1),dense,count==1,factor);
        else
            reduce_input<false><<<count,THREADS,0,stream>>>(x.data_ptr(),partial.data_ptr<float>(),
                x.size(0),x.size(1),x.stride(0),x.stride(1),dense,count==1,factor);
        if(count>1) finish_reduce<<<1,THREADS,0,stream>>>(partial.data_ptr<float>(),pts.data_ptr<float>(),count,factor);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if(pair) {
        TORCH_CHECK(!sr && !rht && x.is_contiguous() && x.scalar_type()==at::kFloat,
                    "paired encoding requires contiguous FP32 RNE weights");
        auto dt = torch::empty({kp,mp/2},bytes);
        auto st = torch::empty({kp/128*32,mp/64*16},bytes);
        dim3 grid(kp/64,mp/32);
        quant_weight_pair<<<grid,THREADS,0,stream>>>(x.data_ptr<float>(),packed.data_ptr<uint8_t>(),
            scales.data_ptr<uint8_t>(),dt.data_ptr<uint8_t>(),st.data_ptr<uint8_t>(),
            pts.data_ptr<float>(),x.size(0),x.size(1),mp,kp);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {packed,scales,pts,dt,st};
    }
    if(x.scalar_type()==at::kFloat)
        select_quant<true>(x.data_ptr(),packed.data_ptr<uint8_t>(),scales.data_ptr<uint8_t>(),
            pts.data_ptr<float>(),signs.data_ptr<float>(),x.size(0),x.size(1),mp,kp,
            x.stride(0),x.stride(1),static_cast<uint32_t>(seed),sr,rht,stream);
    else
        select_quant<false>(x.data_ptr(),packed.data_ptr<uint8_t>(),scales.data_ptr<uint8_t>(),
            pts.data_ptr<float>(),signs.data_ptr<float>(),x.size(0),x.size(1),mp,kp,
            x.stride(0),x.stride(1),static_cast<uint32_t>(seed),sr,rht,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {packed,scales,pts};
}

torch::Tensor rescale_cuda(torch::Tensor raw,torch::Tensor sa,torch::Tensor sb,
    int64_t m,int64_t n,bool out32,c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(raw.is_cuda() && raw.dim()==2 && raw.is_contiguous(),"raw output must be contiguous CUDA matrix");
    TORCH_CHECK(raw.scalar_type()==at::kFloat || raw.scalar_type()==at::kBFloat16,"unsupported raw output dtype");
    TORCH_CHECK(m>0 && n>0 && m<=raw.size(0) && n<=raw.size(1),"invalid output crop");
    for(const auto& s : {sa,sb}) TORCH_CHECK(s.device()==raw.device() && s.scalar_type()==at::kFloat &&
                                          s.numel()==1 && s.is_contiguous(),"invalid output scale");
    const c10::cuda::CUDAGuard guard(raw.device());
    cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    bool reuse=m==raw.size(0) && n==raw.size(1) && out32==(raw.scalar_type()==at::kFloat);
    auto out=reuse?raw:torch::empty({m,n},raw.options().dtype(out32?at::kFloat:at::kBFloat16));
    const void* b=nullptr; bool b32=false;
    if(bias.has_value()) {
        const auto& bt=bias.value();
        TORCH_CHECK(bt.device()==raw.device() && bt.is_contiguous() && bt.numel()==n &&
                    (bt.scalar_type()==at::kFloat || bt.scalar_type()==at::kBFloat16),"invalid bias");
        b=bt.data_ptr();b32=bt.scalar_type()==at::kFloat;
    }
    int64_t blocks=(m*n+THREADS-1)/THREADS;
    if(raw.scalar_type()==at::kFloat) {
        if(out32) rescale_kernel<true,true><<<blocks,THREADS,0,stream>>>(raw.data_ptr(),out.data_ptr(),sa.data_ptr<float>(),sb.data_ptr<float>(),m,n,raw.size(1),b,b32);
        else rescale_kernel<true,false><<<blocks,THREADS,0,stream>>>(raw.data_ptr(),out.data_ptr(),sa.data_ptr<float>(),sb.data_ptr<float>(),m,n,raw.size(1),b,b32);
    } else {
        if(out32) rescale_kernel<false,true><<<blocks,THREADS,0,stream>>>(raw.data_ptr(),out.data_ptr(),sa.data_ptr<float>(),sb.data_ptr<float>(),m,n,raw.size(1),b,b32);
        else rescale_kernel<false,false><<<blocks,THREADS,0,stream>>>(raw.data_ptr(),out.data_ptr(),sa.data_ptr<float>(),sb.data_ptr<float>(),m,n,raw.size(1),b,b32);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
