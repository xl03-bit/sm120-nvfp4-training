"""NVFP4 eager and static-graph training on SM120."""
import torch
from torch import nn
from .nvfp4_fast_v2 import (NVFP4LinearV3, _LinearFn as V3Function, _STATS, Packed,
                           ensure_verified_modes, extension, get_signs, matmul_packed,
                           quantize, round128)

from functools import lru_cache
from pathlib import Path
from .pair import pair

@lru_cache(maxsize=1)
def graph_extension():
    from torch.utils.cpp_extension import load
    src=Path(__file__).parent/'csrc_graph'
    return load(name="sm120_training_graph_sr",sources=[str(src/'bindings.cpp'),str(src/'kernels.cu')],
                extra_cflags=["-O3"],extra_cuda_cflags=["-O3","--fmad=false","-gencode","arch=compute_120a,code=sm_120a"])

def grad_quant(g,signs,w,rht=False):
    clock=getattr(w,"_v5_clock",None)
    if clock is None:
        return quantize(g,signs=signs,sr=True,rht=rht)
    if any(s==0 for s in g.stride()):
        g=g.contiguous()
    offset=clock.sr_calls
    clock.sr_calls+=1
    d,s,p=graph_extension().quantize(g,signs,round128(g.shape[0]),round128(g.shape[1]),
                                     True,rht,offset,None,False,clock.seed)
    _STATS["quant"]+=1
    _STATS["sr_quant"]+=1
    _STATS["rht_quant"]+=int(rht)
    return Packed(d,s,p,g.dtype)

class _LinearFn(V3Function):
    @staticmethod
    def forward(ctx, x, w, bias, signs, need_x, need_w, need_bias, save_packed):
        modes = ensure_verified_modes()
        original_shape = tuple(x.shape)
        x2 = x.reshape(-1, x.shape[-1])
        m, k = x2.shape
        n = w.shape[0]
        is_aligned = all(d % 128 == 0 for d in (m, k, n))
        mode = modes["aligned" if is_aligned else "irregular"]
        cached = getattr(w, "_v4_scale", None)
        pts = None
        if cached is not None:
            version, ptr, stream, value = cached
            if (version == w._version and ptr == w.data_ptr()
                    and stream == torch.cuda.current_stream(w.device).cuda_stream):
                pts = value
                _STATS["optimizer_scale_hits"] = _STATS.get("optimizer_scale_hits",0) + 1
        if hasattr(w,"_v5_graph_scale"):
            pts=w._v5_graph_scale
        a = quantize(x2, signs=signs)
        if mode == "fast_pair" and need_x and w.is_contiguous():
            if pts is not None and w.numel()>=1024*1024 and round128(n)*round128(k)<2**31:
                data,scale,pts,td,ts=pair(w,pts,(64,128,4))
                _STATS["v5_pair"]=_STATS.get("v5_pair",0)+1
            else:
                data, scale, pts, td, ts = extension().quantize(
                    w, signs, round128(n), round128(k), False, False, 0, pts, True)
            b = Packed(data, scale, pts, w.dtype)
            _STATS["quant"] += 2
            _STATS["reused_weight_amax"] += 1
        else:
            b = quantize(w, signs=signs, pts=pts)
        y = matmul_packed(a, b, rows=m, cols=n, output_dtype=x.dtype, bias=bias)
        wpts = b.pts
        # save_for_backward retains parameter version tracking for backward.
        tensors = [w, wpts, signs]
        if need_w:
            if save_packed:
                # Encode the EXACT orientation / RHT needed by wgrad now, then
                # discard the high-precision saved activation for this layer.
                xp = quantize(x2.t(), rht=True, signs=signs)
                tensors += [xp.data, xp.scale, xp.pts]
                _STATS["packed_activation_saved"] += 1
            else:
                tensors += [x2]
        ctx.paired = mode == "fast_pair" and need_x and w.is_contiguous()
        if ctx.paired:
            tensors.extend((td, ts))
        ctx.save_for_backward(*tensors)
        ctx.original_shape = original_shape
        ctx.xdtype = x.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.need_x, ctx.need_w, ctx.need_bias = need_x, need_w, need_bias
        ctx.save_packed = save_packed
        ctx.mode = mode
        ctx.m, ctx.k, ctx.n = m, k, n
        return y.reshape(*original_shape[:-1], n)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        saved = ctx.saved_tensors
        w, wpts, signs = saved[:3]
        g = grad.reshape(ctx.m, ctx.n)
        dx = dw = db = None
        if ctx.need_x:
            aq = grad_quant(g,signs,w)
            if ctx.paired:
                wq = Packed(saved[-2], saved[-1], wpts, w.dtype)
            elif ctx.mode in ("fast_pair", "fast_reuse"):
                wq = quantize(w.t(), signs=signs, pts=wpts)
            else:
                wq = quantize(w.t(), signs=signs)
                _STATS["fresh_dgrad_scale"] += 1
            dx = matmul_packed(aq, wq, rows=ctx.m, cols=ctx.k,
                               output_dtype=ctx.xdtype)
            dx = dx.reshape(ctx.original_shape).to(ctx.xdtype)
        if ctx.need_w:
            aq = grad_quant(g.t(),signs,w,rht=True)
            if ctx.save_packed:
                xq = Packed(saved[3], saved[4], saved[5], ctx.xdtype)
            else:
                xq = quantize(saved[3].t(), rht=True, signs=signs)
            sink = getattr(w, "_v4_pending", None)
            if sink is None or w in sink or aq.dtype != torch.bfloat16:
                dw = matmul_packed(aq,xq,rows=ctx.n,cols=ctx.k,output_dtype=w.dtype)
            else:
                raw = torch._scaled_mm(aq.data.view(torch.float4_e2m1fn_x2),
                    xq.data.t().view(torch.float4_e2m1fn_x2),
                    aq.scale.view(torch.float8_e4m3fn),xq.scale.view(torch.float8_e4m3fn),
                    out_dtype=torch.bfloat16,use_fast_accum=False)
                sink[w] = (raw,aq.pts,xq.pts)
                _STATS["fp4_gemm"] += 1
                _STATS["deferred_wgrad"] = _STATS.get("deferred_wgrad",0)+1
        if ctx.need_bias:
            # Match the usual autograd bias sum: reduce BF16 gradient in FP32,
            # then cast through the forward output's dtype before master dtype.
            db = g.sum(0).to(ctx.bias_dtype)
        return dx, dw, db, None, None, None, None, None


class NVFP4LinearV5(NVFP4LinearV3):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cuda":
            raise RuntimeError("NVFP4 v3 requires CUDA; refusing CPU fallback")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Input last dimension {x.shape[-1]} != {self.in_features}")
        if self.weight.dtype != torch.float32:
            raise TypeError("Keep FP32 master weights; use autocast, not model.bfloat16()")
        grad_enabled = torch.is_grad_enabled()
        return _LinearFn.apply(x, self.weight, self.bias, self._nvfp4_signs,
                               grad_enabled and x.requires_grad,
                               grad_enabled and self.weight.requires_grad,
                               grad_enabled and self.bias is not None and self.bias.requires_grad,
                               self._nvfp4_save_packed)


def enable_nvfp4(model: nn.Module, *, save_quantized_activations: bool = True) -> nn.Module:
    """Convert ordinary nn.Linear, preserving the EXACT Parameter objects.

    Call once before the optimizer, on a CUDA FP32 model. No system package is
    changed. Parameter/state_dict names are preserved; signs are nonpersistent.
    Shared modules and shared Parameter references remain shared. Higher-order
    gradients, FSDP/DTensor/torch.compile/CUDA Graphs need separate integration.
    """
    ensure_verified_modes()
    seen: set[int] = set()
    def convert(m: nn.Module):
        if id(m) in seen:
            return
        seen.add(id(m))
        if isinstance(m, NVFP4LinearV5):
            m._nvfp4_save_packed = bool(save_quantized_activations)
            return
        if isinstance(m, nn.Linear):
            if type(m) is not nn.Linear:
                raise TypeError(f"Refusing to override custom Linear subclass: {type(m)}")
            if m.weight.device.type != "cuda" or m.weight.dtype != torch.float32:
                raise TypeError("Move model to CUDA with FP32 master weights before conversion")
            # No reinitialization, weight copying or extra full-size Parameter.
            m.__class__ = NVFP4LinearV5
            m.weight._v4_request_scale = True
            m.register_buffer("_nvfp4_signs", get_signs(m.weight.device), persistent=False)
            m._nvfp4_save_packed = bool(save_quantized_activations)
            return
        for child in m._modules.values():
            if child is not None:
                convert(child)
    convert(model)
    return model
