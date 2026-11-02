import torch
from .act_pair import dual,supported
from .scale6 import matmul as fast_mm
from .master8 import decoded,encode as encode_fp8
from .nvfp4_v5 import grad_quant
from .nvfp4_v5 import (_LinearFn, NVFP4LinearV5, Packed, quantize, matmul_packed, pair,
                      ensure_verified_modes, extension, round128, _STATS)
class _LinearFn6(_LinearFn):
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
        ctx.fast_scale=getattr(w,'_v6_fast_scale',False)
        mm=fast_mm if ctx.fast_scale else matmul_packed
        ctx.dual=getattr(w,'_v6_act_pair',False)
        xp=None
        if ctx.dual and need_w and save_packed and supported(x2):
            a,xp=dual(x2,signs)
        else:a=quantize(x2,signs=signs)
        ctx.remat=getattr(w,"_v6_remat",False) and w.is_contiguous() and round128(n)*round128(k)<2**31
        paired=not ctx.remat and mode=="fast_pair" and need_x and w.is_contiguous()
        if w.dtype!=torch.float32:
            paired=paired and round128(n)*round128(k)<2**31
            if paired and pts is None:
                pts=decoded(w.detach() if w.dtype!=torch.float8_e4m3fn else w).abs().amax().float().clamp_min(1.e-12)/2688.
        if paired:
            if hasattr(w,'_v6_pack'):
                from .prepack import read_pair
                data,scale,pts,td,ts=read_pair(w,pts,w._v6_pack)
            elif w.dtype==torch.float8_e4m3fn:
                data,scale,pts,td,ts=encode_fp8(w,pts)
            elif w.dtype!=torch.float32 or (pts is not None and w.numel()>=1024*1024 and round128(n)*round128(k)<2**31):
                data,scale,pts,td,ts=pair(w,pts,(64,128,4))
                _STATS["v5_pair"]=_STATS.get("v5_pair",0)+1
            else:
                data, scale, pts, td, ts = extension().quantize(
                    w, signs, round128(n), round128(k), False, False, 0, pts, True)
            b = Packed(data, scale, pts, w.dtype)
            _STATS["quant"] += 2
            _STATS["reused_weight_amax"] += 1
        elif ctx.remat:
            from .remat import single
            if pts is None: pts=decoded(w.detach() if w.dtype!=torch.float8_e4m3fn else w).abs().amax().float().clamp_min(1.e-12)/2688.
            if w.dtype==torch.float8_e4m3fn:
                wd,ws,_,_,_=encode_fp8(w,pts,trans=False)
            else:
                wd,ws=single(w,pts,False)
            b=Packed(wd,ws,pts,w.dtype)
            _STATS["quant"]+=1
        else:
            b = quantize(decoded(w) if w.dtype in (torch.float16,torch.float8_e4m3fn) else w, signs=signs, pts=pts)
        y = mm(a, b, rows=m, cols=n, output_dtype=x.dtype, bias=bias)
        wpts = b.pts
        # save_for_backward retains parameter version tracking for backward.
        tensors = [w, wpts, signs]
        if need_w:
            if save_packed:
                # Encode the EXACT orientation / RHT needed by wgrad now, then
                # discard the high-precision saved activation for this layer.
                if xp is None: xp=quantize(x2.t(),rht=True,signs=signs)
                tensors += [xp.data, xp.scale, xp.pts]
                _STATS["packed_activation_saved"] += 1
            else:
                tensors += [x2]
        ctx.paired = paired
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
        mm=fast_mm if ctx.fast_scale else matmul_packed
        adjoint=None
        if ctx.dual and ctx.need_x and ctx.need_w and supported(g):
            ahead,adjoint=dual(g,signs,sr=True,clock=getattr(w,'_v5_clock',None))
        if ctx.need_x:
            aq=ahead if adjoint is not None else grad_quant(g,signs,w)
            if ctx.remat:
                from .remat import single
                if w.dtype==torch.float8_e4m3fn:
                    _,_,_,wd,ws=encode_fp8(w,wpts,row=False)
                else:
                    wd,ws=single(w,wpts,True)
                wq=Packed(wd,ws,wpts,w.dtype)
                _STATS["quant"]+=1
            elif ctx.paired:
                wq = Packed(saved[-2], saved[-1], wpts, w.dtype)
            elif ctx.mode in ("fast_pair", "fast_reuse"):
                wq = quantize(decoded(w).t() if w.dtype in (torch.float16,torch.float8_e4m3fn) else w.t(), signs=signs, pts=wpts)
            else:
                wq = quantize(decoded(w).t() if w.dtype in (torch.float16,torch.float8_e4m3fn) else w.t(), signs=signs)
                _STATS["fresh_dgrad_scale"] += 1
            dx = mm(aq, wq, rows=ctx.m, cols=ctx.k,
                               output_dtype=ctx.xdtype)
            dx = dx.reshape(ctx.original_shape).to(ctx.xdtype)
        if ctx.need_w:
            aq=adjoint if adjoint is not None else grad_quant(g.t(),signs,w,rht=True)
            if ctx.save_packed:
                xq = Packed(saved[3], saved[4], saved[5], ctx.xdtype)
            else:
                xq = quantize(saved[3].t(), rht=True, signs=signs)
            sink = getattr(w, "_v4_pending", None)
            if w.dtype==torch.float8_e4m3fn and (sink is None or w in sink or aq.dtype!=torch.bfloat16):
                raise RuntimeError("FP8 master requires BF16 activations and a single deferred dW contribution")
            if (sink is not None and w not in sink and getattr(w,'_v6_mma_k',0)>=aq.shape[1]
                    and aq.dtype==torch.bfloat16):
                sink[w]=(aq.data,xq.data,aq.scale,xq.scale,aq.pts,xq.pts,aq.shape[1])
                _STATS["fp4_gemm"]+=1
            elif sink is None or w in sink or aq.dtype != torch.bfloat16:
                dw = mm(aq,xq,rows=ctx.n,cols=ctx.k,output_dtype=torch.bfloat16 if w.dtype==torch.float16 else w.dtype)
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


class Linear6(NVFP4LinearV5):
    def forward(self,x):
        if x.device!=self.weight.device or x.shape[-1]!=self.in_features:
            raise ValueError("Invalid activation device or features")
        active=torch.is_grad_enabled()
        return _LinearFn6.apply(x,self.weight,self.bias,self._nvfp4_signs,
                                active and x.requires_grad,active and self.weight.requires_grad,
                                active and self.bias is not None and self.bias.requires_grad,
                                self._nvfp4_save_packed)
