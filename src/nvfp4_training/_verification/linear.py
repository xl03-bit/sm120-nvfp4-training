import torch
from .nvfp4_cuda import fp4_matmul_cuda as fp4_matmul, fp4_mm_preqB

class FP4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, H, signs, wq_K, wq_N):   # x:[*,K]  w:[N,K]
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        if wq_K is not None:                    # amortized: weight pre-quantized once/step
            y = fp4_mm_preqB(x2, wq_K)          # X @ W^T using cached quant(w) [N,K]
        else:
            y = fp4_matmul(x2, w.t())           # X @ W^T  -> [M,N]  (RNE)
        ctx.save_for_backward(x2, w)
        ctx.H, ctx.signs, ctx.xshape, ctx.wq_N = H, signs, x.shape, wq_N
        return y.reshape(*x.shape[:-1], w.shape[0]).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        x2, w = ctx.saved_tensors
        H, signs, wq_N = ctx.H, ctx.signs, ctx.wq_N
        gy2 = gy.reshape(-1, w.shape[0]).contiguous()
        # Dgrad: dX = dY @ W   (gradient -> SR; weight -> RNE, reuse cached quant(w.t()))
        if wq_N is not None:
            dx = fp4_mm_preqB(gy2, wq_N, sr_a=True)
        else:
            dx = fp4_matmul(gy2, w, sr_a=True, sr_b=False)
        # Wgrad: dW = dY^T @ X  (in-kernel RHT both operands; gradient -> SR; activation -> RNE)
        dw = fp4_matmul(gy2.t().contiguous(), x2, sr_a=True, sr_b=False,
                        rht=True, H=H, signs=signs)
        return dx.reshape(ctx.xshape).to(gy.dtype), dw.to(w.dtype), None, None, None, None
