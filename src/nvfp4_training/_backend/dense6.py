"""Native BF16 control using the same FP16 storage, without narrowing dW to FP16."""
import torch
from .master8 import decoded
from torch import nn
from torch.nn import functional as F

class _StoredFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,b):
        xb=x.reshape(-1,x.shape[-1]).bfloat16()
        wb=decoded(w)
        ctx.save_for_backward(xb,wb,w)
        ctx.shape,ctx.xdtype=x.shape,x.dtype
        ctx.bias_dtype=None if b is None else b.dtype
        out=F.linear(xb,wb,None if b is None else b.bfloat16())
        return out.reshape(*x.shape[:-1],w.shape[0])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,grad):
        x,wb,w=ctx.saved_tensors
        g=grad.reshape(-1,w.shape[0]).bfloat16()
        dx=(g@wb).reshape(ctx.shape).to(ctx.xdtype) if ctx.needs_input_grad[0] else None
        dw=None
        if ctx.needs_input_grad[1]:
            raw=g.t()@x
            sink=getattr(w,'_v4_pending',None)
            if sink is None:
                if w.dtype==torch.float8_e4m3fn: raise RuntimeError("Use backward_step for FP8 storage")
                dw=raw.to(w.dtype)
            elif w in sink:
                sink[w]=(sink[w][0].float()+raw.float(),None,None)
            else:
                sink[w]=(raw,None,None)
        db=g.sum(0).to(ctx.bias_dtype) if ctx.bias_dtype is not None and ctx.needs_input_grad[2] else None
        return dx,dw,db

class DenseStoredLinear(nn.Linear):
    def forward(self,x):
        return _StoredFn.apply(x,self.weight,self.bias)
