"""Official TE compute with the repository's unchanged stored weights and AdamW.

Only the boundary between stored master weights and TE Linear is adapted.
TE sources/quantizers/GEMMs are unmodified. Padding is internal; output shape,
trainable parameter count and weight tying are unchanged.
"""
import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling
from nvfp4_training._backend.master8 import decoded

RECIPE = NVFP4BlockScaling(disable_rht=False, disable_stochastic_rounding=False)

class StoredWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        ctx.weight = weight
        return decoded(weight)

    @staticmethod
    def backward(ctx, grad):
        weight = ctx.weight
        pending = getattr(weight, '_v4_pending', None)
        if pending is None:
            if weight.dtype == torch.float8_e4m3fn:
                raise RuntimeError('FP8 stored master requires original backward_step')
            return grad.to(weight.dtype)
        if weight in pending:
            pending[weight] = (pending[weight][0].float() + grad.float(), None, None)
        else:
            pending[weight] = (grad.contiguous(), None, None)
        return None

class StoredLinear(te.Linear):
    def __init__(self, source):
        m,n = source.weight.shape
        padded_m,padded_n = (m+31)//32*32,(n+31)//32*32
        super().__init__(padded_n,padded_m,bias=source.bias is not None,
                         params_dtype=torch.bfloat16,device='meta')
        self.weight = source.weight
        if source.bias is not None:
            self.bias = source.bias
        for name, buf in source.named_buffers(recurse=False):
            self.register_buffer(name,buf)
        self.logical_m,self.logical_n=m,n
        self.weight._v6_defer_dense=True

    def _get_weight_tensors(self):
        w=StoredWeight.apply(self.weight)
        if self.in_features!=self.logical_n or self.out_features!=self.logical_m:
            w=F.pad(w,(0,self.in_features-self.logical_n,0,self.out_features-self.logical_m))
        return [w]

    def _get_weight_and_bias_tensors(self):
        w=self._get_weight_tensors()[0]
        b=self.bias if self.use_bias else None
        if b is not None and self.out_features!=self.logical_m:b=F.pad(b,(0,self.out_features-self.logical_m))
        return w,b

    def forward(self,x,**kwargs):
        shape=x.shape
        x=x.reshape(-1,shape[-1])
        rows=x.shape[0]
        if self.in_features!=self.logical_n or rows%32:x=F.pad(x,(0,self.in_features-self.logical_n,0,(-rows)%32))
        y=super().forward(x,**kwargs)
        return y[:rows,:self.logical_m].contiguous().reshape(*shape[:-1],self.logical_m)

def convert(model):
    from torch import nn
    for name,module in list(model.named_children()):
        if isinstance(module,nn.Linear):
            setattr(model,name,StoredLinear(module))
        else:
            convert(module)
    return model

def prepare(model,profile):
    from nvfp4_training import prepare_training
    # BF16 branch selects identical master/state formats without project FP4 compute.
    model,opt=prepare_training(model,profile=profile,precision='bf16')
    convert(model)
    return model,opt

def context():
    return te.autocast(enabled=True,recipe=RECIPE)
