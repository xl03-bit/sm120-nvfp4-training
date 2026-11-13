"""Unmodified official NVFP4 recipe, ordinary FP32 parameters and Adam states.

Only logical-shape padding is supplied for TE's multiple-of-32 requirement.
No project weight storage, optimizer, transpose rematerialization or weight cache.
"""
import torch
from torch.nn import functional as F
import transformer_engine.pytorch as te
from transformer_engine.pytorch.optimizers import FusedAdam
from transformer_engine.common.recipe import NVFP4BlockScaling

RECIPE = NVFP4BlockScaling()

class PaddedLinear(te.Linear):
    def __init__(self, source):
        m,n=source.weight.shape
        super().__init__((n+31)//32*32,(m+31)//32*32,bias=source.bias is not None,
                         params_dtype=source.weight.dtype,device='meta')
        self.weight=source.weight
        if source.bias is not None:self.bias=source.bias
        self.logical_m,self.logical_n=m,n

    def _get_weight_tensors(self):
        w=self.weight
        if self.in_features!=self.logical_n or self.out_features!=self.logical_m:
            w=F.pad(w,(0,self.in_features-self.logical_n,0,self.out_features-self.logical_m))
        return [w]

    def _get_weight_and_bias_tensors(self):
        b=self.bias if self.use_bias else None
        if b is not None and self.out_features!=self.logical_m:
            b=F.pad(b,(0,self.out_features-self.logical_m))
        return self._get_weight_tensors()[0],b

    def forward(self,x):
        # The original LoRA expression applies BF16 autocast at its Linear
        # boundary. TE's RHT quantizer requires that cast explicitly when a
        # preceding LayerNorm yields FP32; do not change LayerNorm precision.
        if torch.is_autocast_enabled('cuda'):
            x=x.to(torch.get_autocast_dtype('cuda'))
        shape=x.shape;x=x.reshape(-1,shape[-1]);rows=x.shape[0]
        if self.in_features!=self.logical_n or rows%32:
            x=F.pad(x,(0,self.in_features-self.logical_n,0,(-rows)%32))
        # is_first_microbatch=None is the official default: no persistent cache.
        y=super().forward(x)
        return y[:rows,:self.logical_m].contiguous().reshape(*shape[:-1],self.logical_m)

def convert(model):
    for name,child in list(model.named_children()):
        if isinstance(child,torch.nn.Linear):setattr(model,name,PaddedLinear(child))
        else:convert(child)
    return model

def prepare(model,profile=None):
    model=convert(model.cuda())
    opt=FusedAdam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,
                  weight_decay=.01,capturable=True)
    # Diagnostic ownership only; no project update code is called.
    model.nvfp4_optimizer=opt
    return model,opt

def context():return te.autocast(enabled=True,recipe=RECIPE)

class CapturedStep:
    """Standard forward/backward/official Adam captured from update zero."""
    def __init__(self,model,opt,compute,x,y,steps,**unused):
        self.model,self.opt,self.steps,self.index=model,opt,steps,0
        self.x,self.y=x.detach().clone(),y.detach().clone()
        # FusedAdam initializes states even with no gradients. No parameter update.
        opt.zero_grad(set_to_none=True);opt.step()
        for group in opt.param_groups:group['step'].zero_()
        self.graph=torch.cuda.CUDAGraph()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda.current_blas_handle()
            tiny=torch.ones((16,16),device=x.device,dtype=torch.bfloat16,requires_grad=True)
            (tiny@tiny).sum().backward();del tiny
        opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph,stream=stream):
            self.loss=compute(self.x,self.y)
            self.loss.backward()
            opt.step()
        self.loss=self.loss.detach()
        self.closed=False

    def step(self,x,y):
        if self.closed or self.index>=self.steps:raise RuntimeError('Replay budget exhausted')
        self.x.copy_(x);self.y.copy_(y);self.graph.replay();self.index+=1
        return self.loss

    def close(self):
        if not self.closed:
            self.opt.zero_grad(set_to_none=True);self.graph.reset();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
