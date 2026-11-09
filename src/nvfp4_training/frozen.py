"""Opt-in frozen-weight FP4 storage and fixed-shape graph replay.

Frozen weights have no high-precision master copy. The input Jacobian still uses
native FP4 GEMMs, so an application must validate both outputs and gradients.
"""
from types import SimpleNamespace
from fnmatch import fnmatchcase
from pathlib import Path
import hashlib
import os
import tempfile
import torch
from torch import nn
from torch.nn import functional as F
from ._runtime import load_runtime

_runtime=None
CLOCK=None
PACKS=0
def runtime():
    global _runtime
    if _runtime is None:
        _,_,backend,_=load_runtime()
        from ._backend import remat,scale6
        from ._backend.nvfp4_v5 import graph_extension,grad_quant
        backend.ensure_verified_modes()
        _runtime=backend,remat,scale6,graph_extension,grad_quant
    return _runtime

class PackedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,d,s,td,ts,p,signs,bias,n):
        backend,_,mm,_,_=runtime()
        shape=x.shape;m=x.numel()//shape[-1]
        a=backend.quantize(x.reshape(m,-1),signs=signs)
        y=mm.matmul(a,backend.Packed(d,s,p,x.dtype),rows=m,cols=n,output_dtype=torch.bfloat16,bias=bias)
        ctx.save_for_backward(td,ts,p,signs);ctx.shape=shape;ctx.m=m
        return y.reshape(*shape[:-1],n)
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,g):
        backend,_,mm,_,grad_quant=runtime()
        td,ts,p,signs=ctx.saved_tensors
        carrier=SimpleNamespace(_v5_clock=CLOCK) if CLOCK is not None else SimpleNamespace()
        q=grad_quant(g.reshape(ctx.m,-1),signs,carrier)
        dx=mm.matmul(q,backend.Packed(td,ts,p,torch.bfloat16),rows=ctx.m,cols=ctx.shape[-1],output_dtype=torch.bfloat16)
        return dx.reshape(ctx.shape),None,None,None,None,None,None,None,None

class PackedLinear(nn.Module):
    def __init__(self,weight,bias=None):
        super().__init__()
        global PACKS
        if weight.requires_grad or (bias is not None and bias.requires_grad):raise ValueError('Only frozen weights may be packed')
        backend,remat,_,_,_=runtime()
        self.out_features,self.in_features=weight.shape
        self.source_elements=weight.numel()
        with torch.no_grad():
            w=weight.detach().to(device='cuda',dtype=torch.bfloat16).contiguous()
            p=w.abs().amax().float().clamp_min(1e-12)/2688.
            d,s=remat.single(w,p,False);td,ts=remat.single(w,p,True)
            for name,t in [('data',d),('scale',s),('transpose_data',td),('transpose_scale',ts),('pts',p)]:self.register_buffer(name,t)
            self.register_buffer('signs',backend.get_signs(w.device),persistent=False)
            self.register_buffer('bias',bias.detach().to(device=w.device,dtype=torch.bfloat16) if bias is not None else None)
        PACKS+=1
        # No full precision master parameter is retained in this module.
    def forward(self,x):
        return PackedFn.apply(x.bfloat16(),self.data,self.scale,self.transpose_data,self.transpose_scale,self.pts,self.signs,self.bias,self.out_features)

class Pointwise(nn.Module):
    def __init__(self,conv,linear=None):
        super().__init__();self.linear=linear if linear is not None else PackedLinear(conv.weight.flatten(1),conv.bias)
    def forward(self,x):return self.linear(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous()

class PackedMHA(nn.Module):
    def __init__(self,m,qkv=None,out=None):
        super().__init__()
        if not m._qkv_same_embed_dim or m.bias_k is not None or m.bias_v is not None or m.add_zero_attn or m.dropout!=0:raise ValueError('Unsupported attention configuration')
        self.num_heads=m.num_heads;self.embed_dim=m.embed_dim;self.batch_first=m.batch_first
        self.qkv=qkv if qkv is not None else PackedLinear(m.in_proj_weight,m.in_proj_bias)
        self.out=out if out is not None else PackedLinear(m.out_proj.weight,m.out_proj.bias)
    def forward(self,query,key,value,key_padding_mask=None,need_weights=True,attn_mask=None,average_attn_weights=True,is_causal=False):
        if query is not key or query is not value or need_weights or key_padding_mask is not None or attn_mask is not None or is_causal:raise ValueError('Only unmasked self attention without returned weights is supported')
        x=query if self.batch_first else query.transpose(0,1)
        n,l,c=x.shape;h=self.num_heads
        q,k,v=self.qkv(x).reshape(n,l,3,h,c//h).permute(2,0,3,1,4).unbind(0)
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask,dropout_p=0,is_causal=is_causal)
        y=self.out(y.transpose(1,2).reshape(n,l,c))
        return (y if self.batch_first else y.transpose(0,1)),None

def _matches(name,patterns):
    return any(fnmatchcase(part,p) for p in patterns for part in (name,*[name.rsplit('.',i)[0] for i in range(1,name.count('.')+1)]))

def _plan(model,include,keep):
    if not include or isinstance(include,str) or isinstance(keep,str):
        raise ValueError('Use nonempty include patterns and optional keep patterns as sequences')
    if hasattr(model,'nvfp4_frozen_manifest'):
        raise ValueError('Pass a fresh model; do not convert it twice')
    aliases={}
    for name,m in model.named_modules(remove_duplicate=False):aliases.setdefault(id(m),[]).append(name)
    planned=[];atomic=[];skipped=[];seen_params=set()
    for name,m in model.named_modules():
        if any((not parent and name) or name.startswith(parent+'.') for parent in atomic):continue
        if isinstance(m,nn.MultiheadAttention):atomic.append(name)
        if not isinstance(m,(nn.Linear,nn.Conv2d,nn.MultiheadAttention)):continue
        if not _matches(name,include) or _matches(name,keep):
            skipped.append(dict(name=name,reason='not selected or protected'));continue
        if isinstance(m,nn.MultiheadAttention) and any(_matches((name+'.' if name else '')+child,keep) for child,_ in m.named_modules() if child):
            skipped.append(dict(name=name,reason='protected attention descendant'));continue
        if len(aliases[id(m)])!=1:raise ValueError('Shared module aliases require explicit integration')
        params=list(m.parameters())
        if any(p.requires_grad for p in params):raise ValueError(f'{name}: selected weights must already be frozen')
        if any(id(p) in seen_params for p in params):raise ValueError('Shared parameters require explicit integration')
        seen_params.update(id(p) for p in params)
        if isinstance(m,nn.MultiheadAttention):
            if not m._qkv_same_embed_dim or m.bias_k is not None or m.bias_v is not None or m.add_zero_attn or m.dropout!=0:
                raise ValueError(f'{name}: unsupported attention configuration')
            kind='mha'
        elif type(m) is nn.Linear:kind='linear'
        elif type(m) is nn.Conv2d and m.kernel_size==(1,1) and m.groups==1 and m.stride==(1,1) and m.padding==(0,0):kind='pointwise'
        else:raise ValueError(f'{name}: unsupported selected layer; protect it explicitly')
        planned.append((name,m,kind))
    if not planned:raise ValueError('No eligible frozen projections selected')
    # A selected parameter may also be owned by an unselected boundary module.
    for name,m,_ in planned:
        selected={id(p) for p in m.parameters()}
        for owner,p in model.named_parameters(remove_duplicate=False):
            if id(p) in selected and name and not owner.startswith(name+'.'):
                raise ValueError('Shared parameters require explicit integration')
    return planned,skipped

def _replace(model,name,replacement):
    if not name:return replacement
    pn,_,leaf=name.rpartition('.')
    parent=model.get_submodule(pn) if pn else model
    setattr(parent,leaf,replacement)
    return model

def pack_frozen_layers(model,*,include,keep=()):
    """Convert explicitly selected frozen modules. Mutates and returns model.

    Supported atoms are standard Linear, ungrouped unit-stride 1x1 Conv2d and
    restricted self-attention MHA. Protect boundary layers explicitly. No module
    is frozen automatically, and trainable optimizer state is not compressed.
    """
    if isinstance(include,str) or isinstance(keep,str):raise ValueError('Pass sequences of patterns, not strings')
    include,keep=tuple(include),tuple(keep)
    planned,skipped=_plan(model,include,keep)
    for name,m,kind in planned:
        if kind=='linear':replacement=PackedLinear(m.weight,m.bias)
        elif kind=='pointwise':replacement=Pointwise(m)
        else:replacement=PackedMHA(m)
        model=_replace(model,name,replacement)
    model.nvfp4_frozen_manifest=dict(include=list(include),keep=list(keep),
        converted=[dict(name=n,kind=k) for n,_,k in planned],skipped=skipped)
    return model

def inventory(module):
    layers=[m for m in module.modules() if isinstance(m,PackedLinear)]
    return dict(projections=len(layers),source_elements=sum(m.source_elements for m in layers),packed_bytes=sum(sum(b.numel()*b.element_size() for n,b in m.named_buffers() if n!='signs') for m in layers),master_parameter_elements=sum(p.numel() for m in layers for p in m.parameters()))

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()

def save_frozen_cache(path,model,*,model_fingerprint):
    """Save only packed frozen projections. Adapters/optimizer state are excluded."""
    if not model_fingerprint:raise ValueError('Bind the cache to a model revision or checkpoint digest')
    manifest=model.nvfp4_frozen_manifest
    layers={}
    for name,m in model.named_modules():
        if isinstance(m,PackedLinear):
            layers[name]=dict(in_features=m.in_features,out_features=m.out_features,
                buffers={k:b.detach().cpu() for k,b in m.named_buffers() if k!='signs'})
    payload=dict(schema=1,codec='nvfp4-bf16-input-v1',model_fingerprint=str(model_fingerprint),manifest=manifest,layers=layers)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:tmp=Path(f.name)
    try:
        torch.save(payload,tmp);os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)
    digest=_sha(path)
    path.with_suffix(path.suffix+'.sha256').write_text(digest+'\n')
    return dict(bytes=path.stat().st_size,sha256=digest,projections=len(layers))

def _restore(row,device):
    m=PackedLinear.__new__(PackedLinear);nn.Module.__init__(m)
    m.in_features=int(row['in_features']);m.out_features=int(row['out_features'])
    m.source_elements=m.in_features*m.out_features
    for name,b in row['buffers'].items():m.register_buffer(name,b.to(device))
    if 'bias' not in row['buffers']:m.register_buffer('bias',None)
    m.register_buffer('signs',runtime()[0].get_signs(m.data.device),persistent=False)
    return m

def load_frozen_cache(path,model,*,expected_fingerprint,device='cuda'):
    """Restore packed buffers into a fresh matching frozen model without quantizing."""
    path=Path(path)
    if _sha(path)!=path.with_suffix(path.suffix+'.sha256').read_text().strip():raise ValueError('Packed cache checksum mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema')!=1 or saved.get('codec')!='nvfp4-bf16-input-v1' or saved.get('model_fingerprint')!=str(expected_fingerprint):
        raise ValueError('Packed cache revision or schema mismatch')
    manifest=saved['manifest'];planned,_=_plan(model,manifest['include'],manifest['keep'])
    if [dict(name=n,kind=k) for n,_,k in planned]!=manifest['converted']:raise ValueError('Packed cache model structure mismatch')
    # Validate every logical shape before mutating any modules.
    for name,m,kind in planned:
        prefix=name+'.' if name else ''
        shapes={name:(m.out_features,m.in_features)} if kind=='linear' else (
            {prefix+'linear':(m.out_channels,m.in_channels)} if kind=='pointwise' else
            {prefix+'qkv':tuple(m.in_proj_weight.shape),prefix+'out':tuple(m.out_proj.weight.shape)})
        for key,(n,k) in shapes.items():
            row=saved['layers'][key]
            if (row['out_features'],row['in_features'])!=(n,k):raise ValueError('Packed cache shape mismatch')
    for name,m,kind in planned:
        prefix=name+'.' if name else ''
        if kind=='linear':replacement=_restore(saved['layers'][name],device)
        elif kind=='pointwise':replacement=Pointwise(m,_restore(saved['layers'][prefix+'linear'],device))
        else:replacement=PackedMHA(m,_restore(saved['layers'][prefix+'qkv'],device),_restore(saved['layers'][prefix+'out'],device))
        model=_replace(model,name,replacement)
    model.nvfp4_frozen_manifest=manifest
    return model

class FrozenReplay:
    """Fixed-shape forward or loss/backward replay; optimizer updates stay external.

    Inputs are copied into fixed buffers and detached from outside graphs. For
    end-to-end objectives, capture the complete differentiable objective rather
    than inserting this object as an autograd-aware submodule.
    """
    def __init__(self,model,compute,*inputs,backward=True,warmup=2):
        global CLOCK
        if CLOCK is not None:raise RuntimeError('A capture is already active')
        if not inputs or warmup<1 or any(not isinstance(t,torch.Tensor) or t.device.type!='cuda' for t in inputs):
            raise ValueError('Use fixed CUDA tensor inputs and at least one warmup')
        device=inputs[0].device
        if any(t.device!=device for t in inputs):raise ValueError('Use one CUDA device')
        self.model=model;self.backward=bool(backward);self.closed=False;self.index=0
        self.caller_stream=torch.cuda.current_stream(device)
        self.params=tuple(p for p in model.parameters() if p.requires_grad)
        self.param_ptrs=tuple(p.data_ptr() for p in self.params)
        self.training=tuple((id(m),m.training) for m in model.modules())
        self.stream=torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream):
            self.inputs=tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in inputs)
            self.targets=(*self.params,*[t for t in self.inputs if t.requires_grad])
            if self.backward and not self.targets:raise ValueError('Backward replay needs trainable parameters or gradient inputs')
            def evaluate():
                with torch.set_grad_enabled(self.backward),torch.autocast('cuda',dtype=torch.bfloat16):
                    value=compute(model,*self.inputs)
                if not isinstance(value,torch.Tensor):raise ValueError('compute must return a Tensor')
                if self.backward and (value.ndim!=0 or not value.requires_grad):raise ValueError('Backward compute must return a differentiable scalar')
                return value
            for _ in range(warmup):
                for p in self.targets:p.grad=None
                value=evaluate()
                if self.backward:value.backward()
                del value
            if self.backward and any(p.grad is None for p in self.targets):raise ValueError('Every gradient target must participate')
            if not self.backward:
                self.targets=()
            self.seed=torch.zeros(1,device=device,dtype=torch.int32)
            clock=SimpleNamespace(sr_calls=0,seed=self.seed)
            runtime()[3]()  # Build dynamic-SR extension before capture.
            self.graph=torch.cuda.CUDAGraph()
            try:
                CLOCK=clock
                with torch.cuda.graph(self.graph,stream=self.stream):
                    for p in self.targets:p.grad.zero_()
                    self.value=evaluate()
                    if self.backward:self.value.backward()
            finally:CLOCK=None
        torch.cuda.current_stream(device).wait_stream(self.stream)
        self.sr_calls=clock.sr_calls
        self.grad_ptrs=tuple(p.grad.data_ptr() for p in self.targets)
        self.frozen_buffers=tuple(b for m in model.modules() if isinstance(m,PackedLinear) for b in m.buffers())
        self.buffer_state=tuple((b.data_ptr(),b._version) for b in self.frozen_buffers)
    def replay(self,*inputs):
        if self.closed:raise RuntimeError('Replay is closed')
        if torch.cuda.current_stream(self.inputs[0].device)!=self.caller_stream:raise RuntimeError('Replay and optimizer use must stay on the original caller stream')
        if tuple((id(m),m.training) for m in self.model.modules())!=self.training:raise RuntimeError('Model structure/training mode changed')
        current_params=tuple(p for p in self.model.parameters() if p.requires_grad)
        if tuple(id(p) for p in current_params)!=tuple(id(p) for p in self.params) or tuple(p.data_ptr() for p in current_params)!=self.param_ptrs:raise RuntimeError('Parameter storage changed')
        current_buffers=tuple(b for m in self.model.modules() if isinstance(m,PackedLinear) for b in m.buffers())
        if tuple((b.data_ptr(),b._version) for b in current_buffers)!=self.buffer_state:raise RuntimeError('Frozen packed buffers changed; rebuild the cache')
        if any(p.grad is None for p in self.targets) or tuple(p.grad.data_ptr() for p in self.targets)!=self.grad_ptrs:
            raise RuntimeError('Gradient buffers changed; replay clears them itself')
        if len(inputs)!=len(self.inputs):raise ValueError('Input arity changed')
        for src,dst in zip(inputs,self.inputs):
            if (src.shape,src.dtype,src.device,src.requires_grad)!=(dst.shape,dst.dtype,dst.device,dst.requires_grad):raise ValueError('Input shape, dtype, device or gradient contract changed')
            with torch.no_grad():dst.copy_(src)
        self.seed.fill_((self.index*self.sr_calls)&0x7fffffff)
        self.graph.replay();self.index+=1
        return self.value.detach()
    def close(self):
        if not self.closed:
            torch.cuda.synchronize(self.inputs[0].device)
            self.graph.reset();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

def capture_frozen(model,compute,*inputs,backward=True,warmup=2):
    return FrozenReplay(model,compute,*inputs,backward=backward,warmup=warmup)
