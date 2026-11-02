"""Optional AdamW storage formats. Conversion must precede optimizer creation."""
from collections import Counter
import math
import torch
from torch import nn
import triton
from .adam_kernel import split_init,update
from .nvfp4_v5 import enable_nvfp4,NVFP4LinearV5
from .linear6 import Linear6
from .optimizer_v5 import AdamW as AdamW5, _finish_scale, _partial_max

# (master storage, variance bits, momentum bits)
FORMATS={'tile32':(32,16,8),'fp32_v8':(32,8,8),'split24_v16':(24,16,8),
         'split24_v8':(24,8,8),'sr16_v8':(16,8,8),
         'split24_m4':(24,8,4),'sr16_m4':(16,8,4),'sr16_m4v4':(16,4,4),
         'half16_v16':(17,16,8),'half16_v8':(17,8,8),'half16_m8v4':(17,4,8),
         'half16_m4v4':(17,4,4),'half16_m4v4_sqrt':(17,4,4),'float8_v8':(8,8,8),'float8_m4v4_sqrt':(8,4,4),'half16_v8b64':(17,8,8),'half16_v8b128':(17,8,8),'half16_v8b256':(17,8,8)}

def state_block(fmt):
    return int(fmt.split('v8b')[1]) if 'v8b' in fmt else 512

class AdamW6(AdamW5):
    def __init__(self,params,fmt,lr=1e-4,tiles=4,warps=4,**kw):
        super().__init__(params,lr=lr,**kw)
        for group in self.param_groups:
            group.update(format=fmt,tiles=tiles,warps=warps,state_block=state_block(fmt))
        self.keys={p:12345+7919*i for i,p in enumerate(p for g in self.param_groups for p in g['params'])}

    def _state_for(self,p,g):
        if p.numel()<4096 or not g['compact']:
            return super()._state_for(p,g)
        _,vb,mb=FORMATS[g['format']]
        s=self.state[p]
        if 'm' not in s:
            n=p.numel()
            s.update(step=0,m=torch.zeros((n+1)//2 if mb==4 and vb!=4 else n,device=p.device,dtype=torch.uint8 if mb==4 else torch.int8),
                     ms=torch.zeros(triton.cdiv(n,64 if mb==4 else g["state_block"]),device=p.device,dtype=torch.float32),
                     vs=torch.zeros(triton.cdiv(n,64 if vb==4 else g["state_block"]),device=p.device,dtype=torch.float32))
            if vb!=4 or mb!=4:
                s['v']=torch.zeros(triton.cdiv(n,2) if vb==4 else n,device=p.device,
                                   dtype=torch.uint8 if vb<16 else torch.float16)
        return s

    def _update_param(self,p,g,pack=None,sparse=None):
        n=p.numel()
        if n<4096 or not g['compact']:
            return super()._update_param(p,g,pack)
        master,vb,mb=FORMATS[g['format']]
        if p.dtype==torch.float32: master=32
        group,warps=g['tiles'],g['warps']
        packed_next=getattr(p,'_v6_pack',None)
        if packed_next is not None:
            group,warps=16,g['pack_warps']
        width=g["state_block"]
        tiles=triton.cdiv(n,width)
        s=self._state_for(p,g)
        dyn=getattr(self,'_graph_control',None)
        t=s['step']+1
        b1,b2=g['betas']
        cache=g['cache_scale'] and getattr(p,'_v4_request_scale',False)
        partial=torch.empty(tiles,device=p.device,dtype=torch.float32) if cache else p
        mma=pack is not None and len(pack)==7
        if mma and (p.grad is not None or dyn is None or not cache):
            ad,bd,asc,bsc,sa,sb,_=pack
            raw=torch._scaled_mm(ad.view(torch.float4_e2m1fn_x2),bd.t().view(torch.float4_e2m1fn_x2),
                                 asc.view(torch.float8_e4m3fn),bsc.view(torch.float8_e4m3fn),
                                 out_dtype=torch.bfloat16,use_fast_accum=False)
            pack=(raw,sa,sb)
            mma=False
        with torch.cuda.device(p.device):
            if mma:
                from .mma_update import fused_update
                ad,bd,asc,bsc,sa,sb,contraction=pack
                fused_update[(p.shape[0]//16,p.shape[1]//width)](
                    ad,bd,asc,bsc,sa,sb,p,s['m'],s['v'],s['ms'],s['vs'],s.get('tail',p),partial,dyn,
                    p.shape[1],contraction,g['lr'],1-g['lr']*g['weight_decay'],b1,b2,g['eps'],self.keys[p],
                    master,g['mma_warps'],width,num_warps=g['mma_warps'],enable_fp_fusion=False)
                self.fused_mma_calls=getattr(self,'fused_mma_calls',0)+1
            else:
                grad,sa,sb=pack if pack is not None else (p.grad,p,p)
                scaled=pack is not None and sa is not None
                if sparse is not None and (grad is None or p.ndim!=2):
                    raise ValueError("Compact embedding updates require an active tied output projection")
                c,stride=(p.shape[-1],grad.shape[-1]) if scaled or sparse is not None else (1,1)
                slots,table=sparse if sparse is not None else (p,p)
                d,sf,dt,st,tag=(packed_next[:5] if packed_next is not None else (p,)*5)
                current=getattr(p,'_v5_graph_scale',p)
                update[(triton.cdiv(tiles,group),)](p,grad,s['m'],s.get('v',s['m']),s['ms'],s['vs'],s.get('tail',p),partial,
                    sa if scaled else p,sb if scaled else p,p.grad if p.grad is not None else p,dyn if dyn is not None else p,d,sf,dt,st,current,tag,getattr(p,"_v6_unit",p),slots,table,n,tiles,
                    g['lr'],1-g['lr']*g['weight_decay'],b1,b2,1/(1-b1**t),1/math.sqrt(1-b2**t),g['eps'],t,self.keys[p],
                    scaled,pack is not None and p.grad is not None,cache,dyn is not None,c,stride,master,vb,mb==4,g["format"].endswith("_sqrt"),packed_next is not None,p.shape[0] if packed_next is not None else 1,p.shape[1] if packed_next is not None else 1,
                    group,width,table.shape[0] if sparse is not None else 0,num_warps=warps,enable_fp_fusion=False)
            torch.autograd.graph.increment_version(p)
            s['step']=t
            if cache:
                scale=getattr(p,'_v5_graph_scale',None)
                if scale is None: scale=torch.empty((),device=p.device,dtype=torch.float32)
                if tiles>65536:
                    blocks=triton.cdiv(tiles,1024)
                    tmp=torch.empty(blocks,device=p.device,dtype=torch.float32)
                    _partial_max[(blocks,)](partial,tmp,tiles,1024)
                    partial,tiles=tmp,blocks
                _finish_scale[(1,)](partial,scale,tiles,triton.next_power_of_2(tiles))
                p._v4_scale=(p._version,p.data_ptr(),torch.cuda.current_stream(p.device).cuda_stream,scale)

    @torch.no_grad()
    def step(self,closure=None):
        self._require_eager()
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        active=[(p,g) for g in self.param_groups for p in g['params'] if p.grad is not None]
        if any(not p.is_contiguous() or p.grad.is_sparse or not p.grad.is_contiguous() for p,g in active):
            raise ValueError('Dense contiguous parameters and gradients required')
        for p,g in active: self._update_param(p,g)
        return loss

    def backward_step(self,loss,*,audit=False):
        """Single eager backward + update. Call zero_grad(set_to_none=True) first.

        Leaves confirmed FP4 weight gradients in GEMM output format until update.
        Other gradient contributions (e.g. tied embeddings) are added in FP32.
        """
        self._require_eager()
        params = [(p,g) for g in self.param_groups for p in g["params"]]
        pending = {}
        owners = [p for p,_ in params if getattr(p,"_v4_request_scale",False) or getattr(p,"_v6_defer_dense",False)]
        if any(p.grad is not None for p,_ in params):
            raise RuntimeError("backward_step requires cleared gradients")
        if any(hasattr(p,"_v4_pending") or p._backward_hooks or p._post_accumulate_grad_hooks for p in owners):
            raise RuntimeError("Gradient hooks or nested backward_step require ordinary backward/step")
        try:
            for p in owners:
                p._v4_pending = pending
            loss.backward()
        finally:
            for p in owners:
                del p._v4_pending
        active = [(p,g) for p,g in params if p in pending or p.grad is not None]
        for p,_ in active:
            if (not p.is_cuda or p.dtype not in (torch.float32,torch.float16,torch.bfloat16,torch.float8_e4m3fn) or not p.is_contiguous()
                    or (p.grad is not None and (p.grad.is_sparse or not p.grad.is_contiguous()))):
                raise ValueError("Expected contiguous CUDA FP32 parameters and dense gradients")
        if audit:
            checks = [torch.isfinite(p.grad).all() for p,_ in active if p.grad is not None]
            checks.extend(torch.isfinite(t).all() for pack in pending.values() for t in pack if torch.is_tensor(t))
            if not torch.stack(checks).all().item():
                raise RuntimeError("Nonfinite gradient before update")
        with torch.no_grad():
            for p,g in active:
                self._update_param(p,g,pending.pop(p,None),pending.pop((p,"embedding"),None))

    def capture_streaming_backward(self,loss):
        """Record parameter-local updates once, when AccumulateGrad is finished.

        Requires a fixed graph with non-aliased parameter storages. Multiple uses
        of the *same* Parameter, e.g. a tied head, share one accumulator.
        """
        if not torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Streaming updates are available only during graph capture")
        params=[(p,g) for g in self.param_groups for p in g["params"]]
        storages=[p.untyped_storage().data_ptr() for p,g in params]
        if len(set(storages))!=len(storages):
            raise ValueError("Distinct optimizer entries may not alias storage")
        if any(p.grad is not None or p._backward_hooks or p._post_accumulate_grad_hooks for p,g in params):
            raise ValueError("Clear gradients and remove gradient hooks before capture")
        pending={}
        completed=set()
        handles=[]
        nodes=[]
        owners=[p for p,g in params if getattr(p,"_v4_request_scale",False) or getattr(p,"_v6_defer_dense",False)]
        def ready(p,group):
            def update(grad_inputs,grad_outputs):
                pack=pending.pop(p,None)
                if p in completed or (pack is None and p.grad is None):
                    raise RuntimeError("Unexpected leaf gradient during capture")
                with torch.no_grad():
                    self._update_param(p,group,pack,pending.pop((p,"embedding"),None))
                p.grad=None
                completed.add(p)
            return update
        try:
            for p in owners:
                p._v4_pending=pending
            for p,group in params:
                node=torch.autograd.graph.get_gradient_edge(p).node
                nodes.append(node)
                handles.append(node.register_hook(ready(p,group)))
            loss.backward()
            if pending or len(completed)!=len(params):
                raise RuntimeError("Every parameter must receive its complete gradient once")
        finally:
            for handle in handles:
                handle.remove()
            for p in owners:
                del p._v4_pending

    def load_state_dict(self,state_dict):
        if any(a['format']!=b['format'] for a,b in zip(self.param_groups,state_dict['param_groups'])):
            raise ValueError('Checkpoint and model storage formats differ')
        return super().load_state_dict(state_dict)

def configure(model,mode,fmt,*,lr=1e-4,tiles=4,warps=4,prepack=False,pack_warps=8,remat=False,direct=False,mma_k=0,mma_warps=8,compact_embedding=False,act_pair=False,fast_scale=False):
    """Set up a fresh contiguous CUDA FP32 model for dense BF16 or native NVFP4."""
    if mode not in ('bf16','fp4') or fmt not in FORMATS:
        raise ValueError('Unknown compute mode or storage format')
    if tiles not in (1,2,4,8,16) or warps not in (4,8,16) or mma_warps not in (4,8):
        raise ValueError('Unsupported optimizer launch configuration')
    if mma_k and (mode!='fp4' or (fmt not in ('half16_v8','split24_v8') and 'v8b' not in fmt) or prepack):
        raise ValueError('Native Wgrad fusion requires an eligible v8 NVFP4 format without prepacking')
    if prepack and remat: raise ValueError('Choose one weight encoding strategy')
    if prepack and FORMATS[fmt][0]==8: raise ValueError('Prepacking is not enabled for FP8 storage')
    if direct and FORMATS[fmt][0] not in (8,16,17):
        raise ValueError('Direct CPU conversion requires an FP16, BF16 or FP8 master storage format')
    params=list(model.parameters())
    if any((p.device.type!=('cpu' if direct else 'cuda')) or p.dtype!=torch.float32 or not p.is_contiguous() for p in params):
        raise ValueError('Expected a fresh contiguous FP32 model on CPU for direct conversion, or CUDA otherwise')
    if len({p.untyped_storage().data_ptr() for p in params})!=len(params):
        raise ValueError('Distinct parameters must not share storage')
    counts=Counter(id(p) for _,p in model.named_parameters(remove_duplicate=False))
    if FORMATS[fmt][0]==8:
        from .master8 import unit_for
        for layer in model.modules():
            if isinstance(layer,nn.Linear) and counts[id(layer.weight)]==1 and layer.weight.numel()>=4096:
                layer.register_buffer('_v6_unit_buffer',torch.tensor(unit_for(layer.weight),device=layer.weight.device,dtype=torch.float32))
    if FORMATS[fmt][0]==17:
        bounds=[torch.aminmax(m.weight.detach()) for m in model.modules() if isinstance(m,nn.Linear)
                and counts[id(m.weight)]==1 and m.weight.numel()>=4096]
        if bounds:
            extrema=torch.stack([v for pair in bounds for v in pair])
            if not ((extrema.abs()<=65504).all().item()):
                raise ValueError('The initial weight range is not representable in FP16')
    if direct:
        dtype=torch.float16 if FORMATS[fmt][0]==17 else torch.bfloat16
        for layer in model.modules():
            if isinstance(layer,nn.Linear) and counts[id(layer.weight)]==1 and layer.weight.numel()>=4096:
                if FORMATS[fmt][0]==8:
                    layer.weight.data=(layer.weight.detach()/layer._v6_unit_buffer).to(torch.float8_e4m3fn)
                else:
                    layer.weight.data=layer.weight.detach().to(dtype)
        model.cuda()
        params=list(model.parameters())
        new_counts=Counter(id(p) for _,p in model.named_parameters(remove_duplicate=False))
        if sorted(new_counts.values())!=sorted(counts.values()):
            raise RuntimeError('Device transfer changed parameter sharing')
        counts=new_counts
        if mode=='fp4':
            from .nvfp4_fast_v2 import ensure_verified_modes,get_signs
            ensure_verified_modes()
            for layer in model.modules():
                if isinstance(layer,nn.Linear):
                    if type(layer) is not nn.Linear: raise TypeError('Custom linear layer')
                    layer.__class__=Linear6
                    layer.register_buffer('_nvfp4_signs',get_signs(layer.weight.device),persistent=False)
                    layer._nvfp4_save_packed=True
                    layer.weight._v4_request_scale=True
    elif mode=='fp4': enable_nvfp4(model)
    opt=AdamW6(params,fmt,lr=lr,tiles=tiles,warps=warps)
    opt.param_groups[0].update(prepack=prepack,pack_warps=pack_warps,mma_warps=mma_warps)
    master=FORMATS[fmt][0]
    for layer in model.modules():
        if not isinstance(layer,nn.Linear): continue
        p=layer.weight
        if mode=='fp4' and counts[id(p)]==1 and p.shape[0]%128==0 and p.shape[1]%512==0 and p.numel()<2**31:
            p._v6_packable=True
            if mma_k: p._v6_mma_k=mma_k
            layer.__class__=Linear6
        if fast_scale and mode=='fp4':
            layer.__class__=Linear6
            p._v6_fast_scale=True
        if act_pair and mode=='fp4':
            layer.__class__=Linear6
            p._v6_act_pair=True
        if remat and mode=='fp4':
            layer.__class__=Linear6
            p._v6_remat=True
        if master!=32 and counts[id(p)]==1 and p.numel()>=4096:
            if master==8:
                if not direct:
                    p.data=(p.detach()/layer._v6_unit_buffer).to(torch.float8_e4m3fn)
                p._v6_unit=layer._v6_unit_buffer
            elif not direct:
                hi=torch.empty_like(p,dtype=torch.float16 if master==17 else torch.bfloat16)
                tail=torch.empty_like(p,dtype=torch.int8) if master==24 else hi
                split_init[(triton.cdiv(p.numel(),512),)](p,hi,tail,p.numel(),master==24,512)
                p.data=hi
                if master==24: opt.state[p]['tail']=tail
            if isinstance(layer,NVFP4LinearV5): layer.__class__=Linear6
            elif master in (8,17):
                from .dense6 import DenseStoredLinear
                layer.__class__=DenseStoredLinear
                p._v6_defer_dense=True
    if compact_embedding:
        from .embed6 import configure_tied
        configure_tied(model,mode)
    return model,opt
