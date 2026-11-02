"""Single-GPU AdamW with FP32 masters and block-scaled moment storage."""
import math
import torch
import triton
import triton.language as tl

BLOCK = 512

@triton.jit
def _update(P, G, M, V, MS, VS, PART, SA, SB, EXTRA, CONTROL, N, lr, decay, b1, b2, c1, c2, eps,
            COMPACT: tl.constexpr, AMAX: tl.constexpr, DEFER: tl.constexpr, ADD: tl.constexpr,
            DYNAMIC: tl.constexpr, COLS: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr):
    block = tl.program_id(0)
    i = block * B + tl.arange(0, B)
    mask = i < N
    p = tl.load(P+i, mask, other=0.)
    if DEFER:
        index = i if STRIDE == COLS else (i // COLS)*STRIDE + i % COLS
        alpha = (tl.load(SA)*tl.load(SB)).to(tl.bfloat16).to(tl.float32)
        raw = tl.load(G+index, mask, other=0.).to(tl.float32)
        g = (raw*alpha).to(tl.bfloat16).to(tl.float32)
        if ADD:
            g += tl.load(EXTRA+i,mask,other=0.)
    else:
        g = tl.load(G+i, mask, other=0.)
    m = tl.load(M+i, mask, other=0).to(tl.float32)
    v = tl.load(V+i, mask, other=0).to(tl.float32)
    if COMPACT:
        m *= tl.load(MS+block)
        v *= tl.load(VS+block)
    m = b1*m + (1.-b1)*g
    v = b2*v + (1.-b2)*g*g
    if DYNAMIC:
        c1=tl.load(CONTROL)
        c2=tl.load(CONTROL+1)
    new_p = p*decay - (lr*c1)*m / (tl.sqrt(v)*c2 + eps)
    tl.store(P+i,new_p,mask)
    if COMPACT:
        mscale = tl.maximum(tl.max(tl.abs(m),0)/127., 1.e-30)
        vscale = tl.maximum(tl.max(v,0)/32768., 1.e-35)
        # Nearest-even INT8 for signed momentum, scaled IEEE FP16 for variance.
        rounded = tl.inline_asm_elementwise("cvt.rni.s32.f32 $0, $1;", constraints="=r,f",
                    args=[m/mscale], dtype=tl.int32, is_pure=True, pack=1)
        tl.store(M+i,tl.minimum(127,tl.maximum(-127,rounded)),mask)
        tl.store(V+i,v/vscale,mask)
        tl.store(MS+block,mscale)
        tl.store(VS+block,vscale)
    else:
        tl.store(M+i,m,mask)
        tl.store(V+i,v,mask)
    if AMAX:
        bound = tl.max(tl.where(mask,tl.abs(new_p.to(tl.bfloat16).to(tl.float32)),0.),0)
        tl.store(PART+block,bound)

@triton.jit
def _partial_max(PART, OUT, N:tl.constexpr, B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    tl.store(OUT+tl.program_id(0),tl.max(tl.load(PART+i,i<N,other=0.),0))

@triton.jit
def _finish_scale(PART,SCALE,N:tl.constexpr,B:tl.constexpr):
    i = tl.arange(0,B)
    amax = tl.max(tl.load(PART+i,i<N,other=0.),0)
    tl.store(SCALE,tl.maximum(amax,1.e-12)*(1./2688.))

class AdamW(torch.optim.Optimizer):
    """FP32 updates with compact moments for tensors of at least 4096 elements."""
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8,
                 weight_decay=0.01, *, compact=True, cache_scale=True):
        if not (lr>=0 and eps>0 and weight_decay>=0 and all(0<=b<1 for b in betas)):
            raise ValueError("Invalid AdamW hyperparameters")
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,
                                    compact=compact,cache_scale=cache_scale))

    @torch.no_grad()
    def step(self,closure=None):
        self._require_eager()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        active = [(p,g) for g in self.param_groups for p in g['params'] if p.grad is not None]
        for p,_ in active:
            if (not p.is_cuda or p.dtype!=torch.float32 or not p.is_contiguous()
                    or p.grad.is_sparse or not p.grad.is_contiguous() or p.numel()==0):
                raise ValueError("Expected nonempty dense contiguous CUDA FP32 parameters and gradients")
        for p,g in active:
            self._update_param(p,g)
        return loss

    def backward_step(self,loss,*,audit=False):
        """Single eager backward + update. Call zero_grad(set_to_none=True) first.

        Leaves confirmed FP4 weight gradients in GEMM output format until update.
        Other gradient contributions (e.g. tied embeddings) are added in FP32.
        """
        self._require_eager()
        params = [(p,g) for g in self.param_groups for p in g["params"]]
        pending = {}
        owners = [p for p,_ in params if getattr(p,"_v4_request_scale",False)]
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
            if (not p.is_cuda or p.dtype!=torch.float32 or not p.is_contiguous()
                    or (p.grad is not None and (p.grad.is_sparse or not p.grad.is_contiguous()))):
                raise ValueError("Expected contiguous CUDA FP32 parameters and dense gradients")
        if audit:
            checks = [torch.isfinite(p.grad).all() for p,_ in active if p.grad is not None]
            checks.extend(torch.isfinite(t).all() for pack in pending.values() for t in pack)
            if not torch.stack(checks).all().item():
                raise RuntimeError("Nonfinite gradient before update")
        with torch.no_grad():
            for p,g in active:
                self._update_param(p,g,pending.pop(p,None))

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
        owners=[p for p,g in params if getattr(p,"_v4_request_scale",False)]
        def ready(p,group):
            def update(grad_inputs,grad_outputs):
                pack=pending.pop(p,None)
                if p in completed or (pack is None and p.grad is None):
                    raise RuntimeError("Unexpected leaf gradient during capture")
                with torch.no_grad():
                    self._update_param(p,group,pack)
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

    def _state_for(self,p,g):
        n=p.numel()
        compact=g["compact"] and n>=4096
        tiles=triton.cdiv(n,BLOCK)
        s = self.state[p]
        if not s:
            s['step'] = 0
            s['m'] = torch.zeros_like(p,dtype=torch.int8 if compact else torch.float32)
            s['v'] = torch.zeros_like(p,dtype=torch.float16 if compact else torch.float32)
            if compact:
                for name in ('ms','vs'):
                    s[name] = torch.ones(tiles,device=p.device,dtype=torch.float32)
        return s

    def _update_param(self,p,g,pack=None):
        n = p.numel()
        tiles = triton.cdiv(n,BLOCK)
        compact = g['compact'] and n>=4096
        s = self._state_for(p,g)
        dynamic = getattr(self,"_graph_control",None)
        t = s['step']+1
        b1,b2 = g['betas']
        cache = (g['cache_scale'] and getattr(p,'_v4_request_scale',False))
        partial = torch.empty(tiles,device=p.device,dtype=torch.float32) if cache else p
        with torch.cuda.device(p.device):
            grad,sa,sb = pack if pack is not None else (p.grad,p,p)
            stride = grad.shape[-1] if pack is not None else 1
            cols = p.shape[-1] if pack is not None else 1
            _update[(tiles,)](p,grad,s['m'],s['v'],s.get('ms',p),s.get('vs',p),partial,sa,sb,p.grad if p.grad is not None else p,dynamic if dynamic is not None else p,n,
                g['lr'],1-g['lr']*g['weight_decay'],b1,b2,1/(1-b1**t),
                1/math.sqrt(1-b2**t),g['eps'],compact,cache,pack is not None,pack is not None and p.grad is not None,dynamic is not None,cols,stride,BLOCK,num_warps=4,enable_fp_fusion=False)
            torch.autograd.graph.increment_version(p)
            s['step'] = t
            if cache:
                scale = getattr(p,"_v5_graph_scale",None)
                if scale is None:
                    scale=torch.empty((),device=p.device,dtype=torch.float32)
                if tiles>65536:
                    blocks=triton.cdiv(tiles,1024)
                    smaller=torch.empty(blocks,device=p.device,dtype=torch.float32)
                    _partial_max[(blocks,)](partial,smaller,tiles,1024)
                    partial,tiles=smaller,blocks
                _finish_scale[(1,)](partial,scale,tiles,triton.next_power_of_2(tiles))
                p._v4_scale = (p._version,p.data_ptr(),torch.cuda.current_stream(p.device).cuda_stream,scale)

    def _require_eager(self):
        if getattr(self,"_graph_owner",None) is not None:
            raise RuntimeError("Close StaticStep before eager updates or checkpointing")

    def zero_grad(self,set_to_none=True):
        self._require_eager()
        return super().zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        self._require_eager()
        return super().state_dict()

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict normally casts all floating states to master
        # dtype. Restore encoded tensors from the checkpoint without recasting.
        self._require_eager()
        super().load_state_dict(state_dict)
        for saved,group in zip(state_dict['param_groups'],self.param_groups):
            for key,p in zip(saved['params'],group['params']):
                original = state_dict['state'].get(key)
                if original:
                    self.state[p] = {k:v.to(device=p.device,copy=True) if torch.is_tensor(v) else v
                                     for k,v in original.items()}
                if hasattr(p,'_v4_scale'):
                    del p._v4_scale
