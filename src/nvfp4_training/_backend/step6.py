"""Fixed-shape single-GPU replay with per-update bias correction and SR counters."""
import math
from types import SimpleNamespace
import torch
from . import nvfp4_fast_v2 as rng
from .nvfp4_v5 import graph_extension, NVFP4LinearV5
from .optimizer_v5 import AdamW

class StaticStep:
    """Own a fixed eager training graph for a bounded number of optimizer steps.

    Warm up using real eager updates before construction. Keep model structure,
    optimizer group and batch layout fixed. Close before eager training/checkpoint.
    """
    def __init__(self,model,optimizer,compute,x,y,steps,*,stream_updates=False,from_start=False):
        if not isinstance(optimizer,AdamW) or len(optimizer.param_groups)!=1 or steps<1:
            raise ValueError("One warmed v5 AdamW group and a positive replay count required")
        self.model,self.opt,self.steps,self.index=model,optimizer,steps,0
        self.group=optimizer.param_groups[0]
        self.params=tuple(self.group['params'])
        self.config=self._config()
        self.device=x.device
        if (self.device.type!='cuda' or y.device!=self.device or x.requires_grad or y.requires_grad
                or any(p.device!=self.device or not p.requires_grad for p in self.params)):
            raise ValueError("StaticStep requires one device, trainable parameters and data-only inputs")
        if {id(p) for p in model.parameters() if p.requires_grad}!={id(p) for p in self.params}:
            raise ValueError("Optimizer must own every trainable parameter of this model")
        if from_start:
            if any(optimizer.state.get(p,{}).get("step",0) for p in self.params):
                raise ValueError("from_start requires a new optimizer")
            for p in self.params:
                optimizer._state_for(p,self.group)
        counts={optimizer.state.get(p,{}).get('step',0) for p in self.params}
        if len(counts)!=1 or (not from_start and min(counts)<1):
            raise ValueError("Warm up all parameters together before capture")
        self.start=counts.pop()
        self.training=model.training
        self.stream=torch.cuda.current_stream(self.device)
        self.closed=False
        self.x,self.y=x.detach().clone(),y.detach().clone()
        self.ctrl=torch.empty(4,device=x.device,dtype=torch.float32)
        clock=SimpleNamespace(sr_calls=0,seed=self.ctrl[2:3].view(torch.int32))
        owners=[p for p in self.params if getattr(p,'_v4_request_scale',False)]
        fp4_layers=[m for m in model.modules() if isinstance(m,NVFP4LinearV5)]
        if owners and not fp4_layers:
            raise ValueError("Use the v5 converter for replay-safe SR")
        if owners:
            rng.extension()
            graph_extension()
        if getattr(optimizer,'_graph_owner',None) is not None:
            raise RuntimeError("Optimizer is already captured")
        self.packs={}
        if self.group.get('prepack',False) and self.group['cache_scale']:
            from .prepack import buffers
            self.packs={p:buffers(p) for p in owners if getattr(p,'_v6_packable',False)}
        self.scales={}
        for p in owners:
            cached=getattr(p,'_v4_scale',None)
            if self.group['cache_scale']:
                if cached is None or cached[:3]!=(p._version,p.data_ptr(),self.stream.cuda_stream):
                    if not from_start:
                        raise ValueError("Warm up on this stream after the last parameter edit")
                    from .master8 import decoded
                    with torch.no_grad():
                        value=decoded(p).abs().amax().float().clamp_min(1e-12)/2688.
                    self.scales[p]=value
                else:
                    self.scales[p]=cached[3]
        optimizer.zero_grad(set_to_none=True)
        self.graph=torch.cuda.CUDAGraph()
        capture_stream=torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(self.stream)
        if from_start:
            # cuBLAS creates its first handle lazily; that allocation cannot run
            # inside stream capture. This initializes the handle, not the model.
            with torch.cuda.stream(capture_stream):
                torch.cuda.current_blas_handle()
                # The autograd engine has its own thread-local cuBLAS handle.
                tiny=torch.ones((16,16),device=self.device,dtype=torch.bfloat16,requires_grad=True)
                (tiny@tiny).sum().backward()
                del tiny
        try:
            for p,cache in self.packs.items():
                p._v6_pack=cache
            for p in owners:
                p._v5_clock=clock
            for p,scale in self.scales.items():
                p._v5_graph_scale=scale
            optimizer._graph_control=self.ctrl
            with torch.cuda.graph(self.graph,stream=capture_stream):
                self.loss=compute(self.x,self.y)
                if stream_updates:
                    optimizer.capture_streaming_backward(self.loss)
                else:
                    optimizer.backward_step(self.loss)
            self.loss=self.loss.detach()
            if any(optimizer.state[p]['step']!=self.start+1 for p in self.params):
                raise RuntimeError("Every owned parameter must participate in the captured step")
        finally:
            for p in self.packs:
                del p._v6_pack
            for p in owners:
                del p._v5_clock
            for p in self.scales:
                del p._v5_graph_scale
            del optimizer._graph_control
            for p in self.params:
                optimizer.state[p]['step']=self.start
        self.sr_calls=clock.sr_calls
        if owners and self.sr_calls==0:
            raise RuntimeError("No dynamic SR in FP4 backward")
        b1,b2=self.group['betas']
        schedule=torch.empty((steps,4),dtype=torch.float32,device='cpu')
        seed0=rng._SEED
        for i in range(steps):
            t=self.start+i+1
            schedule[i,0]=1/(1-b1**t)
            schedule[i,1]=1/math.sqrt(1-b2**t)
        bases=(seed0+torch.arange(steps,dtype=torch.int64)*self.sr_calls).to(torch.int32)
        schedule[:,2]=bases.view(torch.float32)
        schedule[:,3]=torch.arange(self.start+1,self.start+steps+1,dtype=torch.int32).view(torch.float32)
        self.schedule=schedule.to(x.device)
        self.seed0=seed0
        self.seed_end=(seed0+steps*self.sr_calls)&0xffffffff
        rng._SEED=self.seed_end
        optimizer._graph_owner=self
        self.versions=tuple(p._version for p in self.params)
        self.pointers=tuple(p.data_ptr() for p in self.params)

    def _config(self):
        return tuple(self.group.get(k) for k in ('lr','betas','eps','weight_decay','compact','cache_scale','format','tiles','warps','prepack','pack_warps','state_block','mma_warps'))

    def step(self,x,y):
        if (self.model.training!=self.training or len(self.opt.param_groups)!=1
                or self.opt.param_groups[0] is not self.group):
            raise RuntimeError("Captured training mode and parameter group must stay fixed")
        if self.closed or self.index>=self.steps:
            raise RuntimeError("Capture closed or schedule exhausted")
        if torch.cuda.current_stream(self.device)!=self.stream or self._config()!=self.config:
            raise RuntimeError("Captured optimizer settings and stream must stay fixed")
        if (tuple(p._version for p in self.params)!=self.versions
                or tuple(p.data_ptr() for p in self.params)!=self.pointers):
            raise RuntimeError("Parameter edited outside captured step")
        for src,dst in ((x,self.x),(y,self.y)):
            if src.device!=dst.device or src.dtype!=dst.dtype or src.shape!=dst.shape or src.requires_grad:
                raise ValueError("Batch device, shape and dtype must match capture")
            dst.copy_(src)
        self.ctrl.copy_(self.schedule[self.index])
        self.graph.replay()
        torch.autograd.graph.increment_version(self.params)
        self.versions=tuple(p._version for p in self.params)
        self.index+=1
        return self.loss.detach()

    def close(self):
        if not self.closed:
            for i,p in enumerate(self.params):
                self.opt.state[p]['step']=self.start+self.index
                if p in self.scales and p._version==self.versions[i] and p.data_ptr()==self.pointers[i]:
                    p._v4_scale=(p._version,p.data_ptr(),self.stream.cuda_stream,self.scales[p])
                elif hasattr(p,'_v4_scale'):
                    del p._v4_scale
            if self.sr_calls and rng._SEED==self.seed_end:
                rng._SEED=(self.seed0+self.index*self.sr_calls)&0xffffffff
            del self.opt._graph_owner
            self.closed=True
            self.opt.zero_grad(set_to_none=True)
            self.graph.reset()

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()
