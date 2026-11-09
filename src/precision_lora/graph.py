"""Fixed-shape LoRA graph replay; warmup/capture do not count as updates."""
import torch
from functools import lru_cache
from .core import lora_parameters, _FrozenFP4
from .packed import PackedNVFP4Linear


@lru_cache(maxsize=None)
def _capture_stream(device):
    # Reuse cuBLAS workspaces too; a fresh stream per capture retains an extra
    # library workspace every time the user recaptures a batch shape.
    return torch.cuda.Stream(device=device)


class LoRAStep:
    def __init__(self, model, optimizer, loss_fn, *samples, warmup=3, max_grad_norm=None):
        if any(isinstance(m, PackedNVFP4Linear) for m in model.modules()):
            raise ValueError("packed_nvfp4 currently uses eager training; CUDA graph replay has not been validated")
        if type(warmup) is not int or warmup < 2:
            raise ValueError("Use at least two warmup steps")
        self.params = lora_parameters(model)
        if not self.params or any(not p.is_cuda or p.device != self.params[0].device for p in self.params):
            raise ValueError("CUDA LoRA model required")
        if not isinstance(optimizer, torch.optim.AdamW) or any(not g.get('fused') for g in optimizer.param_groups):
            raise ValueError("LoRA capture requires fused torch.optim.AdamW")
        if {id(p) for p in self.params} != {id(p) for g in optimizer.param_groups for p in g['params']}:
            raise ValueError("Optimizer must contain exactly the LoRA parameters")
        if any(not torch.is_tensor(t) or t.device != self.params[0].device for t in samples):
            raise ValueError("Pass fixed-shape CUDA tensor inputs")
        self.model, self.optimizer = model, optimizer
        self.capture_flags = [g.get('capturable',False) for g in optimizer.param_groups]
        for group in optimizer.param_groups:group['capturable'] = True
        self.inputs = [t.detach().clone() for t in samples]
        self.closed = False
        self.original_training = model.training
        self.seed_buffers = [m.sr_seed for m in model.modules() if isinstance(m, _FrozenFP4)]
        packed_buffers = {id(b) for m in model.modules() if isinstance(m, _FrozenFP4) for b in m.buffers()}
        self.other_buffers = [b for b in model.buffers() if id(b) not in packed_buffers]
        initial_params = [p.detach().clone() for p in self.params]
        initial_seeds = [p.clone() for p in self.seed_buffers]
        initial_buffers = [p.clone() for p in self.other_buffers]
        initial_rng = torch.cuda.get_rng_state()
        initial_cpu_rng = torch.get_rng_state()
        # Preserve an already-started optimizer as well as a fresh one.
        initial_state = {p:{k:v.clone() if torch.is_tensor(v) else v for k,v in optimizer.state.get(p,{}).items()} for p in self.params}

        def step():
            optimizer.zero_grad(set_to_none=False)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss = loss_fn(model,*self.inputs)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.params,max_grad_norm)
            optimizer.step()
            self.loss = loss.detach()

        @torch.no_grad()
        def restore():
            for p,value in zip(self.params,initial_params):p.copy_(value)
            for p,value in zip(self.seed_buffers,initial_seeds):p.copy_(value)
            for p,value in zip(self.other_buffers,initial_buffers):p.copy_(value)
            for p,state in initial_state.items():
                for k,v in optimizer.state[p].items():
                    if torch.is_tensor(v):
                        if k in state:v.copy_(state[k])
                        else:v.zero_()
            torch.cuda.set_rng_state(initial_rng)
            torch.set_rng_state(initial_cpu_rng)
        stream = _capture_stream(self.params[0].device)
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(stream):
                for _ in range(warmup):step()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            restore()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):step()
            restore()
            optimizer.zero_grad(set_to_none=False)
        except BaseException:
            torch.cuda.synchronize()
            restore()
            for group,flag in zip(optimizer.param_groups,self.capture_flags):group['capturable'] = flag
            raise

    def step(self, *batch):
        if self.closed:
            raise RuntimeError("LoRA graph is closed")
        if self.model.training != self.original_training:
            raise RuntimeError("Do not change train/eval mode during graph replay")
        if len(batch) != len(self.inputs) or any(a.shape != b.shape or a.dtype != b.dtype or a.device != b.device for a,b in zip(batch,self.inputs)):
            raise ValueError("Graph inputs must match the captured shape, dtype and device")
        for dst,src in zip(self.inputs,batch):dst.copy_(src)
        self.graph.replay()
        return self.loss.detach()

    def close(self):
        if not self.closed:
            torch.cuda.synchronize()
            self.closed = True
            self.graph.reset()
            self.optimizer.zero_grad(set_to_none=True)
            for group,flag in zip(self.optimizer.param_groups,self.capture_flags):group['capturable'] = flag

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


def capture_lora_step(model, optimizer, loss_fn, *samples, **options):
    return LoRAStep(model, optimizer, loss_fn, *samples, **options)
