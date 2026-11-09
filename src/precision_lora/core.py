"""LoRA with unchanged floating-point bases or prepacked native NVFP4 bases.

The preserve path imports only PyTorch. Both A and B are trainable FP32
parameters; ordinary optimizers, accumulation and gradient clipping work.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from .packed import PackedNVFP4Linear


class _FrozenFP4(nn.Module):
    def __init__(self, linear, seed):
        super().__init__()
        from nvfp4_training._runtime import load_runtime
        from nvfp4_training._backend.nvfp4_fast_v2 import quantize, get_signs
        from nvfp4_training._backend.nvfp4_v5 import graph_extension
        load_runtime()[2].ensure_verified_modes()
        graph_extension()  # compile before any capture
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.source_dtype = linear.weight.dtype
        self.register_buffer("signs", get_signs(linear.weight.device).clone())
        self.register_buffer("sr_seed", torch.tensor(seed, device=linear.weight.device, dtype=torch.int32))
        # Each contraction direction has its own RHT and 1x16 scales. Transposing
        # packed nibbles would be mathematically wrong.
        w = linear.weight.detach()
        if w.dtype == torch.float16:
            w = w.float()
        for prefix, source in (("forward", w), ("backward", w.t())):
            packed = quantize(source, signs=self.signs, rht=True)
            for key in ("data", "scale", "pts"):
                self.register_buffer(prefix + "_" + key, getattr(packed, key))
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach())

    def multiply(self, x, backward=False):
        from nvfp4_training._backend.nvfp4_fast_v2 import quantize, Packed, round128, _STATS
        from nvfp4_training._backend.nvfp4_v5 import graph_extension
        from nvfp4_training._backend.scale6 import matmul
        prefix = "backward" if backward else "forward"
        if backward:
            x = x.contiguous()
            d, s, p = graph_extension().quantize(
                x, self.signs, round128(x.shape[0]), round128(x.shape[1]),
                True, True, 0, None, False, self.sr_seed)
            self.sr_seed.add_(1)  # GPU counter changes on every graph replay
            a = Packed(d, s, p, x.dtype)
            _STATS["sr_quant"] += 1
            _STATS["rht_quant"] += 1
            _STATS["quant"] += 1
        else:
            a = quantize(x, signs=self.signs, rht=True)
        b = Packed(*(getattr(self, prefix + "_" + key) for key in ("data", "scale", "pts")), x.dtype)
        return matmul(a, b, rows=x.shape[0],
                      cols=self.in_features if backward else self.out_features,
                      output_dtype=x.dtype, bias=None if backward else self.bias)


class _LoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, base, scale):
        shape = x.shape
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        if isinstance(base, _FrozenFP4) and dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("NVFP4 LoRA requires BF16 or FP32 compute; use BF16 autocast")
        if isinstance(base, PackedNVFP4Linear) and dtype != torch.bfloat16:
            raise TypeError("Prepacked NVFP4 LoRA requires BF16 compute; use BF16 autocast")
        # Explicit casts keep adapter state FP32 and make backward independent of
        # whichever autocast context surrounds loss.backward().
        with torch.autocast(x.device.type, enabled=False):
            v = x.reshape(-1, shape[-1]).to(dtype)
            ac, bc = a.to(dtype), b.to(dtype)
            h = F.linear(v, ac)
            if isinstance(base, (_FrozenFP4, PackedNVFP4Linear)):
                y = base.multiply(v)
                weight = v.new_empty(0)
            else:
                weight = base.weight
                y = F.linear(v, weight.to(dtype), None if base.bias is None else base.bias.to(dtype))
            # The base result is private to this Function. Accumulate directly;
            # no full-sized adapter-output and addition buffers.
            y.addmm_(h, bc.t(), alpha=scale)
        ctx.save_for_backward(v, h, ac, bc, weight)
        ctx.base, ctx.scale = base, scale
        ctx.shape, ctx.input_dtype = shape, x.dtype
        ctx.a_dtype, ctx.b_dtype = a.dtype, b.dtype
        return y.reshape(*shape[:-1], b.shape[0])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        x, h, a, b, weight = ctx.saved_tensors
        with torch.autocast(grad.device.type, enabled=False):
            g = grad.reshape(-1, b.shape[0]).to(x.dtype)
            dh = g @ b
            da = torch.mm(dh.t(), x)
            db = torch.mm(g.t(), h)
            if ctx.scale != 1:
                da.mul_(ctx.scale); db.mul_(ctx.scale)
            da, db = da.to(ctx.a_dtype), db.to(ctx.b_dtype)
            dx = None
            if ctx.needs_input_grad[0]:
                dx = ctx.base.multiply(g, backward=True) if isinstance(ctx.base, (_FrozenFP4, PackedNVFP4Linear)) else g @ weight.to(x.dtype)
                dx.addmm_(dh, a, alpha=ctx.scale)
                dx = dx.reshape(ctx.shape).to(ctx.input_dtype)
        return dx, da, db, None, None


class LoRALinear(nn.Module):
    """Y = base(X) + (alpha/rank) * X A^T B^T; zero-initialized B."""
    def __init__(self, linear, rank, alpha, mode, seed):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.rank, self.alpha, self.mode = rank, float(alpha), mode
        self.base = _FrozenFP4(linear, seed) if mode == "nvfp4" else linear
        device = linear.device if isinstance(linear, PackedNVFP4Linear) else linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError("LoRA input feature count does not match the base")
        return _LoRAFunction.apply(x, self.lora_A, self.lora_B, self.base, self.alpha / self.rank)


def _checkpoint_forward(*args, original, **kwargs):
    from torch.utils.checkpoint import checkpoint
    if torch.is_grad_enabled():
        return checkpoint(original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
    return original(*args, **kwargs)


def enable_lora(model, *, mode="preserve", rank=16, alpha=None, target_modules=None,
                checkpoint_modules=()):
    """Freeze the base, replace selected nn.Linear and return the model.

    mode='preserve': original base Parameter objects/dtypes/values are retained.
    mode='nvfp4': frozen bases are replaced by two independently RHT-packed FP4
    orientations; no full-size master weight remains for selected projections.
    mode='packed_nvfp4': adapt explicit PackedNVFP4Linear instances while
    retaining the checkpoint payload and its chosen backward memory policy.
    target_modules: exact paths or final component names; default targets all
    untied Linear layers except head/lm_head/output_layer. Root Linear is valid.
    checkpoint_modules: exact non-overlapping parent block paths, optional.
    """
    if not isinstance(model, nn.Module) or any(isinstance(m, LoRALinear) for m in model.modules()) or hasattr(model, "nvfp4_optimizer"):
        raise ValueError("Pass an unconverted nn.Module")
    if mode not in ("preserve", "nvfp4", "packed_nvfp4"):
        raise ValueError("mode must be 'preserve', 'nvfp4' or 'packed_nvfp4'")
    if type(rank) is not int or rank < 1:
        raise ValueError("rank must be a positive integer")
    alpha = rank if alpha is None else alpha
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if isinstance(target_modules, str) or isinstance(checkpoint_modules, str):
        raise TypeError("Pass a list of module names, not one string")
    targets = None if target_modules is None else set(target_modules)
    paths = dict(model.named_modules(remove_duplicate=False))
    checkpoints = list(checkpoint_modules)
    if any(p not in paths or not p for p in checkpoints) or any(
        a == b or a.startswith(b + ".") or b.startswith(a + ".") for i,a in enumerate(checkpoints) for b in checkpoints[:i]):
        raise ValueError("checkpoint_modules must be distinct, non-overlapping existing block paths")
    counts = {}
    for _, p in model.named_parameters(remove_duplicate=False):
        counts[id(p)] = counts.get(id(p), 0) + 1
    for _, p in model.named_buffers(remove_duplicate=False):
        counts[id(p)] = counts.get(id(p), 0) + 1
    selected = {}
    matched = set()
    for name, module in paths.items():
        if not isinstance(module, (nn.Linear, PackedNVFP4Linear)):
            continue
        is_packed = isinstance(module, PackedNVFP4Linear)
        weight = module.packed if is_packed else module.weight
        if targets is None:
            choose = is_packed == (mode == "packed_nvfp4") and counts[id(weight)] == 1 and name.rsplit(".",1)[-1] not in ("head", "lm_head", "output_layer")
        else:
            keys = {name, name.rsplit(".",1)[-1]} & targets
            choose = bool(keys)
            matched.update(keys)
        if choose:
            if is_packed != (mode == "packed_nvfp4"):
                raise TypeError("packed_nvfp4 targets require PackedNVFP4Linear; floating-point targets use preserve or nvfp4")
            if counts[id(weight)] != 1:
                raise ValueError(f"Target {name!r} shares a weight; leave tied/aliased projections frozen")
            if not is_packed and type(module) is not nn.Linear:
                raise TypeError(f"Custom Linear at {name!r} requires explicit integration")
            if not is_packed and module.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                raise TypeError(f"Unsupported base storage at {name!r}: {module.weight.dtype}")
            if mode == "nvfp4" and (module.weight.device.type != "cuda" or module.weight.dtype == torch.float64 or counts[id(module.weight)] != 1):
                raise ValueError("NVFP4 targets must be untied FP16/BF16/FP32 CUDA Linear layers")
            selected[name] = module
    if not selected or (targets is not None and matched != targets):
        raise ValueError(f"No matching Linear, or unmatched targets: {targets - matched if targets else targets}")
    if any(p in selected for p in checkpoints):
        raise ValueError("Checkpoint a parent block, not an individual LoRA Linear")
    # Validate before altering requires_grad or module structure.
    if mode == "nvfp4":
        from nvfp4_training._runtime import load_runtime
        load_runtime()[2].ensure_verified_modes()
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    converted = {}
    selected_names = sorted(selected)
    # Drop references to each old projection immediately after replacement.
    # Keeping an inventory of original modules alive would retain the entire
    # uncompressed model until packing finishes, inflating conversion peak.
    for name in list(selected):
        module = selected.pop(name)
        if id(module) not in converted:
            converted[id(module)] = LoRALinear(module, rank, alpha, mode, len(converted) * 100003)
        replacement = converted[id(module)]
        replacement.train(module.training)
        if not name:
            model = replacement
        else:
            parent, _, leaf = name.rpartition(".")
            paths[parent]._modules[leaf] = replacement
        paths[name] = replacement
    for name in checkpoints:
        module = paths[name]
        module.forward = partial(_checkpoint_forward, original=module.forward)
    model.lora_config = dict(mode=mode, rank=rank, alpha=float(alpha), targets=selected_names, checkpoint_modules=checkpoints)
    if mode == "packed_nvfp4":
        model.lora_config["packed_layouts"] = {n: dict(shape=(paths[n].in_features, paths[n].out_features),
            layout=paths[n].base.layout, rht_seed=paths[n].base.rht_seed) for n in selected_names}
    return model


def lora_parameters(model):
    """The exact trainable adapter parameters, for any ordinary optimizer."""
    return [p for m in model.modules() if isinstance(m, LoRALinear) for p in (m.lora_A, m.lora_B)]


def prepare_lora(model, *, lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, **options):
    """Return (converted model, fused AdamW on FP32 adapters only)."""
    if not all(math.isfinite(v) for v in (lr, eps, weight_decay)) or lr < 0 or eps <= 0 or weight_decay < 0:
        raise ValueError("AdamW requires finite lr >= 0, eps > 0, weight_decay >= 0")
    if len(betas) != 2 or not all(0 <= b < 1 for b in betas):
        raise ValueError("betas must contain two values in [0,1)")
    model = enable_lora(model, **options)
    params = lora_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                 fused=all(p.is_cuda for p in params))
    return model, optimizer


def save_lora(path, model, *, optimizer=None, base_id=None):
    """Atomic adapter-only checkpoint; frozen bases are never duplicated."""
    layers = {name: m for name,m in model.named_modules() if isinstance(m, LoRALinear)}
    payload = dict(schema=1, config=model.lora_config, base_id=base_id,
                   adapters={n: dict(A=m.lora_A.detach().cpu(), B=m.lora_B.detach().cpu(),
                       sr=m.base.sr_seed.cpu() if isinstance(m.base, (_FrozenFP4, PackedNVFP4Linear)) else None) for n,m in layers.items()},
                   optimizer=None if optimizer is None else optimizer.state_dict(),
                   cpu_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".lora-", suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_lora(path, model, *, optimizer=None, base_id=None, restore_rng=True):
    """Restore into the same base and recipe. base_id can enforce a model revision."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    def recipe(config):return {k:v for k,v in config.items() if k != 'checkpoint_modules'}
    if saved.get("schema") != 1 or recipe(saved.get("config",{})) != recipe(model.lora_config) or (base_id is not None and saved.get("base_id") != base_id):
        raise ValueError("LoRA recipe or base_id mismatch")
    layers = {name: m for name,m in model.named_modules() if isinstance(m, LoRALinear)}
    if layers.keys() != saved["adapters"].keys() or any(
        m.lora_A.shape != saved["adapters"][n]["A"].shape or m.lora_B.shape != saved["adapters"][n]["B"].shape for n,m in layers.items()):
        raise ValueError("Adapter names or shapes do not match")
    if optimizer is not None and saved["optimizer"] is None:
        raise ValueError("Checkpoint has no optimizer state")
    with torch.no_grad():
        for n,m in layers.items():
            item = saved["adapters"][n]
            m.lora_A.copy_(item["A"]); m.lora_B.copy_(item["B"])
            if isinstance(m.base, (_FrozenFP4, PackedNVFP4Linear)):
                m.base.sr_seed.copy_(item["sr"])
    if optimizer is not None:
        optimizer.load_state_dict(saved["optimizer"])
    if restore_rng:
        torch.set_rng_state(saved["cpu_rng"])
        if saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved.get("base_id")


@torch.no_grad()
def merge_lora(model):
    """Merge into the original storage dtype and remove adapters (preserve only).

    Merging necessarily rounds the sum to the base dtype; it changes base values.
    Keep an adapter checkpoint and the original base to resume training.
    """
    layers = [(n,m) for n,m in model.named_modules() if isinstance(m, LoRALinear)]
    if any(m.mode != "preserve" for _,m in layers):
        raise ValueError("Merge requires preserve mode and the original floating-point base")
    ids = [id(m.base.weight) for _,m in layers]
    if len(set(ids)) != len(ids):
        raise ValueError("Cannot merge distinct adapters into a shared base weight")
    for name,m in layers:
        w = m.base.weight
        accumulation_dtype = torch.float64 if w.dtype == torch.float64 else torch.float32
        # Bound FP32 scratch instead of materializing a whole FP32 delta weight.
        for start in range(0, m.out_features, 256):
            delta = m.lora_B[start:start+256].to(accumulation_dtype) @ m.lora_A.to(accumulation_dtype)
            w[start:start+256].copy_((w[start:start+256].to(accumulation_dtype) + delta * (m.alpha/m.rank)).to(w.dtype))
        if not name:
            model = m.base
        else:
            parent, _, leaf = name.rpartition(".")
            model.get_submodule(parent)._modules[leaf] = m.base
    if hasattr(model, "lora_config"):
        del model.lora_config
    return model
