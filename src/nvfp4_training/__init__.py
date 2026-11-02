"""Public single-GPU NVFP4 training interface for the local SM120 release."""
import math
import os
from pathlib import Path
import tempfile

from .profiles import list_profiles, profile_config
from ._runtime import load_runtime

__all__ = ["list_profiles", "prepare_training", "enable_nvfp4", "train_step",
           "capture_step", "save_checkpoint", "load_checkpoint", "check_finite"]


def prepare_training(model, *, profile="balanced", precision="fp4", remat=False,
                     lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """Convert a fresh FP32 model in place and return (model, matching AdamW)."""
    import torch
    from torch import nn

    cfg = profile_config(profile)
    if precision not in ("fp4", "bf16") or type(remat) is not bool:
        raise ValueError("precision must be 'fp4' or 'bf16', and remat must be a bool")
    if remat and precision == "bf16":
        raise ValueError("remat applies to NVFP4 transpose encoding; use remat=False for BF16")
    if not isinstance(betas, (tuple, list)) or len(betas) != 2 or not all(0 <= b < 1 for b in betas):
        raise ValueError("betas must contain two numbers in [0, 1)")
    if not (all(math.isfinite(v) for v in (lr, eps, weight_decay)) and lr >= 0 and eps > 0 and weight_decay >= 0):
        raise ValueError("AdamW requires finite lr >= 0, eps > 0 and weight_decay >= 0")
    if not isinstance(model, nn.Module) or hasattr(model, "nvfp4_optimizer"):
        raise ValueError("Pass a fresh, unconverted torch.nn.Module")
    params = list(model.parameters())
    if not params or not any(type(m) is nn.Linear for m in model.modules()):
        raise ValueError("The model must contain a standard nn.Linear")
    if any(isinstance(m, nn.Linear) and type(m) is not nn.Linear for m in model.modules()):
        raise TypeError("Custom Linear subclasses and previously converted submodules require explicit integration")
    device = params[0].device
    if device.type not in ("cpu", "cuda") or any(
        p.device != device or p.dtype != torch.float32 or not p.is_contiguous() or not p.requires_grad
        or not p.numel() for p in params
    ):
        raise ValueError("Expected trainable, nonempty, contiguous FP32 parameters on one CPU or CUDA device")
    if len({p.untyped_storage().data_ptr() for p in params}) != len(params):
        raise ValueError("Tie weights using the same Parameter object, not distinct views of one storage")
    engine, _, rng, linear = load_runtime()
    if precision == "fp4":
        rng.ensure_verified_modes()
    direct = device.type == "cpu" and cfg.direct_cpu
    if device.type == "cpu" and not direct:
        model = model.cuda()
    model, opt = engine.configure(model, precision, cfg.storage, lr=lr, tiles=cfg.tiles, warps=cfg.warps,
                                  direct=direct, remat=remat, act_pair=precision=="fp4", fast_scale=precision=="fp4",
                                  mma_k=128 if precision=="fp4" and cfg.short_mma_warps else 0,
                                  mma_warps=cfg.short_mma_warps or 8)
    if precision == "fp4":
        for module in model.modules():
            if isinstance(module, engine.Linear6):
                module.__class__ = linear
    settings = dict(betas=tuple(betas), eps=eps, weight_decay=weight_decay)
    opt.defaults.update(settings)
    opt.param_groups[0].update(settings)
    model.nvfp4_optimizer = opt
    model.nvfp4_profile = profile
    model.nvfp4_precision = precision
    model.nvfp4_remat = remat
    return model, opt


def enable_nvfp4(model, *, profile="balanced", **options):
    """Return the converted model; its optimizer is model.nvfp4_optimizer."""
    return prepare_training(model, precision="fp4", profile=profile, **options)[0]


def _loss(model, fn, *batch):
    import torch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = fn(model, *batch)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.requires_grad:
        raise ValueError("loss_fn(model, *batch) must return a scalar loss with gradients")
    return loss


def train_step(model, loss_fn, *batch):
    """One eager step on CUDA inputs; dimensions can vary between steps."""
    opt = model.nvfp4_optimizer
    opt.zero_grad(set_to_none=True)
    loss = _loss(model, loss_fn, *batch)
    opt.backward_step(loss)
    return loss.detach()


def capture_step(model, loss_fn, x, y, *, steps):
    """Capture fixed-shape forward, loss, backward and AdamW. Use as a context manager."""
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    opt = model.nvfp4_optimizer
    _, step_cls, _, _ = load_runtime()
    fresh = not any(s.get("step", 0) for s in opt.state.values())
    def compute(x, y):
        return _loss(model, loss_fn, x, y)
    return step_cls(model, opt, compute, x, y, steps, stream_updates=True, from_start=fresh)


def _recipe(model):
    return (model.nvfp4_profile, model.nvfp4_precision, model.nvfp4_remat)


def save_checkpoint(path, model):
    """Atomically save model, optimizer, CPU/CUDA RNG and the NVFP4 SR counter."""
    import torch
    path = Path(path)
    opt_state = model.nvfp4_optimizer.state_dict()  # also enforces closed graph
    rng = load_runtime()[2]
    payload = dict(schema=1, recipe=_recipe(model), model=model.state_dict(), optimizer=opt_state,
                   sr=int(rng._SEED), cpu_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state(next(model.parameters()).device))
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".nvfp4-", suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_checkpoint(path, model):
    """Restore into a freshly prepared matching model. Returns the restored optimizer step."""
    import torch
    opt = model.nvfp4_optimizer
    opt._require_eager()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema") != 1 or saved.get("recipe") != _recipe(model):
        raise ValueError("Checkpoint profile, precision or remat does not match the prepared model")
    state = model.state_dict()
    other = saved["model"]
    if state.keys() != other.keys() or any(
        p.dtype != other[k].dtype or (p.shape != other[k].shape and
            not (k.rsplit(".", 1)[-1] == "_v6_unit_buffer" and other[k].ndim == 0 and p.ndim == 1))
        for k,p in state.items()
    ):
        raise ValueError("Checkpoint model structure, dtype or shape mismatch")
    if len(saved["optimizer"]["param_groups"]) != 1 or len(saved["optimizer"]["param_groups"][0]["params"]) != len(opt.param_groups[0]["params"]):
        raise ValueError("Checkpoint optimizer parameter group mismatch")
    model.load_state_dict(other)
    opt.load_state_dict(saved["optimizer"])
    load_runtime()[2].reset_stats(saved["sr"])
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"], next(model.parameters()).device)
    return max((s.get("step", 0) for s in opt.state.values()), default=0)


def check_finite(model):
    """Explicit diagnostic over all parameters, buffers and floating optimizer states."""
    import torch
    opt = model.nvfp4_optimizer
    checks = []
    def add(name, tensor):
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            return
        if tensor.dtype == torch.float8_e4m3fn:
            ok = ((tensor.detach().view(torch.uint8) & 127) != 127).all()
        else:
            ok = torch.isfinite(tensor.detach()).all()
        checks.append((name, ok))
    for name,p in model.named_parameters():
        add(name, p)
        for key,v in opt.state.get(p, {}).items():
            add(name+"."+key, v)
    for name,b in model.named_buffers():
        add(name,b)
    flags = torch.stack([v for _,v in checks]).tolist()
    bad = [name for (name,_), ok in zip(checks, flags) if not ok]
    if bad:
        raise FloatingPointError("Nonfinite training tensors: "+", ".join(bad))
