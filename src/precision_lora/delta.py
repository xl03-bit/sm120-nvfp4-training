"""Chunked gated delta-rule inference with FP32 state on SM120."""
import math

import torch

from ._delta_cuda import unit_lower_inverse_


def _normalize(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _head_major(x):
    return x.transpose(1, 2).to(dtype=torch.float32, memory_format=torch.contiguous_format)


def _chunk(query, key, value, gate, beta, state, need_state, normalize):
    if normalize:
        key = _normalize(key)
    k = _head_major(key)
    step = _head_major(beta).unsqueeze(-1)
    log_decay = _head_major(gate).cumsum(-1)
    # Entry (i,j) decays the j-th update up to query i, for j <= i.
    decay = (log_decay.unsqueeze(-1) - log_decay.unsqueeze(-2)).exp_().tril_()
    weighted_key = k * step
    kt = k.transpose(-2, -1)
    transform = unit_lower_inverse_((weighted_key @ kt).mul_(decay).neg_())
    updates = transform @ (_head_major(value) * step)
    if state is not None:
        from_start = log_decay.exp().unsqueeze(-1)
        state_keys = transform @ weighted_key.mul_(from_start)
        updates.sub_(state_keys @ state)
        del state_keys
    del transform, weighted_key

    if normalize:
        query = _normalize(query)
    q = _head_major(query) * (1 / math.sqrt(query.shape[-1]))
    output = (q @ kt).mul_(decay) @ updates
    if state is not None:
        output.add_((q * from_start) @ state)

    next_state = None
    if need_state:
        last = log_decay[..., -1:]
        to_end = (last - log_decay).exp_().unsqueeze(-1)
        next_state = (k * to_end).transpose(-2, -1) @ updates
        if state is not None:
            next_state.add_(state * last.exp().unsqueeze(-1))
    return output, next_state


def gated_delta_rule(query, key, value, g, beta, *, chunk_size=64, initial_state=None,
                     output_final_state=False, use_qk_l2norm_in_kernel=False):
    """Evaluate a causal gated delta rule on dense, equal-length CUDA batches.

    Q, K and V use [batch, time, heads, width]. Log decay g and update strength
    beta use [batch, time, heads]. State is [batch, heads, key_width, value_width].
    Returns (output in the input dtype, optional FP32 final state). Inputs and
    caller-owned state are read-only. Use the model's differentiable rule when
    training with gradients. Normalization, if requested, uses the input dtype.
    """
    if type(chunk_size) is not int or not 1 <= chunk_size <= 64:
        raise ValueError("chunk_size must be an integer between 1 and 64")
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4 or value.shape[:3] != query.shape[:3]:
        raise ValueError("Expected Q/K/V shaped [batch, time, heads, width] with matching Q/K widths")
    batch, length, heads, width = query.shape
    if min(batch, heads, width, value.shape[-1]) < 1 or g.shape != query.shape[:3] or beta.shape != g.shape:
        raise ValueError("Expected positive batch/head dimensions and one gate and beta per token and head")
    if query.dtype not in (torch.float32, torch.float16, torch.bfloat16) or key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("Q, K and V must share a float32, float16 or bfloat16 dtype")
    state_shape = (batch, heads, width, value.shape[-1])
    tensors = [query, key, value, g, beta]
    if initial_state is not None:
        if initial_state.shape != state_shape:
            raise ValueError("initial_state shape does not match the key and value dimensions")
        tensors.append(initial_state)
    if not query.is_cuda or any(x.device != query.device or x.layout != torch.strided for x in tensors):
        raise ValueError("All inputs must be dense tensors on one CUDA device")
    if any(x.dtype not in (torch.float32, torch.bfloat16, torch.float16) for x in tensors[3:]):
        raise TypeError("Gates and initial state must use a supported floating-point dtype")
    if torch.is_grad_enabled() and any(x.requires_grad for x in tensors):
        raise RuntimeError("gated_delta_rule supports inference only")

    output = torch.empty_like(value, memory_format=torch.contiguous_format)
    state = None if initial_state is None else initial_state.float()
    with torch.autocast("cuda", enabled=False):
        for start in range(0, length, chunk_size):
            stop = min(start + chunk_size, length)
            span = slice(start, stop)
            block, state = _chunk(query[:, span], key[:, span], value[:, span], g[:, span], beta[:, span],
                                  state, output_final_state or stop < length, use_qk_l2norm_in_kernel)
            output[:, span].copy_(block.transpose(1, 2))
            del block
    if output_final_state and length == 0:
        state = torch.zeros(state_shape, device=query.device, dtype=torch.float32) if state is None else state.clone()
    return output, state if output_final_state else None
