"""Greedy calibration of frozen BF16 projections on the native packed path."""
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import chain, zip_longest
import math
from numbers import Real

import torch
from torch import nn

from ._selection import matches_module
from .packed import PackedNVFP4Linear


def _patterns(value):
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("Pass module patterns as a sequence of strings")
    if any(not isinstance(p, str) or not p for p in value):
        raise ValueError("Module patterns must be nonempty strings")
    return tuple(value)


def _byte_span(tensor):
    """Byte interval, conservatively including gaps in a strided view."""
    if nn.parameter.is_lazy(tensor) or tensor.layout != torch.strided or tensor.is_meta:
        return None
    try:
        storage = tensor.untyped_storage()
        if storage.device.type == "meta" or not tensor.numel():
            return None
        address = storage.data_ptr()
        if not address:
            return None
        size = tensor.element_size()
        start = address + tensor.storage_offset() * size
        width = 1 + sum((n - 1) * stride for n, stride in zip(tensor.shape, tensor.stride()))
        return start, start + width * size
    except (RuntimeError, ValueError):
        # Tensor wrappers in unselected modules can have no accessible storage.
        return None


def _shared_tensor_ids(model):
    seen, shared = set(), set()
    by_device = defaultdict(list)
    for _, tensor in chain(model.named_parameters(remove_duplicate=False),
                           model.named_buffers(remove_duplicate=False)):
        key = id(tensor)
        if key in seen:
            shared.add(key)
            continue
        seen.add(key)
        span = _byte_span(tensor)
        if span is not None:
            by_device[tensor.device].append((*span, key))
    for spans in by_device.values():
        # The furthest previous end catches nested and partially overlapping views.
        furthest_end, owner = 0, None
        for start, end, key in sorted(spans):
            if start < furthest_end:
                shared.update((owner, key))
            if end > furthest_end:
                furthest_end, owner = end, key
    return shared


def _plan_groups(model, groups, keep):
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("groups must map group names to sequences of module patterns")
    layers = {n: m for n, m in model.named_modules(remove_duplicate=False)
              if isinstance(m, nn.Linear)}
    shared = _shared_tensor_ids(model)
    selected = set()
    plan = {}
    for group, patterns in groups.items():
        patterns = _patterns(patterns)
        if not isinstance(group, str) or not group or not patterns:
            raise ValueError("Each group needs a name and at least one module pattern")
        matched = set()
        for pattern in patterns:
            names = {n for n in layers if matches_module(n, (pattern,))}
            if not names:
                raise ValueError(f"No Linear matched {pattern!r} in group {group!r}")
            matched.update(names)
        paths = [n for n in layers if n in matched and not matches_module(n, keep)]
        for name in paths:
            layer = layers[name]
            if not name:
                raise ValueError("Wrap a root Linear in a parent module for in-place calibration")
            if name in selected:
                raise ValueError(f"Overlapping calibration groups at {name!r}")
            if type(layer) is not nn.Linear:
                raise TypeError(f"Custom Linear at {name!r} requires explicit integration")
            if any(id(p) in shared for p in layer.parameters()):
                raise ValueError(f"Shared parameters or module aliases at {name!r}")
            selected.add(name)
        plan[group] = paths
    return plan


def _score(evaluate, model):
    value = evaluate(model)
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("evaluate must return a finite real scalar")
    return float(value)


def hidden_state_evaluator(batches, observe):
    """Build a calibration score from the worst batch/layer relative L2 error.

    batches() must replay the same inputs in order. observe(model, batch) yields
    unique (name, floating_tensor) pairs per batch, with any padding removed by
    the caller. The first call stores independent CPU references and scores zero.
    Create a new evaluator for each baseline. A changed all-zero reference raises.
    """
    if not callable(batches) or not callable(observe):
        raise TypeError("batches and observe must be callable")
    reference = None

    def states(model):
        for index, batch in enumerate(batches()):
            names = set()
            for name, state in observe(model, batch):
                if not isinstance(name, str) or not name or name in names:
                    raise ValueError("Expected a unique, nonempty hidden-state name per batch")
                names.add(name)
                if not isinstance(state, torch.Tensor) or state.layout != torch.strided or not state.is_floating_point():
                    raise TypeError(f"Expected a dense floating tensor for hidden state {name!r}")
                if not state.numel():
                    raise ValueError(f"Empty hidden state {name!r} in batch {index}")
                yield (index, name), state.detach().to(device="cpu", copy=reference is None)
            if not names:
                raise ValueError(f"No hidden states observed in batch {index}")

    @torch.no_grad()
    def evaluate(model):
        nonlocal reference
        if reference is None:
            captured = []
            for key, state in states(model):
                norm = torch.linalg.vector_norm(state, dtype=torch.float64).item()
                if not math.isfinite(norm):
                    raise ValueError(f"Nonfinite reference norm at {key}")
                captured.append((key, state, norm))
            if not captured:
                raise ValueError("Expected at least one calibration batch")
            reference = captured
            return 0.0

        worst = 0.0
        for current, saved in zip_longest(states(model), reference):
            if current is None or saved is None:
                raise ValueError("Hidden-state observation count changed")
            key, state = current
            expected_key, target, norm = saved
            if key != expected_key or state.shape != target.shape or state.dtype != target.dtype:
                raise ValueError(f"Hidden-state name, order, shape or dtype changed at {expected_key}")
            if norm == 0:
                if not torch.equal(state, target):
                    raise ValueError(f"Changed hidden state with zero reference norm at {key}")
                continue
            error = torch.linalg.vector_norm(state.double() - target).item() / norm
            if not math.isfinite(error):
                raise ValueError(f"Nonfinite relative hidden-state error at {key}")
            worst = max(worst, error)
        return -worst

    return evaluate


@torch.no_grad()
def calibrate_packed_layers(model, *, groups, evaluate, max_drop, keep=(),
                            backward_mode="recompute", seed=0):
    """Try groups in order, keeping a conversion when its score >= baseline - max_drop.

    Each group maps a label to globs for frozen CUDA BF16 Linear paths or their
    ancestors. Protected paths in keep take precedence. evaluate(model) scores
    the same calibration samples on every call, with larger values preferred.
    Scoring runs in eval mode. The caller's per-module training flags are restored.

    Mutates model in place and returns baseline, final_score and one record per
    group. Rejection restores that group's original modules. If packing or
    evaluation raises, every replacement made by this call is undone.
    """
    if not isinstance(model, nn.Module) or not callable(evaluate):
        raise TypeError("Pass a torch.nn.Module and a callable evaluate(model)")
    if isinstance(max_drop, bool) or not isinstance(max_drop, Real) or not math.isfinite(max_drop) or max_drop < 0:
        raise ValueError("max_drop must be a finite nonnegative score difference")
    if backward_mode not in ("recompute", "cache"):
        raise ValueError("backward_mode must be 'recompute' or 'cache'")
    plan = _plan_groups(model, groups, _patterns(keep))
    # train() recurses. Restore every parent before its shared children.
    modes = {}
    for _, module in model.named_modules(remove_duplicate=False):
        modes.pop(module, None)
        modes[module] = module.training
    originals = {}
    records = []
    try:
        model.eval()
        baseline = final_score = _score(evaluate, model)
        for group, paths in plan.items():
            score = None
            accepted = False
            if paths:
                for name in paths:
                    layer = model.get_submodule(name)
                    originals[name] = layer
                    model.set_submodule(name, PackedNVFP4Linear.from_linear(
                        layer, backward_mode=backward_mode, seed=seed))
                score = _score(evaluate, model)
                accepted = score >= baseline - max_drop
                if accepted:
                    final_score = score
                else:
                    for name in paths:
                        model.set_submodule(name, originals.pop(name))
            records.append(dict(name=group, modules=paths, score=score, accepted=accepted))
    except BaseException:
        for name, layer in originals.items():
            model.set_submodule(name, layer)
        raise
    finally:
        for module, training in modes.items():
            if module.training != training:
                module.train(training)
    for name, original in originals.items():
        model.get_submodule(name).train(original.training)
    return dict(baseline=baseline, final_score=final_score, groups=records)
