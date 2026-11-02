"""Paired-seed end-to-end gradient checks, plus independent-seed SR control."""
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from . import nvfp4_cuda as old
from .adapter import enable_nvfp4 as reference
from .._backend import nvfp4_fast_v2 as fast
from .verify_fast_v2 import relative

def run():
    rows = []
    for batch, seq, k, n in [(1, 17, 150, 65), (4, 32, 384, 512), (3, 71, 257, 513)]:
        torch.manual_seed(915)
        base = nn.Linear(k, n).cuda()
        ref = reference(copy.deepcopy(base))
        new = fast.enable_nvfp4(base)
        x = torch.randn(batch, seq, k, device='cuda', dtype=torch.bfloat16)
        g = torch.randn(batch, seq, n, device='cuda', dtype=torch.bfloat16)
        def grads(model):
            v = x.detach().requires_grad_()
            return torch.autograd.grad(model(v), (v, model.weight), g)
        paired, independent = [], []
        means = [torch.zeros_like(x, dtype=torch.float32), torch.zeros_like(new.weight)]
        for seed in range(16):
            old._SEED[0] = fast_seed = seed * 19
            fast.reset_stats(fast_seed)
            a, b = grads(ref), grads(new)
            paired.append(max(relative(i,j) for i,j in zip(a,b)))
            old._SEED[0] = seed * 19 + 500
            other = grads(ref)
            independent.append(max(relative(i,j) for i,j in zip(other,b)))
            for m, v in zip(means,b):
                m.add_(v.float()/16)
        if max(paired) > 0.03 or min(independent) <= max(paired):
            raise AssertionError((paired,independent))
        rows.append(dict(shape=[batch,seq,k,n], paired_errors=paired, independent_errors=independent,
                         mean_gradient_norms=[v.norm().item() for v in means]))
        print(rows[-1],flush=True)
    return rows
