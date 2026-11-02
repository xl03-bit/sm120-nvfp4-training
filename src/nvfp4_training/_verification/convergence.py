"""Fixed-data optimization regression; compare trajectories from identical parameters."""
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .._backend import nvfp4_fast_v2 as fast
from . import nvfp4_cuda
from .adapter import enable_nvfp4 as old

def run():
    all_curves = {}
    for seed in [12, 91]:
        torch.manual_seed(seed)
        base = nn.Sequential(nn.Linear(150, 256), nn.SiLU(), nn.Linear(256, 65)).cuda()
        x = torch.randn(3, 37, 150, device='cuda', dtype=torch.bfloat16)
        target = torch.randn(3, 37, 65, device='cuda')
        curves = {}
        for label in ('bf16', 'original', 'optimized'):
            model = copy.deepcopy(base)
            if label == 'original':
                model = old(model)
                nvfp4_cuda._SEED[0] = seed
            if label == 'optimized':
                model = fast.enable_nvfp4(model)
                fast.reset_stats(seed)
            opt = torch.optim.AdamW(model.parameters(), fused=True, lr=0.001)
            losses = []
            for i in range(96):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss = F.mse_loss(model(x).float(),target)
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError(f'{label} nonfinite gradient at {i}')
                opt.step()
                losses.append(loss.item())
            if losses[-1] >= losses[0]*0.2:
                raise RuntimeError(f'{label} insufficient progress: {losses[0]} -> {losses[-1]}')
            curves[label] = losses
            print(seed,label,losses[0],losses[-1],flush=True)
        all_curves[seed] = curves
        # After the first layer x has no gradient. Reference consumes an SR
        # seed for its discarded dX, v3 only computes requested gradients.
        if abs(curves['original'][-1]-curves['optimized'][-1]) > 0.03*curves['original'][0]:
            raise RuntimeError('FP4 endpoints differ materially')
    return all_curves
