"""Validate official frozen bases with unchanged LoRA graph/checkpoint wrapper."""
import argparse,json,traceback
from pathlib import Path
import torch
from torch import nn
from nvfp4_training import capture_lora_step
from lora_compare import setup

p=argparse.ArgumentParser();p.add_argument('--output',default='results/lora-adapter-check.json');p.add_argument('--disable-rht',action='store_true');a=p.parse_args()
if a.disable_rht:
    import te_adapter
    from transformer_engine.common.recipe import NVFP4BlockScaling
    te_adapter.RECIPE=NVFP4BlockScaling(disable_rht=True,disable_stochastic_rounding=False)
rows=[]
for checkpoint in [False,True]:
    row=dict(checkpoint=checkpoint,diagnostic_only=True,rht=not a.disable_rht)
    try:
        torch.manual_seed(1234)
        model=nn.Sequential(nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.Linear(512,512))).bfloat16().cuda()
        model,opt=setup(model,'nvfp4',16,['0'] if checkpoint else [])
        frozen=[(p,p.detach().clone()) for p in model.parameters() if not p.requires_grad]
        trained=[(p,p.detach().clone()) for p in model.parameters() if p.requires_grad]
        x=torch.randn(2,64,512,device='cuda',dtype=torch.bfloat16);y=torch.randn_like(x)
        fn=lambda m,x,y:(m(x).float()-y.float()).square().mean()
        losses=[]
        with capture_lora_step(model,opt,fn,x,y) as graph:
            for _ in range(4):losses.append(graph.step(x,y).item())
        assert all(torch.equal(p,old) for p,old in frozen)
        assert all(torch.isfinite(p).all() for p,old in trained)
        assert all(not torch.equal(p,old) for p,old in trained)
        assert all(s['step'].item()==4 for s in opt.state.values())
        row.update(status='passed',losses=losses,frozen_unchanged=True,optimizer_steps=4)
    except Exception:row.update(status='failed',traceback=traceback.format_exc())
    rows.append(row);Path(a.output).write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
