"""Real-GPU integration tests of the public training interface."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from . import (capture_step, check_finite, enable_nvfp4, list_profiles, load_checkpoint,
               prepare_training, save_checkpoint, train_step)
from ._runtime import load_runtime


class TestModel(nn.Module):
    def __init__(self, width, hidden, vocab):
        super().__init__()
        self.embedding = nn.Embedding(vocab,width)
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width,hidden)
        self.down = nn.Linear(hidden,width)
        self.head = nn.Linear(width,vocab,bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens):
        x = self.norm(self.embedding(tokens))
        return self.head(x + self.down(F.silu(self.up(x))))


def objective(model, x, y):
    return F.cross_entropy(model(x).flatten(0,1).float(), y.flatten())


def same(a,b):
    if torch.is_tensor(a):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.detach().reshape(-1).view(torch.uint8),
                           b.detach().reshape(-1).view(torch.uint8))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for v,w in zip(a,b): same(v,w)
    else:
        assert a==b


def run_one(profile, shape, remat, path):
    torch.manual_seed(314)
    aligned = shape=="aligned"
    width,hidden,vocab,tokens=(512,512,512,128) if aligned else (131,197,257,19)
    def build():
        net=TestModel(width,hidden,vocab)
        if remat:net.cuda()
        return enable_nvfp4(net,profile=profile,remat=remat,lr=0.0002,betas=(.89,.995),weight_decay=.02)
    model=build()
    rng=load_runtime()[2]
    rng.reset_stats(71)
    opt=model.nvfp4_optimizer
    assert model.head.weight is model.embedding.weight and model.head.weight.dtype==torch.float32
    expected={'balanced':torch.float16,'short128':torch.float16,'conservative':torch.bfloat16,
              'same_as_v5':torch.float32,'experimental_m4v4':torch.float16,'experimental_fp8_master':torch.float8_e4m3fn}
    assert model.up.weight.dtype==expected[profile]
    assert opt.param_groups[0]['betas']==(.89,.995) and opt.param_groups[0]['weight_decay']==.02
    x,y=[torch.randint(vocab,(1,tokens),device='cuda') for _ in range(2)]
    hit=0
    original=torch._scaled_mm
    def probe(a,b,*args,**kw):
        nonlocal hit
        if a.dtype==b.dtype==torch.float4_e2m1fn_x2:hit+=1
        return original(a,b,*args,**kw)
    with patch.object(torch,'_scaled_mm',probe):
        g=capture_step(model,objective,x,y,steps=3)
    fused=getattr(opt,'fused_mma_calls',0)
    assert hit+fused==9 and g.sr_calls==6 and rng.stats()['rht_quant']==6,(hit,fused,g.sr_calls,rng.stats())
    with g:
        losses=[g.step(x,y).item() for _ in range(3)]
    x2,y2=x[:,:tokens-3].contiguous(),y[:,:tokens-3].contiguous()
    losses.append(train_step(model,objective,x2,y2).item())
    check_finite(model)
    assert len(opt.state)==len(list(model.parameters())) and all(s['step']==4 for s in opt.state.values())
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(x)
        assert logits.shape==(1,tokens,vocab) and torch.isfinite(logits).all()
    model.train()
    save_checkpoint(path,model)
    reference=train_step(model,objective,x,y)
    other=build()
    assert load_checkpoint(path,other)==4
    actual=train_step(other,objective,x,y)
    same(actual,reference)
    same(model.state_dict(),other.state_dict())
    same(opt.state_dict(),other.nvfp4_optimizer.state_dict())
    with capture_step(other,objective,x,y,steps=2) as again:
        losses.extend(again.step(x,y).item() for _ in range(2))
    assert all(s['step']==7 for s in other.nvfp4_optimizer.state.values())
    check_finite(other)
    assert all(torch.isfinite(torch.tensor(losses)))
    if profile=='experimental_fp8_master' and not aligned and not remat:
        with torch.no_grad():other.up.weight.view(torch.uint8).flatten()[0]=127
        try:check_finite(other)
        except FloatingPointError as e:assert 'up.weight' in str(e)
        else:raise AssertionError('FP8 NaN was missed')
    return dict(profile=profile,shape=shape,remat=remat,native=hit+fused,fused_wgrad=fused,sr=g.sr_calls,
                rht=6,checkpoint_next_update_exact=True,final_step=7)


def run(profiles,output=None):
    torch.use_deterministic_algorithms(True)
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        for p in profiles:
            for shape in ('aligned','padded'):
                for remat in (False,True):
                    row=run_one(p,shape,remat,Path(tmp)/'train.pt')
                    rows.append(row)
                    print('PASS',p,shape,'remat='+str(remat),'GEMM/SR/RHT=9/6/6; exact resume',flush=True)
    if output:
        Path(output).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
            cuda=torch.version.cuda,tests=rows),indent=2),encoding='utf8')
    return rows
