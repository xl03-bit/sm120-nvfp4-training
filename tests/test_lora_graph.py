import json
import sys
import torch
from torch import nn
from precision_lora import prepare_lora, capture_lora_step, lora_parameters
from precision_lora.core import _FrozenFP4

torch.set_num_threads(4)
def make(mode):
    torch.manual_seed(184)
    return prepare_lora(nn.Sequential(nn.Linear(257,193),nn.GELU(),nn.Linear(193,65)).cuda().bfloat16(),mode=mode,rank=8,lr=.001)
def loss_fn(m,x,y):return (m(x).float()-y).square().mean()
rows=[]
for mode in (('preserve',) if '--preserve-only' in sys.argv else ('preserve','nvfp4')):
    m,opt=make(mode);ref,ro=make(mode)
    x=torch.randn(2,17,257,device='cuda',dtype=torch.bfloat16);y=torch.randn(2,17,65,device='cuda')
    initial=[p.detach().clone() for p in lora_parameters(m)]
    with capture_lora_step(m,opt,loss_fn,x,y) as graph:
        assert all(torch.equal(a,b) for a,b in zip(initial,lora_parameters(m)))
        assert all(s['step'].item()==0 for s in opt.state.values())
        for step in range(8):
            ro.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):expected=loss_fn(ref,x,y)
            expected.backward();ro.step()
            actual=graph.step(x,y)
            assert actual.item()==expected.item(), (mode,step,actual,expected)
        assert all(torch.equal(a,b) for a,b in zip(lora_parameters(m),lora_parameters(ref)))
        assert all(s['step'].item()==8 for s in opt.state.values())
    # Re-capture a nonfresh optimizer without resetting or advancing its step.
    with capture_lora_step(m,opt,loss_fn,x,y) as graph:
        assert all(s['step'].item()==8 for s in opt.state.values())
        graph.step(x,y)
    assert all(s['step'].item()==9 for s in opt.state.values())
    seeds=[b.sr_seed.item() for b in m.modules() if isinstance(b,_FrozenFP4)]
    if seeds:assert seeds[-1]==100003+9
    rows.append(dict(mode=mode,updates=9,exact_eager_graph=True,sr_counters=seeds))
print(json.dumps(rows,indent=2),flush=True)
