"""Check captured official optimizer updates against eager, without FP4 noise."""
import json,torch
from copy import deepcopy
from te_default import CapturedStep
from transformer_engine.pytorch.optimizers import FusedAdam
torch.manual_seed(1301)
a=torch.nn.Linear(32,32).cuda();b=deepcopy(a)
x=torch.randn(8,32,device='cuda');y=torch.randn_like(x)
opts=[FusedAdam(m.parameters(),lr=1e-4,weight_decay=.01,capturable=True) for m in (a,b)]
def loss(m,x,y):return (m(x)-y).square().mean()
with CapturedStep(b,opts[1],lambda x,y:loss(b,x,y),x,y,4) as graph:
    for _ in range(4):
        opts[0].zero_grad(set_to_none=True);loss(a,x,y).backward();opts[0].step()
        graph.step(x,y)
torch.cuda.synchronize()
errors=[(u-v).abs().max().item() for u,v in zip(a.parameters(),b.parameters())]
state_errors=[]
for u,v in zip(a.parameters(),b.parameters()):
    for key in ('exp_avg','exp_avg_sq'):
        state_errors.append((opts[0].state[u][key]-opts[1].state[v][key]).abs().max().item())
steps=[o.param_groups[0]['step'].item() for o in opts]
report=dict(status='passed' if max(errors+state_errors)<1e-6 and steps==[4,4] else 'failed',
            parameter_max_errors=errors,state_max_errors=state_errors,steps=steps)
from pathlib import Path
Path('results/default-graph-check.json').write_text(json.dumps(report,indent=2));print(report)
assert report['status']=='passed',report
