"""One diagnostic full 453M update, separate from all performance measurements."""
import importlib,json
from unittest.mock import patch
from pathlib import Path
import torch
from torch.nn import functional as F
from nvfp4_training._workloads.synthetic import Network
from te_default import prepare,context
torch.set_num_threads(4);torch.manual_seed(1234)
model,opt=prepare(Network(1536,16,12))
gen=torch.Generator(device='cuda').manual_seed(2026)
x,y=[torch.randn(2,64,1536,device='cuda',dtype=torch.bfloat16,generator=gen) for _ in range(2)]
def update():
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16),context():loss=F.mse_loss(model(x).float(),y.float())
    loss.backward();opt.step();return loss
for _ in range(3):update()
mod=importlib.import_module('transformer_engine.pytorch.module.linear');original=mod.general_gemm;calls=[]
def wrapped(a,b,*args,**kwargs):
    role={'TN':'FWD','NN':'DGRAD','NT':'WGRAD'}[kwargs.get('layout','TN')]
    calls.append(dict(role=role,operands=[dict(type=type(t).__name__,dtype=str(getattr(t,'_fp4_dtype',None)),shape=list(t.shape)) for t in (a,b)]))
    with torch.autograd.profiler.record_function('FULL453_GEMM_'+role):return original(a,b,*args,**kwargs)
with patch.object(mod,'general_gemm',wrapped),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    loss=update();torch.cuda.synchronize()
prof.export_chrome_trace('results/default453-full-update.trace.json')
report=dict(diagnostic_only=True,loss=loss.item(),gemm_calls=calls,
            kernels=[dict(name=e.name,microseconds=e.device_time_total) for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA])
Path('results/default453-full-update-profile.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(loss=loss.item(),roles={role:sum(c['role']==role for c in calls) for role in ['FWD','DGRAD','WGRAD']},kernels=len(report['kernels']))),flush=True)
