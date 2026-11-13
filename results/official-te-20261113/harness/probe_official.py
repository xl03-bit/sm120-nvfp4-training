import argparse, importlib, json, traceback
from pathlib import Path
from unittest.mock import patch
import torch
import transformer_engine as te_root
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

torch.set_num_threads(4)
p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path,default=Path('results'));args=p.parse_args()
out=args.outdir;out.mkdir(parents=True,exist_ok=True)
result={'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'te':te_root.__version__,'te_path':te_root.__file__,'gpu':torch.cuda.get_device_name(),
        'capability':torch.cuda.get_device_capability(),'cases':[]}
result['loaded_gpu_libraries']=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if any(name in line for name in ('libcuda','libcublas','libcurand','libnvrtc','transformer_engine')) and '/' in line})
for rht,sr in [(True,True),(True,False),(False,True),(False,False)]:
    row={'rht_requested':rht,'sr_requested':sr}
    try:
        torch.manual_seed(1234)
        recipe=NVFP4BlockScaling(disable_rht=not rht,disable_stochastic_rounding=not sr)
        layer=te.Linear(512,512,bias=False,params_dtype=torch.float32,device='cuda')
        x=torch.randn(512,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        opt=torch.optim.AdamW(layer.parameters(),lr=1e-4,weight_decay=.01,fused=True)
        losses=[]
        for i in range(16):
            opt.zero_grad(set_to_none=True);x.grad=None
            with torch.autocast('cuda',dtype=torch.bfloat16),te.autocast(enabled=True,recipe=recipe):
                loss=layer(x).float().square().mean()
            loss.backward();opt.step();losses.append(loss.item())
        row['quantizers']={key:[{'type':type(q).__name__,'rht':getattr(q,'with_rht',None),'sr':getattr(q,'stochastic_rounding',None),
                                'two_dimensional_scaling':getattr(q,'with_2d_quantization',None),
                                'rht_matrix_shape':list(q.rht_matrix.shape) if hasattr(q,'rht_matrix') else None} for q in qs] for key,qs in layer.quantizers.items()}
        linear_module=importlib.import_module('transformer_engine.pytorch.module.linear')
        original_gemm=linear_module.general_gemm
        row['gemm_calls']=[]
        def annotated_gemm(a,b,*args,**kwargs):
            layout=kwargs.get('layout','TN')
            role={'TN':'FWD','NN':'DGRAD','NT':'WGRAD'}[layout]
            row['gemm_calls'].append(dict(role=role,layout=layout,
                operands=[dict(type=type(t).__name__,fp4_dtype=str(getattr(t,'_fp4_dtype',None)),
                               scale_dtype=str(getattr(t,'_scale_dtype',None))) for t in (a,b)]))
            with torch.autograd.profiler.record_function('PROBE_GEMM_'+role):return original_gemm(a,b,*args,**kwargs)
        with patch.object(linear_module,'general_gemm',annotated_gemm),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            opt.zero_grad(set_to_none=True);x.grad=None
            with torch.autograd.profiler.record_function('PROBE_FORWARD'):
                with torch.autocast('cuda',dtype=torch.bfloat16),te.autocast(enabled=True,recipe=recipe):
                    loss=layer(x).float().square().mean()
            with torch.autograd.profiler.record_function('PROBE_BACKWARD'):loss.backward()
            opt.step();torch.cuda.synchronize()
        prof.export_chrome_trace(str(out/f'official_rht{int(rht)}_sr{int(sr)}_trace.json'))
        row.update(status='passed',losses=losses,finite_grad=bool(torch.isfinite(x.grad).all()),recipe=repr(recipe),
                   kernels=[{'name':e.name,'device_us':e.device_time_total} for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA])
    except Exception as exc:
        row.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    result['cases'].append(row)
    (out/'official_probe.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in row.items() if k not in ('kernels','traceback','recipe') }),flush=True)
