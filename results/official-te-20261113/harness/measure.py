"""Run full, unchanged repository workloads with original or official TE compute."""
import argparse, contextlib, ctypes, gc, importlib.util, json, math, statistics, threading, time, traceback
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F

ROOT=Path(__file__).parent
torch.set_num_threads(4)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

class DeviceMemory:
    class Info(ctypes.Structure):
        _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    def __init__(self):
        self.samples=[];self.stop=threading.Event();self.error=None
        try:
            self.lib=ctypes.CDLL('libnvidia-ml.so.1')
            assert self.lib.nvmlInit_v2()==0
            self.handle=ctypes.c_void_p();assert self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))==0
            self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
        except Exception as exc:self.error=repr(exc)
    def loop(self):
        while not self.stop.is_set():
            info=self.Info()
            rc=self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(info))
            if rc==0:self.samples.append([time.time(),info.used,info.total])
            else:self.error=f'nvmlDeviceGetMemoryInfo returned {rc}'
            self.stop.wait(.02)
    def finish(self):
        self.stop.set()
        if hasattr(self,'thread'):self.thread.join()
        return dict(source='NVML whole-device, sampled every 20 ms; includes non-PyTorch memory',
                    error=self.error,samples=self.samples,peak_bytes=max((x[1] for x in self.samples),default=None))

def memory():
    free,total=torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                cuda_device_used=total-free)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def run(a):
    monitor=DeviceMemory()
    report=dict(config=vars(a),status='running',torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),memory_initial=memory(),losses=[],points=[])
    try:
        from nvfp4_training import prepare_training, capture_step, check_finite
        from nvfp4_training._workloads.synthetic import Network
        from nvfp4_training._workloads.lm import LanguageModel,TextBatches
        from nvfp4_training._workloads.token_loss import token_cross_entropy
        from nvfp4_training._backend.step6 import StaticStep
        torch.manual_seed(a.seed);torch.cuda.reset_peak_memory_stats()
        if a.task=='453m':
            model=Network(1536,16,12)
            gen=torch.Generator(device='cuda').manual_seed(2026)
            sample=tuple(torch.randn(2,64,1536,device='cuda',dtype=torch.bfloat16,generator=gen) for _ in range(2))
            batches=[sample]*(12+150)
            objective=lambda m,x,y:F.mse_loss(m(x).float(),y.float())
            valid=[];tokens=128
        else:
            model=LanguageModel(512)
            args=SimpleNamespace(batch=2,seq=512,seed=a.seed,warmup=0 if a.curve else 12,steps=8192 if a.curve else 30,rounds=1 if a.curve else 5,val_batches=64)
            if a.curve:
                source=module(ROOT/'project/results/fineweb-4b-2048/train.py','large_run')
                data=source.MatchedBatches(args)
            else:data=TextBatches(args)
            batches,valid=data.train,data.validation
            report['provenance']=data.provenance
            objective=lambda m,x,y:token_cross_entropy(m(x),y)
            tokens=1024
        report['parameters']=sum(p.numel() for p in model.parameters())
        if a.backend in ('te','te_default'):
            if a.backend=='te_default':
                import te_default as te_adapter
                StaticStep=te_adapter.CapturedStep
                report['stack']='Official NVFP4 defaults; FP32 parameters/gradients/Adam moments; capturable FusedAdam for unchanged graph timing'
            else:import te_adapter
            if a.remat:raise NotImplementedError('Official TE has no equivalent to project transpose recomputation; configuration not changed')
            model,opt=te_adapter.prepare(model,a.profile)
            ctx=te_adapter.context
            report['recipe']=repr(te_adapter.RECIPE)
        else:
            model,opt=prepare_training(model,profile=a.profile,precision='fp4',remat=a.remat)
            ctx=contextlib.nullcontext
            from nvfp4_training._backend import nvfp4_fast_v2
            nvfp4_fast_v2.reset_stats(0)
        report['optimizer']={k:v.item() if isinstance(v,torch.Tensor) else v for k,v in opt.param_groups[0].items() if k!='params'}
        def compute(x,y):
            with torch.autocast('cuda',dtype=torch.bfloat16),ctx():return objective(model,x,y)
        @torch.no_grad()
        def evaluate():
            values=[]
            for x,y in valid:
                with torch.autocast('cuda',dtype=torch.bfloat16),ctx():
                    values.append(F.cross_entropy(model(x).flatten(0,1).float(),y.flatten()).item())
            return dict(ce=statistics.mean(values),per_batch=values)
        if a.backend in ('te','te_default'):
            # Initialize TE handles/JIT without any optimizer update or changed weights.
            rng=torch.cuda.get_rng_state()
            for _ in range(3):
                pending={}
                for p in model.parameters():
                    if getattr(p,'_v6_defer_dense',False):p._v4_pending=pending
                compute(*batches[0]).backward()
                for p in model.parameters():
                    if hasattr(p,'_v4_pending'):del p._v4_pending
                    p.grad=None
            torch.cuda.synchronize();torch.cuda.set_rng_state(rng)
        if a.curve:report['points'].append(dict(step=0,**evaluate()))
        total=8192 if a.curve else 162
        if a.phase_memory:
            report['memory_before_capture']=memory()
            torch.cuda.reset_peak_memory_stats()
        graph=StaticStep(model,opt,compute,*batches[0],total,stream_updates=True,from_start=True)
        report['memory_capture']=memory()
        write(a.output,report)
        with graph:
            if a.curve:
                points=[0,64,162,312,512,768,1024,1536,2048,3072,4096,6144,8192]
                history=torch.empty(total,device='cuda')
                for lo,hi in zip(points,points[1:]):
                    torch.cuda.synchronize();start=time.perf_counter()
                    for i in range(lo,hi):history[i].copy_(graph.step(*batches[i]))
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    losses=history[lo:hi].tolist()
                    report['executed_updates']=hi
                    if not all(map(math.isfinite,losses)):
                        report['nonfinite_steps']=[lo+i+1 for i,v in enumerate(losses) if not math.isfinite(v)]
                        report['failed_training_segment']=[dict(step=lo+i+1,loss=v if math.isfinite(v) else str(v)) for i,v in enumerate(losses)]
                        raise FloatingPointError(f"nonfinite training loss first observed at step {report['nonfinite_steps'][0]}")
                    report['losses'].extend(losses)
                    report['points'].append(dict(step=hi,seconds_per_step=elapsed/(hi-lo),train_mean=statistics.mean(losses),**evaluate(),**memory()))
                    write(a.output,report);print(a.backend,a.profile,hi,report['points'][-1]['ce'],flush=True)
            else:
                for i in range(12):graph.step(*batches[i])
                times=[]
                for r in range(5):
                    torch.cuda.synchronize();start=time.perf_counter()
                    for j in range(30):loss=graph.step(*batches[12+r*30+j])
                    torch.cuda.synchronize();times.append((time.perf_counter()-start)/30)
                    report['losses'].append(loss.item())
                if not all(map(math.isfinite,report['losses'])):raise FloatingPointError('nonfinite training loss')
                report.update(rounds_seconds=times,seconds_per_step=statistics.median(times),tokens_per_second=tokens/statistics.median(times))
            report['memory_training']=memory()
            if a.phase_memory:
                report['memory_capture_and_training_only']=dict(report['memory_training'])
                for key in ('peak_allocated','peak_reserved'):
                    report['memory_training'][key]=max(report['memory_training'][key],report['memory_before_capture'][key])
            report['device_memory_training_peak_bytes']=max((s[1] for s in monitor.samples),default=None)
        check_finite(model)
        steps=([g['step'].item() for g in opt.param_groups] if a.backend=='te_default' else [s['step'] for s in opt.state.values()])
        assert set(steps)=={total},steps
        report.update(status='complete',memory_final=memory(),optimizer_steps=sorted(set(steps)))
    except Exception as exc:
        report.update(status='unsupported' if isinstance(exc,NotImplementedError) else 'failed',error=repr(exc),traceback=traceback.format_exc(),memory_failure=memory())
        print(report['traceback'],flush=True)
    finally:
        report['device_memory']=monitor.finish();write(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('device_memory','losses','traceback','recipe','points')}),flush=True)
    return report['status']=='complete'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['original','te','te_default'],required=True)
    p.add_argument('--profile',default='balanced');p.add_argument('--task',choices=['453m','fineweb'],default='453m')
    p.add_argument('--remat',action='store_true');p.add_argument('--curve',action='store_true')
    p.add_argument('--phase-memory',action='store_true',help='Separate preparation from graph capture/training peaks; leave computations unchanged')
    p.add_argument('--seed',type=int,default=1234);p.add_argument('--output',required=True)
    a=p.parse_args();raise SystemExit(0 if run(a) else 1)
