"""Same real cached DreamBooth batches for each LoRA implementation."""
import sys, os, json, time, statistics, argparse, hashlib
from pathlib import Path
from bootstrap import ROOT as EXPERIMENT
ROOT = EXPERIMENT/'benchmark-runs'
from bootstrap import torch, configure_attention
from klein_loader import load_model, targets, BASE
from train import forward_loss, validate
from precision_lora import prepare_lora, lora_parameters
from diffusers import FlowMatchEulerDiscreteScheduler

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode', choices=['preserve','packed_nvfp4'], default='preserve')
    p.add_argument('--backward-mode', default='recompute')
    p.add_argument('--label', required=True)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--validate', action='store_true')
    a=p.parse_args()
    out=ROOT/'results'/a.label; out.mkdir(parents=True, exist_ok=False)
    configure_attention('cudnn')
    model=load_model('nvfp4' if a.mode=='packed_nvfp4' else 'bf16', a.backward_mode)
    torch.manual_seed(7849)
    model,opt=prepare_lora(model,mode=a.mode,rank=16,alpha=16,target_modules=targets(),lr=1e-4)
    model.enable_gradient_checkpointing();model.train();params=lora_parameters(model)
    text=torch.load(EXPERIMENT/'cache/text.pt', weights_only=True)
    latents=torch.load(EXPERIMENT/'cache/latents.pt', weights_only=True)
    text={k:{j:v.cuda() if torch.is_tensor(v) else v for j,v in x.items()} for k,x in text.items()}
    latents={k:{j:v.cuda() for j,v in x.items()} for k,x in latents.items()}
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(BASE/'scheduler',local_files_only=True)
    order=[];gen=torch.Generator().manual_seed(7850)
    while len(order)<a.steps: order.extend(torch.randperm(4,generator=gen).tolist())
    settings=dict(**vars(a),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        trainable=sum(p.numel() for p in params),resolution=1024,batch=1,rank=16,alpha=16,lr=1e-4,attention='cudnn',
        precision='BF16 compute / FP32 adapters and fused AdamW',checkpointing=True,
        timing='synchronized forward + loss + backward + clip + optimizer; no validation, profile or first warmup steps')
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    def log(r):
        with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        print(json.dumps(r),flush=True)
    if a.validate:
        log(dict(event='validation',step=0,**validate(model,latents['04'],text['heldout_04'],scheduler)))
    records=[]
    for i in range(a.steps):
        key=f'{order[i]:02d}'; data=latents[key]
        rg=torch.Generator().manual_seed(7851+i*1009);index=int(torch.randint(0,1000,(),generator=rg))
        ts=scheduler.timesteps[index:index+1].cuda();sigma=scheduler.sigmas[index:index+1].cuda().float().reshape(1,1,1,1)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss=forward_loss(model,data['latent'],data['ids'],text['train_'+key],ts,sigma,7852+i*997)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        r=dict(event='step',step=i+1,loss=float(loss.detach()),grad_norm=float(norm.detach()),seconds=elapsed,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        records.append(r);log(r)
    if a.validate:
        log(dict(event='validation',step=a.steps,**validate(model,latents['04'],text['heldout_04'],scheduler)))
    if a.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            opt.zero_grad(set_to_none=True)
            loss=forward_loss(model,data['latent'],data['ids'],text['train_'+key],ts,sigma,7852+i*997)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step()
            torch.cuda.synchronize()
        stats={}
        for e in prof.events():
            if e.device_type==torch.autograd.DeviceType.CUDA:
                r=stats.setdefault(e.name,dict(calls=0,microseconds=0));r['calls']+=1;r['microseconds']+=e.time_range.elapsed_us()
        (out/'profile.json').write_text(json.dumps(sorted([dict(name=k,**v) for k,v in stats.items()],key=lambda x:-x['microseconds']),indent=2))
    timed=records[a.warmup:];times=[r['seconds'] for r in timed]
    summary=dict(status='passed',measured_steps=len(times),mean_seconds=statistics.mean(times),median_seconds=statistics.median(times),
        round_means=[statistics.mean(times[i:i+10]) for i in range(0,len(times),10)],images_per_second=1/statistics.mean(times),
        peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in records),peak_reserved_GiB=max(r['peak_reserved_GiB'] for r in records))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));log(summary)
if __name__=='__main__':main()
