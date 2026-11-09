"""Controlled Flux2 DreamBooth flow matching with the existing precision_lora API.

No text-encoder training, model offload, latent compression, changed attention,
or base requantization is hidden in this comparison. See protocol.json.
"""
from bootstrap import ROOT,torch,configure_attention
from klein_loader import load_model,BASE,targets
from precision_lora import prepare_lora,lora_parameters,save_lora,load_lora
from diffusers import Flux2KleinPipeline,FlowMatchEulerDiscreteScheduler
import argparse,json,time,statistics,hashlib,traceback

def frozen_digest(model):
    h=hashlib.sha256()
    for name,p in list(model.named_parameters())+list(model.named_buffers()):
        if p.requires_grad or name.endswith('.sr_seed'):continue
        h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
        h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def forward_loss(model,latent,ids,text,step,sigma,seed):
    gen=torch.Generator('cuda').manual_seed(seed)
    noise=torch.randn(latent.shape,device='cuda',dtype=latent.dtype,generator=gen)
    noisy=(1-sigma)*latent+sigma*noise
    with torch.autocast('cuda',dtype=torch.bfloat16):
        pred=model(hidden_states=Flux2KleinPipeline._pack_latents(noisy),timestep=step/1000,
            guidance=None,encoder_hidden_states=text['embeds'],txt_ids=text['ids'],img_ids=ids,return_dict=False)[0]
    target=Flux2KleinPipeline._pack_latents(noise-latent)
    return (pred.float()-target.float()).square().mean()

@torch.no_grad()
def validate(model,latent,text,scheduler):
    was_training=model.training;model.eval();records=[]
    # Same heldout photograph, eight fixed noise/time pairs for every checkpoint/mode.
    for j,index in enumerate([50,150,300,450,550,700,850,950]):
        loss=forward_loss(model,latent['latent'],latent['ids'],text,
            scheduler.timesteps[index:index+1].cuda(),scheduler.sigmas[index:index+1].cuda().to(latent['latent'].dtype).reshape(1,1,1,1),18749+j)
        records.append(float(loss))
    if was_training:model.train()
    return dict(loss=statistics.mean(records),per_timestep=records,indices=[50,150,300,450,550,700,850,950])

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['bf16','nvfp4'],required=True)
    p.add_argument('--backward-mode',choices=['cache','recompute'],default='cache')
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--label',required=True)
    p.add_argument('--validate-every',type=int,default=100);p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--resume');p.add_argument('--resume-step',type=int,default=0);p.add_argument('--skip-validation',action='store_true')
    p.add_argument('--attention',choices=['efficient','cudnn'],default='efficient')
    a=p.parse_args();configure_attention(a.attention);protocol=json.loads((ROOT/'protocol.json').read_text())
    if a.mode=='nvfp4' and not (ROOT/'metadata/layout-verification.json').exists():raise RuntimeError('Verify layout first')
    out=ROOT/'results'/a.label;out.mkdir(parents=True,exist_ok=False)
    def log(event,**data):
        r=dict(event=event,**data)
        with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
        print(json.dumps(r),flush=True)
    try:
        cfg=protocol['proposed_training'];model=load_model(a.mode,a.backward_mode)
        torch.manual_seed(cfg['seed'])
        model,optim=prepare_lora(model,mode='preserve' if a.mode=='bf16' else 'packed_nvfp4',
            rank=cfg['rank'],alpha=cfg['alpha'],target_modules=targets(),lr=cfg['lr'],
            betas=cfg['betas'],eps=cfg['eps'],weight_decay=cfg['weight_decay'])
        model.enable_gradient_checkpointing();model.train();params=lora_parameters(model)
        base_id='FLUX.2-klein-base-4B:a3b4f4849157f664bdbc776fd7453c2783562f4d:'+a.mode
        if a.resume:
            load_lora(a.resume,model,optimizer=optim,base_id=base_id)
            restored_steps={int(s['step'].item()) for s in optim.state.values() if 'step' in s}
            if restored_steps!={a.resume_step}:
                raise ValueError(f'Optimizer checkpoint step {restored_steps} differs from --resume-step {a.resume_step}')
        before=frozen_digest(model)
        text=torch.load(ROOT/'cache/text.pt',weights_only=True)
        latents=torch.load(ROOT/'cache/latents.pt',weights_only=True)
        text={k:{j:v.cuda() if torch.is_tensor(v) else v for j,v in x.items()} for k,x in text.items()}
        latents={k:{j:v.cuda() for j,v in x.items()} for k,x in latents.items()}
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(BASE/'scheduler',local_files_only=True)
        plan_rng=torch.Generator().manual_seed(cfg['seed']+1)
        order=[]
        while len(order)<a.steps:order.extend(torch.randperm(4,generator=plan_rng).tolist())
        # Independent per-step seeds avoid a changed requested run length changing its prefix.
        settings=dict(**vars(a),recipe=cfg,base_id=base_id,trainable_parameters=sum(p.numel() for p in params),
            frozen_sha256_before=before,activation_scale=getattr(model,'activation_scale_recipe',None),
            precision='BF16 compute with FP32 adapters and AdamW states',attention_description='PyTorch '+a.attention+' SDPA',
            environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
        (out/'settings.json').write_text(json.dumps(settings,indent=2));log('start',**settings)
        if not a.skip_validation:log('validation',step=a.resume_step,**validate(model,latents['04'],text['heldout_04'],scheduler))
        times=[];max_allocated=0;max_reserved=0;start_run=time.perf_counter()
        for i in range(a.resume_step,a.steps):
            key=f'{order[i]:02d}';data=latents[key]
            trng=torch.Generator().manual_seed(cfg['seed']+2+i*1009)
            index=int(torch.randint(0,1000,(),generator=trng))
            timestep=scheduler.timesteps[index:index+1].cuda()
            sigma=scheduler.sigmas[index:index+1].cuda().to(data['latent'].dtype).reshape(1,1,1,1)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            optim.zero_grad(set_to_none=True)
            loss=forward_loss(model,data['latent'],data['ids'],text['train_'+key],timestep,sigma,cfg['seed']+3+i*997)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,cfg['grad_clip'],error_if_nonfinite=True)
            optim.step();torch.cuda.synchronize();seconds=time.perf_counter()-start
            loss_value=float(loss.detach())
            if not torch.isfinite(loss.detach()):raise RuntimeError(f'Nonfinite loss at step {i+1}')
            allocated=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30
            max_allocated=max(max_allocated,allocated);max_reserved=max(max_reserved,reserved)
            timing_included=i>=a.resume_step+10
            if timing_included:times.append(seconds)
            log('step',step=i+1,image=key,timestep_index=index,loss=loss_value,grad_norm=float(norm.detach()),
                seconds=seconds,timing_included=timing_included,peak_allocated_GiB=allocated,peak_reserved_GiB=reserved)
            if (i+1)%a.save_every==0 or i+1==a.steps:
                save_lora(out/f'adapter-{i+1:04d}.pt',model,optimizer=optim,base_id=base_id)
                log('checkpoint',step=i+1)
            if not a.skip_validation and ((i+1)%a.validate_every==0 or i+1==a.steps):
                log('validation',step=i+1,**validate(model,latents['04'],text['heldout_04'],scheduler))
        after=frozen_digest(model)
        if before!=after:raise RuntimeError('Frozen model tensors changed during LoRA training')
        report=dict(status='passed',steps=a.steps,start_step=a.resume_step,segment_steps=a.steps-a.resume_step,wall_seconds=time.perf_counter()-start_run,
            measured_steps=len(times),median_step_seconds=statistics.median(times) if times else None,
            mean_step_seconds=statistics.mean(times) if times else None,peak_allocated_GiB=max_allocated,
            peak_reserved_GiB=max_reserved,frozen_sha256_after=after,frozen_unchanged=True)
        (out/'summary.json').write_text(json.dumps(report,indent=2));log('complete',**report)
    except BaseException as e:
        log('failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
