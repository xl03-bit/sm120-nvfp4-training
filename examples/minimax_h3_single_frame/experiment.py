"""Foreground, fail-fast H3 image LoRA experiment; all timing includes AdamW."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from bootstrap import ROOT,CODE_ROOT,LIB,torch
from h3_runtime import load_model,save_adapter,restore_adapter,Adapter,FrozenQuantLinear
from musubi_tuner.minimax_h3.packing import build_h3_layout
from musubi_tuner.minimax_h3.sampling import sample_joint_av_latents,shift_sigma
import argparse,json,time,random,statistics,math,hashlib
from pathlib import Path
from safetensors.torch import save_file

P=argparse.ArgumentParser();P.add_argument('action',choices=['pilot','train','sample'])
P.add_argument('--mode',choices=['nvfp4','native','preserve'],default='nvfp4');P.add_argument('--max-steps',type=int,default=12)
P.add_argument('--schedule-steps',type=int,default=800);P.add_argument('--out',default='nvfp4-main');P.add_argument('--resume',action='store_true')
P.add_argument('--rank',type=int,default=8);P.add_argument('--first-block',type=int,default=20);P.add_argument('--no-checkpoint',action='store_true')
P.add_argument('--seconds-limit',type=float,default=9000);P.add_argument('--sample-steps',type=int,default=20)
P.add_argument('--sample-every',type=int,default=200);P.add_argument('--lr',type=float,default=1e-4)
P.add_argument('--checkpoint-path');P.add_argument('--backward-cache-gib',type=float,default=0.0)
P.add_argument('--attention',choices=['efficient-deterministic','sdpa-auto','math'],default='efficient-deterministic')
args=P.parse_args()
if args.attention!='sdpa-auto':
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(args.attention=='efficient-deterministic')
    torch.backends.cuda.enable_math_sdp(args.attention=='math')
out=ROOT/'results'/args.out;out.mkdir(parents=True,exist_ok=True)
cache=ROOT/'cache';data=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf8'))
train=[r for r in data if r['split']=='train'];heldout=[r for r in data if r['split']=='heldout']
def log(event,**kw):
    row=dict(event=event,time=time.time(),**kw);s=json.dumps(row);print(s,flush=True)
    with (out/'run.jsonl').open('a') as f:f.write(s+'\n')
def read(name):return torch.load(cache/name,map_location='cpu',weights_only=True)
def layout(hidden):return build_h3_layout(task='t2va',text_length=hidden.shape[0],target_video=(1,32,32),target_audio_frames=2,one_frame=True)
inputs={}
for r in data:
    text=read(f"item-{r['id']:03d}.pt");inputs[r['id']]=dict(video=read(f"image-{r['id']:03d}.pt").cuda(),
        hidden=text['hidden'].cuda(),tags=text['tags'].cuda(),layout=layout(text['hidden']))
uncond=read('uncond.pt');uncond={**uncond,'hidden':uncond['hidden'].cuda(),'tags':uncond['tags'].cuda()}
uncond_layout=layout(uncond['hidden']);silence=read('silence.pt').cuda()
model=load_model(args.mode,args.first_block,args.rank,not args.no_checkpoint,args.backward_cache_gib)
model._recipe['attention']=args.attention
model._recipe['training']=dict(lr=args.lr,schedule_steps=args.schedule_steps,seed=20261109,
    guidance_scale=4.,guidance_sigma_threshold=.15,video_shift=1.,audio_shift=3.,
    manifest_sha256=hashlib.sha256((ROOT/'data/manifest.json').read_bytes()).hexdigest(),
    prompts_sha256=hashlib.sha256((cache/'prompts.json').read_bytes()).hexdigest())
params=[p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.AdamW(params,lr=args.lr,betas=(.9,.999),eps=1e-8,weight_decay=0.,fused=True)
last=out/'last.pt';start_step=0
if args.resume:
    extra=restore_adapter(last,model,optimizer);start_step=extra['step']
elif args.checkpoint_path:
    extra=restore_adapter(Path(args.checkpoint_path),model);start_step=extra.get('step',0)
frozen_versions={k:v._version for k,v in list(model.named_parameters())+list(model.named_buffers()) if not v.requires_grad}
def frozen_ok():
    return all(v._version==frozen_versions[k] for k,v in list(model.named_parameters())+list(model.named_buffers()) if not v.requires_grad)
def forward(item,video,audio,u,unconditional=False):
    text=uncond if unconditional else item
    return model(video_latents=video,audio_latents=audio,text_hidden_states=text['hidden'].unsqueeze(0),
        text_token_tags=text['tags'].unsqueeze(0),layout=uncond_layout if unconditional else item['layout'],
        model_t_video=1-u,model_t_audio=1-shift_sigma(u,3.0))
def draw(item,seed,u=None):
    gen=torch.Generator().manual_seed(seed)
    if u is None:u=float(torch.rand((),generator=gen).clamp(.001,.999))
    nv=torch.randn(item['video'].shape,generator=gen).cuda();na=torch.randn(silence.shape,generator=gen).cuda()
    sa=shift_sigma(u,3.0)
    return (1-u)*item['video']+u*nv,(1-sa)*silence+sa*na,item['video']-nv,u
def objective(item,seed,u=None):
    video,audio,target,u=draw(item,seed,u)
    if u>=.15:
        with torch.no_grad():unconditional=forward(item,video,audio,u,True).video.float()
        guided=unconditional+4*(target-unconditional)
    else:guided=target
    prediction=forward(item,video,audio,u).video.float()
    return (prediction-guided).square().mean(),(prediction.detach()-target).square().mean(),u
@torch.no_grad()
def validate(step):
    model.eval();values=[];raw=[];t=time.monotonic()
    for r in heldout[:4]:
        for i,u in enumerate([.25,.75]):
            v,b,_=objective(inputs[r['id']],900000+r['id']*7+i,u);values.append(float(v));raw.append(float(b))
    log('validation',step=step,guided_mse=statistics.mean(values),plain_flow_mse=statistics.mean(raw),seconds=time.monotonic()-t)
    model.train()
@torch.no_grad()
def sample(step):
    model.eval();folder=out/f'step-{step:04d}';folder.mkdir(exist_ok=True)
    for name in ['style-fox','style-dragon','style-robot','control-photo','control-landscape','control-portrait']:
        path=folder/f'sample-{name}.pt'
        if path.exists():continue
        text=read(name+'.pt');t=time.monotonic()
        z=sample_joint_av_latents(model,layout=layout(text['hidden']),seed=424242,text_hidden_states=text['hidden'].unsqueeze(0),
            text_token_tags=text['tags'].unsqueeze(0),steps=args.sample_steps,video_shift=1.,audio_shift=3.,device='cuda')
        assert torch.isfinite(z.video).all() and torch.isfinite(z.audio).all()
        torch.save(dict(video=z.video,audio=z.audio,prompt=text['prompt'],seed=424242,steps=args.sample_steps,
                        mode=args.mode,training_step=step),path)
        log('sample',step=step,name=name,seconds=time.monotonic()-t)
    model.train()
def export(step):
    tensors={}
    for name,m in model.named_modules():
        if isinstance(m,Adapter):
            prefix='lora_unet_'+name.replace('.','_')
            tensors[prefix+'.lora_down.weight']=m.a.detach().cpu().contiguous()
            tensors[prefix+'.lora_up.weight']=m.b.detach().cpu().contiguous()
            tensors[prefix+'.alpha']=torch.tensor(float(m.alpha))
    save_file(tensors,str(out/f'lora-step-{step:04d}.safetensors'),metadata={
        'ss_network_module':'networks.lora_minimax_h3','ss_network_dim':str(args.rank),'ss_network_alpha':str(args.rank),
        'ss_minimax_h3_one_frame':'True','experiment':json.dumps(vars(args)),
        'base_revision':'15646562c19be9d62aee6d28a7070d00d026027c','source_dataset':'diffusers/tuxemon','training_step':str(step)})

code_files=[CODE_ROOT/n for n in ('experiment.py','h3_runtime.py','nv4_cuda.py','cache_inputs.py')]
code_files += [LIB/'precision_lora'/n for n in ('_packed_cuda.py','packed.py','core.py')]
code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files}
log('start',config=vars(args),code_sha256=code_hashes,recipe=model._recipe,torch=torch.__version__,gpu=torch.cuda.get_device_name(),trainable=sum(p.numel() for p in params),
    allocated_gib=torch.cuda.memory_allocated()/2**30,reserved_gib=torch.cuda.memory_reserved()/2**30,start_step=start_step)
if args.action=='sample':
    sample(start_step)
else:
    if start_step==0:validate(0)
    model.train();timings=[];begin=time.monotonic();step=start_step
    for index in range(start_step,args.max_steps):
        if time.monotonic()-begin>=args.seconds_limit:
            log('time_limit',step=step,seconds=time.monotonic()-begin);break
        epoch,index_in_epoch=divmod(index,len(train));order=list(range(len(train)))
        random.Random(20261109+epoch).shuffle(order);r=train[order[index_in_epoch]]
        if index<50:factor=(index+1)/50
        else:factor=.1+.9*.5*(1+math.cos(math.pi*min(1.,(index-50)/max(args.schedule_steps-50,1))))
        for group in optimizer.param_groups:group['lr']=args.lr*factor
        torch.cuda.synchronize();t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        loss,raw,u=objective(inputs[r['id']],20261109+index*997)
        if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss at step {index+1}')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize();seconds=time.perf_counter()-t;timings.append(seconds);step=index+1
        assert frozen_ok(),'A frozen parameter or buffer changed'
        log('step',step=step,image_id=r['id'],loss=float(loss.detach()),plain_flow_mse=float(raw),sigma=u,grad_norm=float(gn),
            lr=args.lr*factor,seconds=seconds,allocated_gib=torch.cuda.memory_allocated()/2**30,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        if step%args.sample_every==0 and args.action=='train':
            save_adapter(last,model,optimizer,step=step);export(step);validate(step);sample(step)
    save_adapter(last,model,optimizer,step=step);export(step)
    if args.action=='train':validate(step);sample(step)
    stable=timings[3:] or timings
    log('complete',step=step,training_seconds=sum(timings),median_seconds=statistics.median(stable),
        estimated_800_steps_minutes=statistics.median(stable)*800/60,frozen_versions_unchanged=frozen_ok())
