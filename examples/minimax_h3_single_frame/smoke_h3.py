"""Real-weight execution smoke test. Synthetic inputs; NOT a fine-tuning result."""
from bootstrap import ROOT,torch
from h3_runtime import load_model
from musubi_tuner.minimax_h3.packing import build_h3_layout
import json,time,faulthandler,sys,argparse
ap=argparse.ArgumentParser();ap.add_argument('--cache-gib',type=float,default=0.0)
ap.add_argument('--profile',action='store_true');ap.add_argument('--output',default='h3-real-weight-smoke.json')
args=ap.parse_args()
faulthandler.dump_traceback_later(180,repeat=True)
torch.manual_seed(10843)
model=load_model('nvfp4',20,8,True,args.cache_gib).train()
params=[p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.AdamW(params,lr=1e-6,fused=True)
video=torch.randn(1,24,1,32,32,device='cuda')
audio=torch.randn(1,32,2,2,device='cuda')
hidden=torch.randn(1,128,5120,device='cuda',dtype=torch.bfloat16)
tags=torch.zeros(1,128,device='cuda',dtype=torch.long)
layout=build_h3_layout(task='t2va',text_length=128,target_video=(1,32,32),target_audio_frames=2,one_frame=True)
records=[]
for i in range(4):
    optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    output=model(video_latents=video,audio_latents=audio,text_hidden_states=hidden,text_token_tags=tags,layout=layout,
                 model_t_video=.5,model_t_audio=.25)
    loss=output.video.float().square().mean()
    assert torch.isfinite(loss),'Nonfinite synthetic smoke output'
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
    row=dict(step=i+1,seconds=time.perf_counter()-t,loss=float(loss.detach()),grad_norm=float(norm),
             allocated_gib=torch.cuda.memory_allocated()/2**30,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
             peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    records.append(row);print(json.dumps(row),flush=True)
(ROOT/'metadata'/args.output).write_text(json.dumps(dict(status='passed',scope='synthetic-input execution only',cache_gib=args.cache_gib,steps=records),indent=2))
faulthandler.cancel_dump_traceback_later()
if '--profile' in sys.argv:
    from torch.profiler import profile,ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
        optimizer.zero_grad(set_to_none=True)
        output=model(video_latents=video,audio_latents=audio,text_hidden_states=hidden,text_token_tags=tags,layout=layout,
                     model_t_video=.5,model_t_audio=.25)
        loss=output.video.float().square().mean();loss.backward();optimizer.step();torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30),flush=True)
    prof.export_chrome_trace(str(ROOT/'metadata/h3-native-profile.json'))
    events=[]
    for e in prof.key_averages():
        events.append(dict(key=e.key,count=e.count,cpu_us=e.self_cpu_time_total,cuda_us=e.self_device_time_total))
    (ROOT/'metadata/h3-native-profile-summary.json').write_text(json.dumps(sorted(events,key=lambda r:r['cuda_us'],reverse=True),indent=2))
