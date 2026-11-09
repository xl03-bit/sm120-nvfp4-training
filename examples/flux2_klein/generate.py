"""Labelled, seeded before/after inference through the unmodified Diffusers pipeline."""
from bootstrap import ROOT,torch,configure_attention
from klein_loader import load_model,BASE,targets
from precision_lora import enable_lora,load_lora
from diffusers import Flux2KleinPipeline,AutoencoderKLFlux2,FlowMatchEulerDiscreteScheduler
from torch.nn.attention import sdpa_kernel,SDPBackend
import json,time,argparse

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['bf16','nvfp4'],required=True)
    p.add_argument('--adapter');p.add_argument('--label',required=True);p.add_argument('--keys',nargs='+')
    p.add_argument('--text-cache',default='cache/text.pt',help='Prompt embeddings; defaults to the original unchanged cache')
    p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--attention',choices=['efficient','cudnn'],default='efficient')
    p.add_argument('--memory-fraction',type=float,help='Optional allocator cap for an explicitly concurrent preview');a=p.parse_args()
    if a.memory_fraction is not None:
        if not 0<a.memory_fraction<=1:raise ValueError('memory-fraction must be in (0, 1]')
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    configure_attention(a.attention);protocol=json.loads((ROOT/'protocol.json').read_text())
    if a.mode=='nvfp4' and not (ROOT/'metadata/layout-verification.json').exists():
        raise RuntimeError('Verify official FP4 layout before generating images')
    text=torch.load(ROOT/a.text_cache,weights_only=True)
    keys=a.keys or protocol['baseline_keys']
    for key in ['empty',*keys]:
        if key not in text:raise KeyError(f'Prompt {key} absent from {a.text_cache}')
    out=ROOT/'results'/a.label;out.mkdir(parents=True,exist_ok=False)
    model=load_model(a.mode)
    if a.adapter:
        saved=torch.load(a.adapter,weights_only=True,map_location='cpu')
        config=saved['config'];mode='preserve' if a.mode=='bf16' else 'packed_nvfp4'
        if mode!=config['mode']:raise ValueError('Cross-base adapter transfer requires a separately labelled experiment')
        model=enable_lora(model,mode=mode,rank=config['rank'],alpha=config['alpha'],target_modules=config['targets'])
        load_lora(a.adapter,model,restore_rng=False)
    model.eval()
    vae=AutoencoderKLFlux2.from_pretrained(BASE/'vae',torch_dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    pipe=Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(BASE/'scheduler',local_files_only=True),
        vae=vae,text_encoder=None,tokenizer=None,transformer=model,is_distilled=False)
    pipe.set_progress_bar_config(disable=True)
    cfg=protocol['inference'];records=[]
    (out/'settings.json').write_text(json.dumps(dict(mode=a.mode,adapter=a.adapter,inference=cfg,attention=a.attention,
        activation_scale=getattr(model,'activation_scale_recipe',None),label=a.label,
        memory_fraction=a.memory_fraction,text_cache=a.text_cache),indent=2))
    for key in keys:
        for seed in (a.seeds or protocol['seeds']):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            latents=pipe(prompt_embeds=text[key]['embeds'].cuda(),negative_prompt_embeds=text['empty']['embeds'].cuda(),
                height=cfg['height'],width=cfg['width'],num_inference_steps=cfg['steps'],guidance_scale=cfg['guidance_scale'],
                generator=torch.Generator('cuda').manual_seed(seed),output_type='latent').images
            # The VAE has a different attention head width. Keep its established
            # efficient backend explicitly; the DiT uses the selected backend.
            torch.save(latents.cpu(),out/f'{key}-{seed}-latents.pt')
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                image=pipe.image_processor.postprocess(vae.decode(latents,return_dict=False)[0],output_type='pil')[0]
            torch.cuda.synchronize()
            file=f'{key}-{seed}.png';image.save(out/file)
            r=dict(key=key,prompt=text[key]['prompt'],seed=seed,file=file,seconds=time.perf_counter()-start,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
            records.append(r)
            (out/'images.json').write_text(json.dumps(records,indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':main()
