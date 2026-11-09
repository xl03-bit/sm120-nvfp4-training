from bootstrap import ROOT,torch
import argparse,gc,json,time
from pathlib import Path
import numpy as np
from PIL import Image
P=argparse.ArgumentParser();P.add_argument('phase',choices=['latents','text','decode']);P.add_argument('--source')
P.add_argument('--text-blocks-to-swap',type=int,default=20);args=P.parse_args()
cache=ROOT/'cache';cache.mkdir(exist_ok=True)
data=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf8'))
def log(**kw):
    print(json.dumps(kw),flush=True)
    with (cache/'preparation.jsonl').open('a') as f:f.write(json.dumps(kw)+'\n')
if args.phase=='latents':
    from musubi_tuner.minimax_h3.video_vae import load_video_vae,encode_video_target
    from musubi_tuner.minimax_h3.audio_vae import load_audio_vae,encode_audio_mode
    vae=load_video_vae(ROOT/'models/vae/minimax_h3_video_vae_fp16.safetensors',device='cuda',dtype=torch.float32,load_decoder=False)
    for r in data:
        dst=cache/f"image-{r['id']:03d}.pt"
        if dst.exists():continue
        t=time.monotonic();im=np.array(Image.open(ROOT/r['path']).convert('RGB'))
        pixels=torch.from_numpy(im).permute(2,0,1).unsqueeze(0).unsqueeze(2).float().cuda()/127.5-1
        z=encode_video_target(vae,pixels,20261109,str(r['id'])).cpu()
        assert z.shape==(1,24,1,32,32) and torch.isfinite(z).all()
        torch.save(z,dst);log(phase='latent',id=r['id'],seconds=time.monotonic()-t)
    del vae,pixels,z;gc.collect();torch.cuda.empty_cache()
    avae=load_audio_vae(ROOT/'models/vae/minimax_h3_audio_vae_fp32.safetensors',device='cuda')
    with torch.no_grad():silence=encode_audio_mode(avae,torch.zeros(1,2,1600,device='cuda')).cpu()
    assert silence.shape==(1,32,2,2),silence.shape
    torch.save(silence,cache/'silence.pt');log(phase='latents_complete',peak_gib=torch.cuda.max_memory_allocated()/2**30)
elif args.phase=='text':
    from transformers import Qwen3VLProcessor
    from musubi_tuner.minimax_h3 import text_encoder as te
    import musubi_tuner.modules.nvfp4_utils as nvutils
    import nv4_cuda
    te.H3_REPO_ID=str(ROOT/'official')
    processor=Qwen3VLProcessor.from_pretrained(str(ROOT/'official/processor'),local_files_only=True)
    def fast_dequant(packed,bs,gs,shape,dtype):
        assert dtype==torch.bfloat16
        return nv4_cuda.dequant(packed,bs,gs)[:shape[0],:shape[1]]
    nvutils.dequantize_nvfp4=fast_dequant
    t=time.monotonic()
    model=te.load_h3_text_encoder(ROOT/'models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',device='cuda',
                                 blocks_to_swap=args.text_blocks_to_swap,attn_mode='sdpa',disable_numpy_memmap=True)
    log(phase='text_model_loaded',seconds=time.monotonic()-t)
    probes=[
        ('style-fox','A cute blue fox creature with leaf-shaped ears and a curled tail, full body, in txmnstyle.'),
        ('style-dragon','A small orange dragon wearing a green scarf, full body, in txmnstyle.'),
        ('style-robot','A friendly round robot with one large eye and small arms, full body, in txmnstyle.'),
        ('control-photo','A realistic photograph of a tabby cat sitting on a wooden chair beside a sunny window.'),
        ('control-landscape','A delicate watercolor painting of a quiet mountain lake with pine trees and distant snow.'),
        ('control-portrait','A realistic studio photograph of an adult woman in a navy sweater, soft window light.'),
    ]
    prompts={f"item-{r['id']:03d}":r['prompt'] for r in data}
    prompts.update({k:f'integrated_multimodal_description: [Shot 1] {v} overall_soundscape: Silent. non_diegetic_music: None.' for k,v in probes})
    prompts['uncond']=' '
    (cache/'prompts.json').write_text(json.dumps(prompts,indent=2))
    for name,prompt in prompts.items():
        dst=cache/(name+'.pt')
        if dst.exists():continue
        t=time.monotonic();hidden,tags=te.encode_h3_presentation(processor,model,te.H3Presentation(text=prompt))
        assert torch.isfinite(hidden).all()
        torch.save(dict(hidden=hidden.cpu(),tags=tags.cpu(),prompt=prompt),dst)
        log(phase='text',name=name,rows=len(hidden),seconds=time.monotonic()-t)
    log(phase='text_complete',peak_gib=torch.cuda.max_memory_allocated()/2**30)
else:
    from musubi_tuner.minimax_h3.video_vae import load_video_vae
    source=Path(args.source) if args.source else ROOT/'results'
    files=sorted(source.rglob('sample-*.pt'))
    assert files,'No saved generated latents'
    vae=load_video_vae(ROOT/'models/vae/minimax_h3_video_vae_fp16.safetensors',device='cuda',dtype=torch.float16)
    for p in files:
        out=p.with_suffix('.png')
        if out.exists():continue
        t=time.monotonic();z=torch.load(p,map_location='cpu',weights_only=True)['video']
        with torch.no_grad():image=vae.decode(z.cuda().half())[0,:,0].float().cpu()
        assert torch.isfinite(image).all()
        image=((image.clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).numpy()
        Image.fromarray(image).save(out);log(phase='decode',file=str(out),seconds=time.monotonic()-t)
