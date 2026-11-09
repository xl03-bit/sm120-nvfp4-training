"""Use the original BF16 text encoder and VAE once; store identical inputs for all modes."""
from bootstrap import ROOT,torch,deterministic
from klein_loader import BASE
from diffusers import Flux2KleinPipeline,AutoencoderKLFlux2
from transformers import Qwen3ForCausalLM,Qwen2TokenizerFast
from PIL import Image
import numpy as np,json,gc,argparse,time

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['text','vae'],required=True);a=p.parse_args()
    deterministic();torch.manual_seed(7849)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    out=ROOT/'cache';out.mkdir(exist_ok=True)
    if a.part=='text':
        model=Qwen3ForCausalLM.from_pretrained(BASE/'text_encoder',dtype=torch.bfloat16,
            local_files_only=True,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        tokenizer=Qwen2TokenizerFast.from_pretrained(BASE/'tokenizer',local_files_only=True)
        result={}
        for key,prompt in protocol['prompts'].items():
            start=time.perf_counter()
            emb=Flux2KleinPipeline._get_qwen3_prompt_embeds(model,tokenizer,prompt,device='cuda',dtype=torch.bfloat16)
            if not torch.isfinite(emb).all():raise ValueError('Nonfinite text embedding: '+key)
            result[key]=dict(prompt=prompt,embeds=emb.cpu(),ids=Flux2KleinPipeline._prepare_text_ids(emb).cpu())
            print(json.dumps(dict(event='text_cached',key=key,seconds=time.perf_counter()-start)),flush=True)
        torch.save(result,out/'text.pt')
    else:
        vae=AutoencoderKLFlux2.from_pretrained(BASE/'vae',torch_dtype=torch.float32,
            local_files_only=True).to('cuda').eval().requires_grad_(False)
        # Match upstream training: capture FP32 BN statistics before casting
        # the frozen VAE to BF16. Normalized training latents stay FP32.
        mean=vae.bn.running_mean.reshape(1,-1,1,1).clone()
        std=(vae.bn.running_var.reshape(1,-1,1,1)+vae.config.batch_norm_eps).sqrt()
        vae.to(dtype=torch.bfloat16)
        result={}
        for name in protocol['train']+protocol['heldout']:
            key=name.split('.')[0]
            im=Image.open(ROOT/f'data/monster_toy/crops{protocol["resolution"]}'/f'{key}.png').convert('RGB')
            x=torch.from_numpy(np.array(im)).permute(2,0,1).unsqueeze(0).cuda().to(torch.bfloat16)/127.5-1
            raw=vae.encode(x).latent_dist.mode()
            z=(Flux2KleinPipeline._patchify_latents(raw)-mean)/std
            if not torch.isfinite(z).all():raise ValueError('Nonfinite latent: '+key)
            result[key]=dict(latent=z.cpu(),ids=Flux2KleinPipeline._prepare_latent_ids(z).cpu())
            print(json.dumps(dict(event='latent_cached',key=key,shape=list(z.shape))),flush=True)
        torch.save(result,out/'latents.pt')
    print(json.dumps(dict(event='cache_complete',part=a.part,peak_GiB=torch.cuda.max_memory_allocated()/2**30)),flush=True)
if __name__=='__main__':main()
