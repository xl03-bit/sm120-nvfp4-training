"""Export adapter-only training checkpoints in Diffusers' documented LoRA format."""
from bootstrap import torch
from diffusers import Flux2KleinPipeline
import argparse

def export(source,output):
    saved=torch.load(source,map_location='cpu',weights_only=True)
    cfg=saved['config'];weights={}
    for name,value in saved['adapters'].items():
        weights[name+'.lora_A.weight']=value['A'].contiguous()
        weights[name+'.lora_B.weight']=value['B'].contiguous()
    Flux2KleinPipeline.save_lora_weights(output,transformer_lora_layers=weights,
        transformer_lora_adapter_metadata=dict(r=cfg['rank'],lora_alpha=cfg['alpha'],
            target_modules=cfg['targets'],bias='none',use_dora=False,use_rslora=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();export(a.source,a.output)
