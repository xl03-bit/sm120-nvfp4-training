"""Load the pinned BFL checkpoint through the existing public LoRA API.

The BFL-to-Diffusers name conversion is the installed Apache-2.0 Diffusers
implementation. This file contains format integration, not new GEMM kernels.
This first portability recipe explicitly uses dynamic activation quantization;
checkpoint input_scale scalars are calibration values, NOT AWQ input vectors.
"""
from bootstrap import ROOT, torch
from diffusers import Flux2Transformer2DModel
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers as convert
from precision_lora import PackedNVFP4Linear
from safetensors import safe_open
import json, time

BASE = ROOT / 'models/base'
NV4 = ROOT / 'models/nvfp4/flux-2-klein-base-4b-nvfp4.safetensors'
LAYOUT = 'comfy-high-first-swizzle128x4'

def mapped_names(stem):
    # QKV uses three contiguous 128-row-aligned slices, including scale bytes.
    return [k.removesuffix('.weight') for k in convert({stem+'.weight':torch.empty(3,1,device='meta')})]

def targets():
    with safe_open(NV4,framework='pt',device='cpu') as f:
        layers=json.loads(f.metadata()['_quantization_metadata'])['layers']
    return sorted(n for stem in layers for n in mapped_names(stem))

def load_model(mode='bf16', backward_mode='recompute'):
    start=time.perf_counter()
    if mode=='bf16':
        model=Flux2Transformer2DModel.from_pretrained(BASE/'transformer',torch_dtype=torch.bfloat16,
            local_files_only=True).to('cuda').requires_grad_(False)
    elif mode=='nvfp4':
        with safe_open(NV4,framework='pt',device='cpu') as f:
            layers=json.loads(f.metadata()['_quantization_metadata'])['layers']
            if any(v!={'format':'nvfp4'} for v in layers.values()):
                raise ValueError('Unexpected quantization metadata')
            normal={k:f.get_tensor(k) for k in f.keys() if k.rsplit('.',1)[0] not in layers}
            normal=convert(normal)
            with torch.device('meta'):
                model=Flux2Transformer2DModel.from_config(Flux2Transformer2DModel.load_config(BASE/'transformer'))
                model.to(dtype=torch.bfloat16)
            missing,unexpected=model.load_state_dict({k:v.to('cuda') for k,v in normal.items()},strict=False,assign=True)
            del normal
            expected={n+'.weight' for stem in layers for n in mapped_names(stem)}
            if set(missing)!=expected or unexpected:
                raise ValueError(f'Checkpoint map mismatch: {set(missing)^expected}, {unexpected}')
            calibration={}
            for i,stem in enumerate(layers):
                q=f.get_tensor(stem+'.weight')
                s=f.get_tensor(stem+'.weight_scale')
                g=f.get_tensor(stem+'.weight_scale_2').cuda()
                calibration[stem]=float(f.get_tensor(stem+'.input_scale'))
                names=mapped_names(stem)
                if q.shape[0]%len(names) or (q.shape[0]//len(names))%128:
                    raise ValueError('QKV split is not a contiguous aligned scale slice')
                qs=q.chunk(len(names),dim=0)
                # Float8 chunk is a view; no arithmetic or requantization.
                ss=s.view(torch.uint8).reshape(q.shape[0],-1).chunk(len(names),dim=0)
                for j,(name,qpart,spart) in enumerate(zip(names,qs,ss)):
                    old=model.get_submodule(name)
                    if old.bias is not None:raise ValueError('Unexpected quantized bias')
                    base=PackedNVFP4Linear(qpart.cuda(),spart.cuda().view(torch.float8_e4m3fn),g,
                        shape=(old.out_features,old.in_features),layout=LAYOUT,
                        backward_mode=backward_mode,seed=i*100003+j)
                    parent,_,leaf=name.rpartition('.')
                    setattr(model.get_submodule(parent),leaf,base)
            model.requires_grad_(False)
            if any(p.device.type!='cuda' for p in model.parameters()):raise ValueError('Unloaded parameters')
            model.source_activation_calibration=calibration
            model.activation_scale_recipe='dynamic per input tensor; source static calibration retained as metadata'
    else:raise ValueError(mode)
    model.eval()
    torch.cuda.synchronize()
    print(json.dumps(dict(event='model_loaded',mode=mode,backward_mode=backward_mode,
        seconds=time.perf_counter()-start,allocated_GiB=torch.cuda.memory_allocated()/2**30,
        activation_scale=getattr(model,'activation_scale_recipe',None))),flush=True)
    return model

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['bf16','nvfp4'],default='nvfp4');a=p.parse_args()
    load_model(a.mode)
