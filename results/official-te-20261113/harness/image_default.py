"""Archived FLUX workload with official TE defaults and ordinary LoRA algebra.

Loads the pinned BF16 source checkpoint, as required by default TE Linear.
Does not import packed weights into TE or implement compressed/cache strategies.
"""
import json,sys,types,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from measure import ROOT,DeviceMemory,write,memory
import te_default
import precision_lora
from precision_lora.core import LoRALinear

assets=ROOT.parent/'image-lora-klein-20261109'
scripts=ROOT/'project/examples/flux2_klein'
bootstrap=types.ModuleType('bootstrap');bootstrap.ROOT=assets;bootstrap.torch=torch
def configure_attention(backend):
    torch.use_deterministic_algorithms(backend=='efficient')
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(backend=='cudnn')
    torch.backends.cuda.enable_mem_efficient_sdp(backend=='efficient')
    torch.backends.cuda.enable_math_sdp(False)
bootstrap.configure_attention=configure_attention
sys.modules['bootstrap']=bootstrap

def load_source(name,file):
    m=types.ModuleType(name);m.__file__=str(file);sys.modules[name]=m
    exec(compile(file.read_text(),str(file),'exec'),m.__dict__)
    return m

loader=load_source('klein_loader',scripts/'klein_loader.py')
source_load=loader.load_model
loader.load_model=lambda mode='bf16',backward_mode=None:source_load('bf16')
def forward(self,x):
    with te_default.context():base=self.base(x)
    return base+F.linear(F.linear(x,self.lora_A),self.lora_B)*(self.alpha/self.rank)
def setup(model,**kwargs):
    kwargs['mode']='preserve'
    model,opt=precision_lora.prepare_lora(model,**kwargs)
    count=0
    for layer in model.modules():
        if isinstance(layer,LoRALinear):
            layer.base=te_default.PaddedLinear(layer.base)
            layer.forward=types.MethodType(forward,layer);count+=1
    assert count==98,count
    print('OFFICIAL_TE_PROJECTIONS',count,flush=True)
    return model,opt

train=load_source('train',scripts/'train.py');train.prepare_lora=setup
action=sys.argv.pop(1)
out=ROOT/'image-default';out.mkdir(exist_ok=True)
if action=='bench':
    m=load_source('matched_flux_benchmark',scripts/'benchmark.py');m.prepare_lora=setup;m.ROOT=out
else:
    # Redirect only the output destination; all data/protocol reads remain unchanged.
    code=(scripts/'train.py').read_text().replace("out=ROOT/'results'/a.label", "out=BENCH_OUTPUT/'results'/a.label")
    m=types.ModuleType('matched_flux_train');m.BENCH_OUTPUT=out
    exec(compile(code,str(scripts/'train.py'),'exec'),m.__dict__);m.prepare_lora=setup
monitor=DeviceMemory()
report=dict(action=action,backend='Official TE NVFP4 default recipe, no persistent weight cache; BF16 frozen parameters; ordinary FP32 LoRA + fused torch AdamW',
            differences=['TE starts from pinned BF16 source checkpoint; project NVFP4 rows use published prequantized checkpoint payload',
                         'WSL Linux rather than archived native Windows; matched PyTorch/CUDA/diffusers/transformers versions',
                         'Official H16/default 2D weight scaling instead of project H128/1D packed layout'],
            recipe=repr(te_default.RECIPE),torch=torch.__version__,cuda=torch.version.cuda)
try:m.main();report['status']='complete'
except Exception as exc:
    report.update(status='oom' if isinstance(exc,torch.OutOfMemoryError) else 'failed',error=repr(exc),traceback=traceback.format_exc(),memory_failure=memory())
    print(report['traceback'],flush=True)
finally:
    report['device_memory']=monitor.finish();write(out/(action+'-status.json'),report)
raise SystemExit(0 if report['status']=='complete' else 1)
