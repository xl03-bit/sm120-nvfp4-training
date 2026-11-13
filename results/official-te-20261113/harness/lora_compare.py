"""Reuse the repository LoRA workload and timing loops with official TE bases."""
import argparse, json, types, traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from measure import ROOT,DeviceMemory,write
import te_default as te_adapter
from nvfp4_training import prepare_lora
from nvfp4_training.lora import LoRALinear

source=ROOT/'project/tests/benchmark_lora.py'
text=source.read_text().split("if __name__=='__main__':")[0]
# Keep the two NVFP4 rows, with and without the original block checkpointing.
text=text.replace("[('reference',False),('preserve',False),('preserve',True),('nvfp4',False),('nvfp4',True)]",
                  "[('nvfp4',False),('nvfp4',True)]")
text=text.replace("for execution in ('eager','graph'):\n", "for execution in ('eager','graph'):\n                row_monitor=DeviceMemory()\n")
text=text.replace("report['rows'].append(row);write(args.output,report)",
                  "row['device_memory']=row_monitor.finish();report['rows'].append(row);write(args.output,report)",1)
text=text.replace("    for mode,checkpoint in [('nvfp4',False),('nvfp4',True)]:\n        if any", 
                  "    for mode,checkpoint in [('nvfp4',False),('nvfp4',True)]:\n        row_monitor=DeviceMemory()\n        if any")
text=text.replace("row['heldout_improved']=", "row['device_memory']=row_monitor.finish()\n        row['heldout_improved']=")
text=text.replace("                ops=sorted(prof.key_averages()", "                prof.export_chrome_trace(str(Path(args.output).with_suffix('.trace.json')))\n                ops=sorted(prof.key_averages()")
ns={'__name__':'matched_lora','DeviceMemory':DeviceMemory};exec(compile(text,str(source),'exec'),ns)
original_environment=ns['environment']
def environment():
    import transformer_engine
    return dict(**original_environment(),transformer_engine=transformer_engine.__version__,
                official_nvfp4_recipe=repr(te_adapter.RECIPE),
                frozen_base_storage='same BF16 source retained; official default is_first_microbatch=None, no persistent quantized weight cache',
                base_compute='official TE forward/dgrad; FP32 LoRA parameters and unchanged fused AdamW')
ns['environment']=environment

def official_forward(self,x):
    with te_adapter.context():
        base=self.base(x)
    return base+F.linear(F.linear(x,self.lora_A),self.lora_B)*(self.alpha/self.rank)

def setup(model,mode,rank,checkpoints=()):
    model,opt=prepare_lora(model,mode='preserve',rank=rank,checkpoint_modules=checkpoints,lr=2e-4)
    for m in model.modules():
        if isinstance(m,LoRALinear):
            m.base=te_adapter.PaddedLinear(m.base)
            m.forward=types.MethodType(official_forward,m)
    return model,opt
ns['setup']=setup

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['bench','fineweb']);p.add_argument('--output',required=True)
    a=p.parse_args();a.steps=512;a.quick=False;a.resume=False
    monitor=DeviceMemory();status='complete';error=None
    try:ns[a.task](a)
    except Exception:
        status='failed';error=traceback.format_exc();print(error,flush=True)
    write(str(a.output)+'.status.json',dict(status=status,error=error,device_memory=monitor.finish(),
          backend='official TE NVFP4; BF16 frozen weight retained plus official quantized workspace; ordinary two-linear adapter expression',
          recipe=repr(te_adapter.RECIPE)))
    raise SystemExit(0 if status=='complete' else 1)
