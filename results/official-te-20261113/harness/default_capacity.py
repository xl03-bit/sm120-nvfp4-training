"""Attempt unchanged large-model default storage; report OOM without downscaling."""
import argparse,json,struct,traceback
from pathlib import Path
import torch
from measure import ROOT,DeviceMemory,memory,module,write

def main(task):
    monitor=DeviceMemory();report=dict(task=task,status='running',memory_initial=memory())
    try:
        torch.cuda.reset_peak_memory_stats()
        if task=='4b':
            import te_default
            src=module(ROOT/'project/results/fineweb-4b-2048/train.py','large_source')
            report['intended_config']=src.CONFIG
            report['changed_policy']='Official default FP32 parameters/moments, no project FP8 storage or transpose rematerialization'
            gen=torch.Generator(device='cuda').manual_seed(1234)
            model,next_block=src.model_parts(2560,50,512,gen)
            for i in range(50):
                report['allocating_block']=i
                model.blocks.add_module(str(i),te_default.convert(next_block()))
            report['loaded_parameters']=sum(p.numel() for p in model.parameters())
            model,opt=te_default.prepare(model)
            opt.zero_grad(set_to_none=True);opt.step()
            report['status']='capacity_passed_training_not_run'
        else:
            path=ROOT.parent/'h3-lora-20261109/models/minimax_h3_ref2va_pruned_nvfp4.safetensors'
            with path.open('rb') as f:
                size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
            quant=sorted(k[:-12] for k in header if k.endswith('.comfy_quant'))
            shapes={stem: [header[stem+'.weight']['shape'][0],header[stem+'.weight']['shape'][1]*2] for stem in quant}
            report.update(quantized_projections=len(shapes),dense_base_dtype='bfloat16',
                          dense_quantized_projection_bytes=sum(m*n*2 for m,n in shapes.values()),
                          source=str(path),probe='Projection storage allocation only; no changed model scale, optimizer, offload or training claim')
            weights=[]
            for name,shape in shapes.items():
                report['allocating_projection']=name
                weights.append(torch.empty(shape,device='cuda',dtype=torch.bfloat16))
                # Touch storage so NVML can observe residency.
                weights[-1].zero_();torch.cuda.synchronize()
            report['status']='capacity_passed_training_not_run'
    except Exception as exc:
        report.update(status='oom' if isinstance(exc,torch.OutOfMemoryError) else 'failed',
                      error=repr(exc),traceback=traceback.format_exc(),memory_failure=memory())
        print(report['traceback'],flush=True)
    finally:
        report['device_memory']=monitor.finish();write(ROOT/f'results/default-capacity-{task}.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('traceback','device_memory')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['4b','h3']);main(p.parse_args().task)
