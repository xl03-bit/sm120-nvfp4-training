"""Matched LoRA performance/convergence on a real GPU; no downloaded dependencies.

Examples:
  python tests/benchmark_lora.py bench --output results/lora/bench.json
  LORA_GPT2=/path/to/gpt2 NVFP4_FINEWEB=/path/to/shard.bin python tests/benchmark_lora.py fineweb --steps 512 --output results/lora/fineweb.json
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import time
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from nvfp4_training import prepare_lora, capture_lora_step, lora_parameters
from nvfp4_training.lora import LoRALinear
from nvfp4_training._workloads.synthetic import Network

GIB=2**30
torch.set_num_threads(4)


def reference_forward(self,x):
    return F.linear(x,self.base.weight,self.base.bias)+F.linear(F.linear(x,self.lora_A),self.lora_B)*(self.alpha/self.rank)


def setup(model,mode,rank,checkpoints=()):
    model,opt=prepare_lora(model,mode='preserve' if mode=='reference' else mode,rank=rank,
                           checkpoint_modules=checkpoints,lr=2e-4)
    if mode=='reference':
        for module in model.modules():
            if isinstance(module,LoRALinear):module.forward=types.MethodType(reference_forward,module)
    return model,opt


def step(model,opt,fn,x,y):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=fn(model,x,y)
    loss.backward();opt.step()
    return loss.detach()


def mem():
    return dict(allocated_gib=torch.cuda.memory_allocated()/GIB,reserved_gib=torch.cuda.memory_reserved()/GIB,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB)


def environment():
    return dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,seed=1234,
                base_dtype='bfloat16',adapter_dtype='float32',adamw_moments='float32',rank=16,alpha=16,
                optimizer='fused AdamW',lr=2e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,
                timing='synchronized wall, forward+loss+backward+AdamW; graph includes device input copies')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def bench(args):
    report=dict(environment=environment(),warmup=12,rounds=3,steps_per_round=20,rows=[])
    configs=[(1536,16,2,64),(1536,16,2,512),(1024,4,1,257)]
    if args.quick:configs=configs[:1]
    for d,depth,b,t in configs:
        for mode,checkpoint in [('reference',False),('preserve',False),('preserve',True),('nvfp4',False),('nvfp4',True)]:
            torch.manual_seed(1234);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            model=Network(d,depth,d//64).bfloat16().cuda()
            base_count=sum(p.numel() for p in model.parameters())
            model,opt=setup(model,mode,16,[f'blocks.{i}' for i in range(depth)] if checkpoint else [])
            x=torch.randn(b,t,d,device='cuda',dtype=torch.bfloat16);y=torch.randn_like(x)
            fn=lambda m,x,y:(m(x).float()-y).square().mean()
            construction=mem()
            for execution in ('eager','graph'):
                graph=capture_lora_step(model,opt,fn,x,y) if execution=='graph' else None
                capture_memory=mem()
                execute=(lambda:graph.step(x,y)) if graph else (lambda:step(model,opt,fn,x,y))
                for _ in range(12):execute()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                rounds=[]
                for _ in range(3):
                    times=[]
                    for _ in range(20):
                        start=time.perf_counter();loss=execute();torch.cuda.synchronize()
                        times.append((time.perf_counter()-start)*1000)
                    rounds.append(times)
                assert torch.isfinite(loss), (mode,checkpoint,execution)
                times=[v for r in rounds for v in r]
                row=dict(mode=mode,checkpoint=checkpoint,execution=execution,shape=[b,t,d,depth],parameters=base_count,
                         adapter_parameters=sum(p.numel() for p in lora_parameters(model)),step_ms=statistics.mean(times),
                         median_ms=statistics.median(times),tokens_per_second=b*t*1000/statistics.mean(times),
                         memory=mem(),construction_memory=construction,capture_memory=capture_memory,rounds_ms=rounds)
                report['rows'].append(row);write(args.output,report)
                print({k:v for k,v in row.items() if k not in ('rounds_ms','construction_memory')},flush=True)
                if graph:graph.close();del graph
                del execute
            # Actual profiling, retaining operator costs for inspection.
            if (d,depth,b,t)==configs[0] and not checkpoint:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(3):step(model,opt,fn,x,y)
                    torch.cuda.synchronize()
                ops=sorted(prof.key_averages(),key=lambda a:a.self_device_time_total,reverse=True)
                report.setdefault('profiles',{})[mode]=[dict(name=a.key,calls=a.count,gpu_us=a.self_device_time_total,cpu_us=a.self_cpu_time_total) for a in ops[:30]]
                write(args.output,report)
            del model,opt,x,y,loss;gc.collect();torch.cuda.empty_cache()
    report['status']='passed';write(args.output,report)


class GPTBlock(nn.Module):
    def __init__(self,d,heads):
        super().__init__();self.heads=heads
        self.ln_1=nn.LayerNorm(d,eps=1e-5);self.ln_2=nn.LayerNorm(d,eps=1e-5)
        self.qkv=nn.Linear(d,3*d);self.proj=nn.Linear(d,d);self.up=nn.Linear(d,4*d);self.down=nn.Linear(4*d,d)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(self.ln_1(x)).reshape(b,t,3,self.heads,d//self.heads).unbind(2)
        out=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True,dropout_p=0)
        x=x+self.proj(out.transpose(1,2).reshape(b,t,d))
        return x+self.down(F.gelu(self.up(self.ln_2(x)),approximate='tanh'))


class GPT2(nn.Module):
    def __init__(self,path):
        super().__init__()
        cfg=json.loads((path/'config.json').read_text());d=cfg['n_embd']
        self.wte=nn.Embedding(cfg['vocab_size'],d);self.wpe=nn.Embedding(cfg['n_positions'],d)
        self.blocks=nn.Sequential(*(GPTBlock(d,cfg['n_head']) for _ in range(cfg['n_layer'])))
        self.ln_f=nn.LayerNorm(d,eps=cfg['layer_norm_epsilon']);self.head=nn.Linear(d,cfg['vocab_size'],bias=False);self.head.weight=self.wte.weight
        # Safetensors is a JSON header and contiguous raw arrays. Only F32
        # weights are read; causal-mask buffers in the archive are unused by SDPA.
        file=path/'model.safetensors'
        with file.open('rb') as f:
            header_bytes=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(header_bytes))
        def value(key):
            meta=header[key];assert meta['dtype']=='F32'
            data=np.memmap(file,mode='r',dtype='<f4',offset=8+header_bytes+meta['data_offsets'][0],shape=tuple(meta['shape']))
            return torch.from_numpy(data.copy())
        with torch.no_grad():
            for name,p in self.named_parameters():
                key=name
                if name.startswith('blocks.'):
                    key='h.'+name[len('blocks.'):]
                    for src,dst in [('qkv','attn.c_attn'),('proj','attn.c_proj'),('up','mlp.c_fc'),('down','mlp.c_proj')]:
                        if f'.{src}.' in key:
                            key=key.replace(f'.{src}.',f'.{dst}.');break
                source=value(key)
                if name.startswith('blocks.') and any(f'.{a}.weight' in name for a in ('qkv','proj','up','down')):source=source.t()
                assert source.shape==p.shape,(name,key,source.shape,p.shape)
                p.copy_(source)
    def forward(self,x):
        h=self.wte(x)+self.wpe(torch.arange(x.shape[1],device=x.device))
        return self.head(self.ln_f(self.blocks(h)))


def fineweb(args):
    source=Path(os.environ['LORA_GPT2']);shard=Path(os.environ['NVFP4_FINEWEB'])
    digest=hashlib.sha256(shard.read_bytes()).hexdigest()
    assert digest=='771fa4a99b9fe0946ffb6e848b4ba5c6a9b0fe87860ebf03bc2c1c7e45f8178e'
    data=np.memmap(shard,mode='r',dtype='<u2',offset=1024)
    b,t=2,256
    def batches(lo,hi,count,seed):
        starts=np.random.default_rng(seed).integers(lo,hi-t,size=(count,b))
        all_tokens=torch.tensor(data[starts[:,:,None]+np.arange(t+1)].astype('int64'),device='cuda')
        return [(v[:,:-1].contiguous(),v[:,1:].contiguous()) for v in all_tokens]
    train=batches(0,95_000_000,args.steps,701);valid=batches(95_001_000,len(data),16,702)
    report=dict(environment=environment(),model='pretrained GPT-2 124M, transposed Conv1D weights in nn.Linear',
                dropout=0,attention='causal PyTorch SDPA',steps=args.steps,batch=b,sequence=t,dataset_sha256=digest,
                model_sha256=hashlib.sha256((source/'model.safetensors').read_bytes()).hexdigest(),rows=[])
    if args.resume and Path(args.output).exists():
        previous=json.loads(Path(args.output).read_text())
        assert {k:v for k,v in previous.items() if k not in ('rows','status')}=={k:v for k,v in report.items() if k!='rows'}
        assert all(r.get('frozen_base_unchanged') and len(r['training'])==args.steps for r in previous['rows'])
        report['rows']=previous['rows']
    fn=lambda m,x,y:F.cross_entropy(m(x).float().flatten(0,1),y.flatten())
    @torch.no_grad()
    def evaluate(model):
        with torch.autocast('cuda',dtype=torch.bfloat16):values=[fn(model,x,y).item() for x,y in valid]
        assert all(map(math.isfinite,values));return statistics.mean(values)
    for mode,checkpoint in [('reference',False),('preserve',False),('preserve',True),('nvfp4',False),('nvfp4',True)]:
        if any(r['mode']==mode and r['checkpoint']==checkpoint for r in report['rows']):continue
        torch.manual_seed(1234);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        model=GPT2(source).cuda().bfloat16()
        params=sum(p.numel() for p in model.parameters())
        model,opt=setup(model,mode,16,[f'blocks.{i}' for i in range(12)] if checkpoint else [])
        # Hash frozen tensors before/after on CPU outside timed training.
        def frozen_hash():
            digest=hashlib.sha256()
            for n,p in model.named_parameters():
                if not p.requires_grad:digest.update(p.detach().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
            for n,p in model.named_buffers():
                if not n.endswith('sr_seed'):digest.update(p.detach().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()
        before=frozen_hash()
        row=dict(mode=mode,checkpoint=checkpoint,parameters=params,adapter_parameters=sum(p.numel() for p in lora_parameters(model)),
                 validation=[dict(step=0,loss=evaluate(model))],training=[])
        report['rows'].append(row);write(args.output,report)
        print(mode,checkpoint,'validation',row['validation'][0],flush=True)
        with capture_lora_step(model,opt,fn,*train[0]) as graph:
            for i,(x,y) in enumerate(train,1):
                start=time.perf_counter();loss=graph.step(x,y);torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
                value=loss.item();assert math.isfinite(value)
                row['training'].append(dict(step=i,loss=value,ms=ms,**mem()))
                if i%64==0 or i==args.steps:
                    row['validation'].append(dict(step=i,loss=evaluate(model)))
                    print(mode,checkpoint,'validation',row['validation'][-1],flush=True);write(args.output,report)
        assert before==frozen_hash(),'frozen base changed'
        assert all(torch.isfinite(p).all() and torch.isfinite(p.grad).all() if p.grad is not None else torch.isfinite(p).all() for p in lora_parameters(model))
        assert all(s['step'].item()==args.steps for s in opt.state.values())
        row['frozen_base_unchanged']=True;row['memory']=mem();row['mean_step_ms']=statistics.mean(r['ms'] for r in row['training'][12:])
        row['heldout_improved']=row['validation'][-1]['loss']<row['validation'][0]['loss']
        assert row['heldout_improved'],(mode,row['validation'])
        write(args.output,report)
        del model,opt,graph;gc.collect();torch.cuda.empty_cache()
    report['status']='passed';write(args.output,report)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['bench','fineweb']);ap.add_argument('--output',required=True)
    ap.add_argument('--steps',type=int,default=512);ap.add_argument('--quick',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    globals()[args.task](args)
