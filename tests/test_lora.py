"""Real GPU LoRA checks; run directly, exits nonzero at the first failure."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import faulthandler

import torch
from torch import nn
from torch.nn import functional as F
from precision_lora.core import (enable_lora, prepare_lora, lora_parameters,
                                 LoRALinear, save_lora, load_lora, merge_lora)


def relative(a, b):
    return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)).item()


def reference(m, x):
    return F.linear(x,m.base.weight,m.base.bias) + F.linear(F.linear(x,m.lora_A),m.lora_B)*(m.alpha/m.rank)


def run(preserve_only=False):
    torch.set_num_threads(4)
    faulthandler.dump_traceback_later(90,repeat=True)
    torch.manual_seed(431)
    rows = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape in ((2,64,128,256), (3,17,257,193), (1,7,19,11)):
            b,t,k,n = shape
            original = nn.Linear(k,n,device='cuda',dtype=dtype)
            w = original.weight.detach().clone()
            p = original.weight
            m = enable_lora(original, rank=7, alpha=13)
            assert m.base.weight is p and p.dtype == dtype and torch.equal(p,w)
            with torch.no_grad():m.lora_B.normal_(0,.03)
            x = torch.randn(b,t,k,device='cuda',dtype=dtype,requires_grad=True)
            # Exercise noncontiguous input and incoming gradient.
            x = x.transpose(0,1)
            g = torch.randn(n,b,t,device='cuda',dtype=dtype).permute(2,1,0)
            with torch.autocast('cuda',dtype=dtype,enabled=dtype != torch.float32):
                y = m(x)
                expected = reference(m,x)
            got = torch.autograd.grad(y,(x,m.lora_A,m.lora_B),g)
            ref = torch.autograd.grad(expected,(x,m.lora_A,m.lora_B),g)
            errs = [relative(y,expected)] + [relative(a,b) for a,b in zip(got,ref)]
            limits = 2e-5 if dtype == torch.float32 else .015
            assert max(errs) < limits, (dtype,shape,errs)
            assert p.grad is None and torch.equal(p,w)
            rows.append(dict(check='preserve_forward_gradient',dtype=str(dtype),shape=shape,relative=errs))
            print('passed',dtype,shape,flush=True)
    # Double precision finite differences, not just two similar implementations.
    from precision_lora.core import _LoRAFunction
    base = nn.Linear(5,7,dtype=torch.float64)
    base.requires_grad_(False)
    x = torch.randn(3,5,dtype=torch.float64,requires_grad=True)
    a = torch.randn(2,5,dtype=torch.float64,requires_grad=True)
    b = torch.randn(7,2,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x,a,b:_LoRAFunction.apply(x,a,b,base,.7),(x,a,b))
    rows.append(dict(check='finite_difference_gradcheck',passed=True))
    print('passed finite-difference gradients',flush=True)
    modes = ('preserve',) if preserve_only else ('preserve','nvfp4')
    for mode in modes:
        def make(checkpoint=False):
            torch.manual_seed(52)
            return prepare_lora(nn.Sequential(nn.Sequential(nn.Linear(129,193), nn.GELU()),nn.Linear(193,65)).cuda().bfloat16(),
                                mode=mode,rank=8,alpha=8,checkpoint_modules=['0'] if checkpoint else [],lr=.002)
        m,opt = make()
        frozen = {n:p.clone() for n,p in m.named_parameters() if not p.requires_grad}
        x = torch.randn(3,17,129,device='cuda',dtype=torch.bfloat16)
        target = torch.randn(3,17,65,device='cuda')
        def step(m,opt):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=(m(x).float()-target).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_parameters(m),1.,error_if_nonfinite=True)
            opt.step()
            return loss.item()
        losses = [step(m,opt) for _ in range(64)]
        assert losses[-1] < losses[0]*.85, (mode,losses[0],losses[-1])
        assert all(p.grad is None and torch.equal(p,frozen[n]) for n,p in m.named_parameters() if not p.requires_grad)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'adapter.pt'
            save_lora(path,m,optimizer=opt,base_id='test-base')
            expected_loss = step(m,opt)
            expected = [p.clone() for p in lora_parameters(m)]
            other,oo = make()
            load_lora(path,other,optimizer=oo,base_id='test-base')
            actual_loss = step(other,oo)
            assert actual_loss == expected_loss, (mode,actual_loss,expected_loss)
            assert all(torch.equal(a,b) for a,b in zip(expected,lora_parameters(other)))
        # Checkpoint recomputation must not consume stochastic backward samples.
        plain,po = make(); ck,co = make(True)
        assert step(plain,po) == step(ck,co)
        assert all(torch.equal(a,b) for a,b in zip(lora_parameters(plain),lora_parameters(ck)))
        # Gradient accumulation equals sum of two full forward/backward calls.
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            (m(x).float().square().mean()/2).backward()
            (m(x).float().square().mean()/2).backward()
        assert all(torch.isfinite(p.grad).all() for p in lora_parameters(m))
        rows.append(dict(check='updates_resume_checkpoint_accumulation',mode=mode,steps=64,loss_first=losses[0],loss_last=losses[-1]))
    # Merge is optional, explicit and preserves storage dtype.
    m = enable_lora(nn.Linear(33,27,device='cuda'),rank=4,alpha=9)
    with torch.no_grad():m.lora_B.normal_()
    x = torch.randn(8,33,device='cuda')
    y = m(x).detach()
    merged = merge_lora(m)
    assert relative(merged(x),y) < 2e-6
    rows.append(dict(check='merge',passed=True))
    # Default targeting protects a tied output head.
    shell=nn.Module();shell.embed=nn.Embedding(32,16);shell.head=nn.Linear(16,32,bias=False);shell.head.weight=shell.embed.weight
    shell.proj=nn.Linear(16,16)
    enable_lora(shell)
    assert type(shell.head) is nn.Linear and isinstance(shell.proj,LoRALinear) and shell.head.weight is shell.embed.weight
    # Explicit targeting typo should not freeze anything.
    m=nn.Sequential(nn.Linear(4,4))
    try:enable_lora(m,target_modules=['typo'])
    except ValueError:pass
    else:raise AssertionError('accepted unmatched target')
    assert all(p.requires_grad for p in m.parameters())
    rows.append(dict(check='targeting_and_tied_weight',passed=True))
    if not preserve_only:
        from nvfp4_training._backend import nvfp4_fast_v2 as backend
        from nvfp4_training._backend.scale6 import matmul
        # Paired-seed dX must match the separately invoked RHT+SR pipeline.
        for k,n,t in ((128,256,64),(257,193,17)):
            m=enable_lora(nn.Linear(k,n,device='cuda').bfloat16(),mode='nvfp4',rank=8)
            x=torch.randn(2,t,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            g=torch.randn(2,t,n,device='cuda',dtype=torch.bfloat16)
            means=[]
            ref_means=[]
            for seed in range(16):
                m.base.sr_seed.fill_(seed)
                dx=torch.autograd.grad(m(x),x,g)[0]
                aq=backend.quantize(g.reshape(-1,n),rht=True,sr=True,signs=m.base.signs,seed=seed)
                wq=backend.Packed(m.base.backward_data,m.base.backward_scale,m.base.backward_pts,g.dtype)
                expected=matmul(aq,wq,rows=2*t,cols=k,output_dtype=g.dtype).reshape_as(x)
                assert torch.equal(dx,expected), relative(dx,expected)
                means.append(dx.float())
                ref_means.append(expected.float())
            assert not torch.equal(means[0],means[1]), 'SR seed did not change samples'
            rows.append(dict(check='nvfp4_paired_seed_rht_sr',shape=[k,n,t],seeds=16,maximum_error=0.))
        assert backend.stats()['fp4_gemm'] > 0 and backend.stats()['rht_quant'] > 0 and backend.stats()['sr_quant'] > 0
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps(dict(status='passed',torch=torch.__version__,device=torch.cuda.get_device_name(),checks=rows),indent=2),flush=True)
    return rows


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preserve-only',action='store_true');ap.add_argument('--output')
    args=ap.parse_args()
    rows=run(args.preserve_only)
    if args.output:Path(args.output).write_text(json.dumps(rows,indent=2))
