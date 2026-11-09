"""Native packed LoRA checks on SM120. No downloaded model is required."""
import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from precision_lora import PackedNVFP4Linear, prepare_lora, lora_parameters, save_lora, load_lora
from precision_lora import _packed_cuda as cuda
from precision_lora.packed import _indices


def source(n, k, seed=91):
    g = torch.Generator(device='cuda').manual_seed(seed)
    full = F.pad(torch.randn(n, k, generator=g, device='cuda', dtype=torch.bfloat16)*.08,
                 (0, -k % 128, 0, -n % 128))
    q, s, p = cuda.quantize(full)
    width = (k+15)//16*16
    # Simulate a checkpoint whose stored rows/columns have only the minimum
    # scale-block padding, not the backend's 128-value alignment.
    small = torch.zeros(((n+127)//128*128)*((width+63)//64)*4, device='cuda', dtype=torch.uint8)
    small[_indices(n,width//16,width,q.device)] = s.view(torch.uint8).flatten()[_indices(n,width//16,full.shape[1],q.device)]
    return q[:n,:width//2].contiguous(), small.view(torch.float8_e4m3fn), p


def make_base(n,k,mode='recompute',seed=91,awq=False):
    return PackedNVFP4Linear(*source(n,k,seed), shape=(n,k), layout='comfy-high-first-swizzle128x4',
                            backward_mode=mode, seed=seed,
                            input_scale=torch.linspace(.7,1.3,k,device='cuda') if awq else None)


def run():
    torch.set_num_threads(4)
    torch.manual_seed(431)
    checks=[]
    # Scalar division must match the original quantizer, not a rounded host
    # reciprocal multiplied on the GPU. Float64 provides an independent check.
    for value in [0.,1.0872942,1.0872943,1.,.001,10000.]:
        amax=torch.tensor(value,dtype=torch.float32,device='cuda');scale=torch.empty_like(amax)
        cuda.launch('finish_scale',1,1,amax,scale)
        expected=(amax.cpu().double()/2688.).clamp_min(1e-30).float()
        torch.testing.assert_close(scale.cpu(),expected,atol=0,rtol=0)
    checks.append(dict(check='backward_scale_correctly_rounded_division',passed=True))
    for n,k in [(128,384),(193,257),(11,19),(256,512)]:
        for awq in [False,True]:
            a=make_base(n,k,awq=awq);b=make_base(n,k,'cache',awq=awq)
            reference=cuda.quantize(cuda.dequant(*a.quant).T.contiguous(),a.signs)
            rebuilt=cuda.transpose_rht_quant(a.quant,a.signs,a.backward_scale)
            for x,y in zip(reference,rebuilt):
                torch.testing.assert_close(x.reshape(-1).view(torch.uint8),y.reshape(-1).view(torch.uint8),rtol=0,atol=0)
            for batch,seq in [(1,1),(2,17),(3,64)]:
                x=torch.randn(batch,seq,k,device='cuda',dtype=torch.bfloat16).transpose(0,1).requires_grad_()
                grad=torch.randn(n,batch,seq,device='cuda',dtype=torch.bfloat16).permute(2,1,0)
                ya,yb=a(x),b(x)
                torch.testing.assert_close(ya,yb,rtol=0,atol=0)
                ga,gb=torch.autograd.grad(ya,x,grad)[0],torch.autograd.grad(yb,x,grad)[0]
                torch.testing.assert_close(ga,gb,rtol=0,atol=0)
                flat=F.pad(grad.reshape(-1,n),(0,a.packed.shape[0]-n)).contiguous()
                seed=int(a.sr_seed.item())-1
                ref=cuda.matmul(cuda.quantize(flat,a.signs,seed=seed,sr=True),reference)[:batch*seq,:k]
                if awq:ref=(ref*a.input_scale).bfloat16()
                torch.testing.assert_close(ga,ref.reshape_as(x),atol=0,rtol=0)
            first=a.sr_seed.clone()
            g1=torch.autograd.grad(a(x),x,grad)[0];g2=torch.autograd.grad(a(x),x,grad)[0]
            assert not torch.equal(g1,g2),'SR must change between backward calls'
            a.sr_seed.copy_(first)
            assert torch.equal(torch.autograd.grad(a(x),x,grad)[0],g1),'SR repeatability'
            checks.append(dict(check='padding_cache_recompute_paired_sr_awq',shape=[n,k],awq=awq,bitwise=True))
            print(checks[-1],flush=True)

    def make(mode='recompute',checkpoint=False):
        torch.manual_seed(157)
        net=nn.Sequential(nn.Sequential(make_base(193,129,mode),nn.GELU()),make_base(65,193,mode,seed=93))
        return prepare_lora(net,mode='packed_nvfp4',rank=8,alpha=8,lr=.002,checkpoint_modules=['0'] if checkpoint else [])
    net,opt=make();cached,co=make('cache');ck,cko=make(checkpoint=True)
    x=torch.randn(3,17,129,device='cuda',dtype=torch.bfloat16)
    target=torch.randn(3,17,65,device='cuda')
    frozen={n:t.clone() for n,t in net.named_buffers() if not n.endswith('sr_seed')}
    def step(m,o):
        o.zero_grad(set_to_none=True)
        loss=(m(x).float()-target).square().mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(m),1.,error_if_nonfinite=True);o.step()
        return loss.item()
    losses=[]
    for i in range(64):
        v=step(net,opt);losses.append(v)
        if i<3:
            assert v==step(cached,co)==step(ck,cko)
            for p,q,r in zip(lora_parameters(net),lora_parameters(cached),lora_parameters(ck)):
                assert torch.equal(p,q) and torch.equal(p,r)
    assert losses[-1]<losses[0]*.85,losses
    for n,t in net.named_buffers():
        if n in frozen:assert torch.equal(t.reshape(-1).view(torch.uint8),frozen[n].reshape(-1).view(torch.uint8)),n
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'adapter.pt';save_lora(path,net,optimizer=opt,base_id='synthetic@91')
        expected=step(net,opt);restored,ro=make('cache')
        load_lora(path,restored,optimizer=ro,base_id='synthetic@91')
        assert step(restored,ro)==expected
        assert all(torch.equal(p,q) for p,q in zip(lora_parameters(net),lora_parameters(restored)))
    checks.append(dict(check='mlp_updates_resume_checkpoint',steps=64,loss_start=losses[0],loss_end=losses[-1],base_bytes_unchanged=True))
    print(checks[-1],flush=True)
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv=make_base(384,128)
            self.proj=make_base(128,128,seed=17)
            self.fc1=make_base(193,128,seed=18)
            self.fc2=make_base(128,193,seed=19)
        def forward(self,x):
            q,k,v=self.qkv(F.layer_norm(x,(128,))).chunk(3,dim=-1)
            def heads(t):return t.reshape(*t.shape[:-1],4,32).transpose(1,2)
            z=F.scaled_dot_product_attention(heads(q),heads(k),heads(v)).transpose(1,2).reshape_as(x)
            y=x+self.proj(z)
            return y+self.fc2(F.gelu(self.fc1(F.layer_norm(y,(128,)))))
    transformer,to=prepare_lora(nn.Sequential(Attention(),Attention()),mode='packed_nvfp4',rank=8,lr=.001,checkpoint_modules=['0','1'])
    tx=torch.randn(2,37,128,device='cuda',dtype=torch.bfloat16)
    ty=torch.randn_like(tx).float()
    tl=[]
    for i in range(32):
        to.zero_grad(set_to_none=True);loss=(transformer(tx).float()-ty).square().mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(transformer),1.,error_if_nonfinite=True);to.step();tl.append(loss.item())
    assert tl[-1]<tl[0],tl
    checks.append(dict(check='two_block_attention_checkpoint_updates',steps=32,batch=2,sequence=37,loss_start=tl[0],loss_end=tl[-1]))
    print(checks[-1],flush=True)
    invalid=make_base(11,19).bfloat16()
    try:invalid(torch.randn(2,19,device='cuda',dtype=torch.bfloat16))
    except TypeError:pass
    else:raise AssertionError('Accepted corrupted scale dtypes')
    autocast_base=make_base(11,19)
    float_input=torch.randn(2,19,device='cuda',dtype=torch.float32,requires_grad=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):auto=autocast_base(float_input)
    assert auto.dtype==torch.bfloat16
    auto.sum().backward();assert float_input.grad.dtype==torch.float32 and torch.isfinite(float_input.grad).all()
    return dict(status='passed',torch=torch.__version__,gpu=torch.cuda.get_device_name(),checks=checks)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    report=run();Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
