"""Verify byte layout against independently published BF16 weights, before inference."""
from bootstrap import ROOT,torch
from klein_loader import BASE,NV4,mapped_names
from precision_lora import _packed_cuda as cuda
from safetensors import safe_open
import json,hashlib

@torch.no_grad()
def main():
    results=[]
    with safe_open(NV4,framework='pt',device='cpu') as quant, safe_open(
            BASE/'transformer/diffusion_pytorch_model.safetensors',framework='pt',device='cpu') as bf16:
        layers=json.loads(quant.metadata()['_quantization_metadata'])['layers']
        for stem in layers:
            q=quant.get_tensor(stem+'.weight').cuda();s=quant.get_tensor(stem+'.weight_scale').cuda()
            g=quant.get_tensor(stem+'.weight_scale_2').cuda()
            w=cuda.dequant(q,s,g)
            ref=torch.cat([bf16.get_tensor(n+'.weight') for n in mapped_names(stem)],0).cuda()
            if w.shape!=ref.shape:raise ValueError('Shape mismatch '+stem)
            # Accumulate in FP32, one matrix at a time; no full dense model on GPU.
            wf=w.float();rf=ref.float()
            cosine=float(torch.nn.functional.cosine_similarity(wf.flatten(),rf.flatten(),dim=0))
            rel=float((wf-rf).norm()/rf.norm())
            record=dict(layer=stem,shape=list(w.shape),cosine=cosine,relative_l2=rel)
            # A generous format detection guard, not a gradient or convergence criterion.
            if cosine<.9 or rel>.4:raise ValueError('Source layout/weight mismatch: '+json.dumps(record))
            results.append(record)
            del w,ref,wf,rf,q,s,g
        # Independent CPU decoding checks scale indexing and nibble convention.
        stem=next(iter(layers));q=quant.get_tensor(stem+'.weight');s=quant.get_tensor(stem+'.weight_scale');g=quant.get_tensor(stem+'.weight_scale_2')
        n,k=q.shape[0],q.shape[1]*2
        rows=torch.tensor([0,1,31,32,63,64,127,128,255,n-1]);cols=torch.arange(128)
        pos=(rows[:,None]//128)*((k+63)//64)*512+(cols[None,:]//16//4)*512+(rows[:,None]%32)*16+((rows[:,None]%128)//32)*4+(cols[None,:]//16)%4
        scale=(s.view(torch.uint8).flatten()[pos].view(torch.float8_e4m3fn).float()*g).bfloat16()
        byte=q[rows[:,None],cols[None,:]//2].int()
        code=torch.where(cols[None,:]%2==0,byte>>4,byte&15)
        table=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6])
        expected=(table[code]*scale.float()).bfloat16()
        actual=cuda.dequant(q.cuda(),s.cuda(),g.cuda())[rows.cuda()[:,None],cols.cuda()[None,:]].cpu()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    report=dict(status='passed',layers=len(results),independent_decode='bitwise exact',
        layout='high nibble first; cuBLAS SWIZZLE_32_4_4 block scales',
        note='Weight reconstruction checks only. Activation quantization and training need separate evaluation.',records=results)
    (ROOT/'metadata/layout-verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(status='passed',layers=len(results),min_cosine=min(r['cosine'] for r in results),
        max_relative_l2=max(r['relative_l2'] for r in results))),flush=True)
if __name__=='__main__':main()
