"""Check that official SR changes payloads on CUDA Graph replay at fixed input."""
import argparse,json,traceback
from pathlib import Path
import torch
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

p=argparse.ArgumentParser();p.add_argument('--output',default='results/official-graph-rng.json');args=p.parse_args()
torch.manual_seed(1234)
x=torch.randn(512,512,device='cuda',dtype=torch.bfloat16)
rows=[]
for rht,sr in [(True,True),(True,False),(False,True),(False,False)]:
    row=dict(rht=rht,sr=sr)
    try:
        q=NVFP4Quantizer(rowwise=True,columnwise=True,with_rht=rht,with_post_rht_amax=rht,stochastic_rounding=sr)
        for _ in range(3):y=q(x)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):y=q(x)
        data=[]
        for _ in range(4):
            graph.replay()
            data.append({k:getattr(y,k).clone() for k in ('_rowwise_data','_columnwise_data') if getattr(y,k,None) is not None})
        torch.cuda.synchronize()
        row['successive_payload_changed_bytes']={k:[int((data[i][k]!=data[i+1][k]).sum()) for i in range(3)] for k in data[0]}
        changed=any(n>0 for counts in row['successive_payload_changed_bytes'].values() for n in counts)
        assert changed==sr, f'Expected changing payload iff SR enabled, got changed={changed}'
        row['status']='passed'
    except Exception:
        row.update(status='failed',traceback=traceback.format_exc())
    rows.append(row)
    print(json.dumps(row),flush=True)
Path(args.output).write_text(json.dumps(rows,indent=2))
