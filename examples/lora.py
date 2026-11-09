"""A complete LoRA loop, checkpoint and optional fixed-shape capture."""
import argparse
import torch
from torch import nn
from precision_lora import prepare_lora, capture_lora_step, save_lora, lora_parameters

ap=argparse.ArgumentParser()
ap.add_argument('--mode',choices=['preserve','nvfp4'],default='preserve')
ap.add_argument('--graph',action='store_true')
ap.add_argument('--steps',type=int,default=100)
ap.add_argument('--save',default='adapter.pt')
args=ap.parse_args()
torch.manual_seed(1234)
# Replace this model with your pretrained model, already in its desired dtype.
model=nn.Sequential(nn.Linear(257,512),nn.GELU(),nn.Linear(512,193)).to(device='cuda',dtype=torch.bfloat16)
model,optimizer=prepare_lora(model,mode=args.mode,rank=16,alpha=16,lr=2e-4)
x=torch.randn(2,64,257,device='cuda',dtype=torch.bfloat16)
y=torch.randn(2,64,193,device='cuda')
def objective(model,x,y):return (model(x).float()-y).square().mean()
if args.graph:
    with capture_lora_step(model,optimizer,objective,x,y) as graph:
        for _ in range(args.steps):loss=graph.step(x,y)
else:
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=objective(model,x,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_parameters(model),1.,error_if_nonfinite=True)
        optimizer.step()
assert torch.isfinite(loss)
save_lora(args.save,model,optimizer=optimizer,base_id='replace-with-your-model-revision')
print(f'Completed {args.steps} updates; loss={loss.item():.6f}; adapter={args.save}')
