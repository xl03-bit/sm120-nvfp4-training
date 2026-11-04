"""Synthetic frozen-layer example. No application data or checkpoints required."""
import torch
from torch import nn
from torch.nn import functional as F
from nvfp4_training import pack_frozen_layers, capture_frozen

class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(256,256)
        self.backbone=nn.Linear(256,256).requires_grad_(False)
        self.output=nn.Linear(256,8)
    def forward(self,x):
        x=self.input(x)
        return self.output(F.silu(self.backbone(x)))

def main():
    torch.manual_seed(7)
    model=Example().cuda()
    model=pack_frozen_layers(model,include=('backbone',),keep=('input','output'))
    x=torch.randn(8,256,device='cuda')
    target=torch.randn(8,8,device='cuda')
    def objective(model,x,target):return F.mse_loss(model(x).float(),target)
    with capture_frozen(model,objective,x,target) as replay:
        for _ in range(3):
            loss=replay.replay(x,target)
            print('synthetic loss:',loss.item())
    print('No optimizer updates were performed.')

if __name__=='__main__':main()
