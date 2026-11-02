"""Train a small model with the installed package and the selected profile."""
import argparse

import torch
from torch import nn
from torch.nn import functional as F
from nvfp4_training import capture_step, check_finite, enable_nvfp4, list_profiles, train_step


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--profile',choices=list_profiles(),default='balanced')
    ap.add_argument('--remat',action='store_true')
    ap.add_argument('--eager',action='store_true')
    ap.add_argument('--steps',type=int,default=8)
    args=ap.parse_args()
    if args.steps<1:ap.error('--steps must be positive')
    torch.manual_seed(1234)
    model=enable_nvfp4(nn.Sequential(nn.Linear(257,512),nn.GELU(),nn.Linear(512,193)),
                       profile=args.profile,remat=args.remat)
    x=torch.randn(2,64,257,device='cuda')
    y=torch.randn(2,64,193,device='cuda')
    def loss_fn(model,x,y):return F.mse_loss(model(x).float(),y)
    if args.eager:
        for _ in range(args.steps):loss=train_step(model,loss_fn,x,y)
    else:
        with capture_step(model,loss_fn,x,y,steps=args.steps) as graph:
            for _ in range(args.steps):loss=graph.step(x,y)
    check_finite(model)
    print(args.profile,'steps',args.steps,'loss',loss.item())


if __name__=='__main__':main()
