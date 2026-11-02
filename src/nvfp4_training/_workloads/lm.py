"""GPT-2-style causal LM and reproducible GPT-2-tokenized FineWeb batches."""
import hashlib
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self,d,heads):
        super().__init__()
        self.heads=heads
        self.norm1,self.norm2 = nn.LayerNorm(d),nn.LayerNorm(d)
        self.qkv,self.proj = nn.Linear(d,3*d),nn.Linear(d,d)
        self.up,self.down = nn.Linear(d,4*d),nn.Linear(4*d,d)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v = self.qkv(self.norm1(x).bfloat16()).view(b,t,3,self.heads,d//self.heads).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True,dropout_p=0.)
        x = x+self.proj(a.transpose(1,2).reshape(b,t,d))
        return x+self.down(F.gelu(self.up(self.norm2(x).bfloat16()),approximate='tanh'))

class LanguageModel(nn.Module):
    def __init__(self,seq):
        super().__init__()
        d,n,heads=1024,24,16
        self.tok = nn.Embedding(50257,d)
        self.pos = nn.Embedding(max(512,seq),d)
        self.blocks = nn.Sequential(*(Block(d,heads) for _ in range(n)))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d,50257,bias=False)
        self.head.weight = self.tok.weight
        for p in self.parameters():
            if p.ndim>=2:
                nn.init.normal_(p,std=.02)
        for block in self.blocks:
            for out in (block.proj,block.down):
                nn.init.normal_(out.weight,std=.02/(2*n)**.5)
                nn.init.zeros_(out.bias)
            nn.init.zeros_(block.qkv.bias)
            nn.init.zeros_(block.up.bias)
    def forward(self,idx):
        x = (self.tok(idx)+self.pos(torch.arange(idx.shape[1],device=idx.device))).bfloat16()
        return self.head(self.ln(self.blocks(x)).bfloat16())

class TextBatches:
    def __init__(self,args):
        location = os.environ.get("NVFP4_FINEWEB")
        if not location:
            raise ValueError("Set NVFP4_FINEWEB to the GPT-2 FineWeb 100M-token shard; see README.")
        path = Path(location).expanduser()
        with path.open('rb') as f:
            digest = hashlib.sha256()
            for chunk in iter(lambda:f.read(1<<20),b''):
                digest.update(chunk)
        if digest.hexdigest()!='771fa4a99b9fe0946ffb6e848b4ba5c6a9b0fe87860ebf03bc2c1c7e45f8178e':
            raise RuntimeError('FineWeb shard checksum mismatch')
        data=np.memmap(path,mode='r',offset=1024,dtype='<u2')
        assert len(data)==100_000_000
        boundary=95_000_000+int(np.flatnonzero(data[95_000_000:]==50256)[0])
        self.provenance = dict(sha256=digest.hexdigest(),split_token=boundary,windows_sha256={})
        def make(lo,hi,count,seed):
            rng=np.random.default_rng(seed)
            starts=rng.integers(lo,hi-args.seq,size=(count,args.batch))
            self.provenance['windows_sha256'][str(seed)] = hashlib.sha256(starts.astype('<i8').tobytes()).hexdigest()
            ids=starts[:,:,None]+np.arange(args.seq+1)
            tokens=torch.tensor(data[ids].astype('int64'),device='cuda')
            return [(row[:,:-1].contiguous(),row[:,1:].contiguous()) for row in tokens]
        self.train=make(0,boundary,args.warmup+args.steps*args.rounds,701+args.seed-1234)
        self.validation=make(boundary,len(data),16,702)
