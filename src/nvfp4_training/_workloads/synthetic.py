import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    """Four Linear layers, two LayerNorms and real scaled-dot-product attention."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(self.norm1(x).to(torch.bfloat16)).reshape(b, t, 3, self.heads, d // self.heads)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        a = a.transpose(1, 2).contiguous().reshape(b, t, d)
        x = x + self.proj(a)
        return x + self.down(F.gelu(self.up(self.norm2(x).to(torch.bfloat16))))


class Network(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.last_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.last_norm(x)
