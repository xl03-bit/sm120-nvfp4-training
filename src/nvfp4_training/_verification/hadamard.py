import math
import torch

BLK = 16

def hadamard16(device):
    H = torch.ones(1, 1, device=device, dtype=torch.float32)
    while H.shape[0] < BLK:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(BLK)
