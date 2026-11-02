import torch
from torch import nn
from torch.nn import functional as F

from .linear import FP4LinearFn
from .hadamard import hadamard16


def _round128(n):
    return (n + 127) // 128 * 128


class NVFP4Linear(nn.Linear):
    """FP4 forward/dgrad/wgrad with SR/RHT in backward; 128-pad transparent."""

    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=torch.float32)
        dev = self.weight.device
        self.register_buffer("H", hadamard16(dev))
        rng = torch.Generator(device="cpu").manual_seed(1)
        signs = torch.randint(0, 2, (16,), generator=rng).float() * 2 - 1
        self.register_buffer("signs", signs.to(dev))

    def forward(self, x):
        if x.device.type != "cuda":
            raise RuntimeError("NVFP4Linear requires CUDA; refusing BF16/CPU fallback")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input features {self.in_features}, got {x.shape[-1]}")
        m = x.numel() // self.in_features
        k = self.in_features
        n = self.out_features
        # Pad each GEMM dimension to 128; autograd maps the padding gradients
        # back into the original input and FP32 parameter tensors.
        px = F.pad(x.reshape(m, k), (0, _round128(k)-k, 0, _round128(m)-m))
        pw = F.pad(self.weight, (0, _round128(k)-k, 0, _round128(n)-n))
        y = FP4LinearFn.apply(px.contiguous(), pw.contiguous(),
                                         self.H, self.signs, None, None)
        y = y[:m, :n]
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype)
        return y.reshape(*x.shape[:-1], n)


def enable_nvfp4(model):
    """Convert nn.Linear modules recursively. Call before creating optimizer.

    Does not transform arbitrary matmul/F.linear/conv operations, tied weights,
    or already compiled models. No silent fallback for replaced Linear layers.
    """
    if isinstance(model, NVFP4Linear):
        return model
    if isinstance(model, nn.Linear):
        converted = NVFP4Linear(model.in_features, model.out_features,
                                bias=(model.bias is not None),
                                device=model.weight.device)
        with torch.no_grad():
            converted.weight.copy_(model.weight.to(dtype=torch.float32))
            if model.bias is not None:
                converted.bias.copy_(model.bias.to(dtype=torch.float32))
        converted.weight.requires_grad_(model.weight.requires_grad)
        if model.bias is not None:
            converted.bias.requires_grad_(model.bias.requires_grad)
        converted.train(model.training)
        return converted
    for name, child in list(model.named_children()):
        replaced = enable_nvfp4(child)
        if replaced is not child:
            setattr(model, name, replaced)
    return model
