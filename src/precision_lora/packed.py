"""Explicit, frozen Comfy-layout NVFP4 Linear with native input gradients.

This loader preserves source nibbles and scales. It does not infer layouts from
filenames or quantize floating-point pretrained weights without being asked.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _round128(n):
    return (n + 127) // 128 * 128


def _indices(rows, groups, width, device):
    r = torch.arange(rows, device=device)[:, None]
    b = torch.arange(groups, device=device)[None, :]
    return (r // 128) * ((width + 63) // 64) * 512 + (b // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + b % 4


class _BaseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, base):
        ctx.base, ctx.shape, ctx.input_dtype = base, x.shape, x.dtype
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        return base.multiply(x.reshape(-1, x.shape[-1]).to(dtype)).reshape(*x.shape[:-1], base.out_features)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        return ctx.base.multiply(grad.reshape(-1, grad.shape[-1]), backward=True).reshape(ctx.shape).to(ctx.input_dtype), None


class PackedNVFP4Linear(nn.Module):
    """Native frozen Linear for high-nibble-first E2M1, cuBLAS-swizzled E4M3 scales.

    ``shape`` is the logical (out_features, in_features), before zero padding.
    ``backward_mode='recompute'`` retains one packed orientation and builds its
    RHT-transposed layout on the GPU per backward. ``'cache'`` keeps both on GPU.
    Both modes use identical values and SR sequences. Bias and optional AWQ
    ``input_scale`` stay frozen; input scaling is included in the input gradient.
    """
    def __init__(self, packed, block_scale, tensor_scale, *, shape=None, bias=None,
                 input_scale=None, layout, backward_mode="recompute", seed=0):
        super().__init__()
        if layout != "comfy-high-first-swizzle128x4":
            raise ValueError("Declare layout='comfy-high-first-swizzle128x4'; other layouts need conversion")
        if backward_mode not in ("recompute", "cache"):
            raise ValueError("backward_mode must be 'recompute' or 'cache'")
        if packed.ndim != 2 or packed.dtype != torch.uint8 or not packed.is_cuda:
            raise TypeError("packed must be a CUDA uint8 [rows, cols/2] tensor")
        if block_scale.dtype != torch.float8_e4m3fn or tensor_scale.dtype != torch.float32 or tensor_scale.numel() != 1:
            raise TypeError("Expected E4M3 block scales and one FP32 tensor scale")
        tensors = [packed, block_scale, tensor_scale] + [t for t in (bias, input_scale) if t is not None]
        if any(t.device != packed.device or t.requires_grad for t in tensors):
            raise ValueError("Base tensors must be frozen and on the same CUDA device")
        rows, width = packed.shape[0], packed.shape[1] * 2
        if rows < 1 or width % 16 or width < 16:
            raise ValueError("Stored width must contain complete 16-value scale blocks")
        n, k = (rows, width) if shape is None else tuple(shape)
        if type(n) is not int or type(k) is not int or not (0 < n <= rows and 0 < k <= width):
            raise ValueError("Invalid logical shape")
        if block_scale.numel() != _round128(rows) * ((width + 63) // 64) * 4:
            raise ValueError("Swizzled scale storage size does not match the packed shape")
        if not torch.isfinite(tensor_scale).all() or tensor_scale.item() <= 0:
            raise ValueError("Tensor scale must be finite and positive")
        if bias is not None and (bias.shape != (n,) or not bias.is_floating_point()):
            raise ValueError("Bias must have logical output width")
        if input_scale is not None and (input_scale.shape != (k,) or not torch.isfinite(input_scale).all()):
            raise ValueError("AWQ input_scale must be a finite vector of logical input width")
        # Padding must represent zero. Nonzero hidden columns/rows would change
        # the quantized backward operator when mixed by the RHT.
        if n < rows and (packed[n:].bitwise_and(0x77) != 0).any():
            raise ValueError("Nonzero row padding in source weights")
        if k < width:
            start = (k + 1) // 2
            if (packed[:n, start:].bitwise_and(0x77) != 0).any() or (k % 2 and (packed[:n, k // 2].bitwise_and(7) != 0).any()):
                raise ValueError("Nonzero column padding in source weights")
        self.in_features, self.out_features = k, n
        self.backward_mode, self.layout = backward_mode, layout
        self.source_dtype = torch.bfloat16
        nr, kr = _round128(rows), _round128(width)
        q = F.pad(packed, (0, (kr-width)//2, 0, nr-rows)).contiguous() if (nr, kr) != (rows, width) else packed.contiguous()
        old = block_scale.contiguous().view(torch.uint8).flatten()
        source_indices = _indices(rows, width//16, width, q.device)
        active = old[source_indices]
        if not torch.isfinite(active.view(torch.float8_e4m3fn).float()).all() or (active.bitwise_and(128) != 0).any():
            raise ValueError("Block scales must be finite and nonnegative")
        if (nr, kr) == (rows, width):
            scales = block_scale.contiguous()
        else:
            raw = torch.zeros(nr*kr//16, dtype=torch.uint8, device=q.device)
            raw[_indices(rows, width//16, kr, q.device)] = active
            scales = raw.view(torch.float8_e4m3fn)
        self.register_buffer("packed", q)
        self.register_buffer("block_scale", scales)
        self.register_buffer("tensor_scale", tensor_scale.reshape(()))
        self.register_buffer("bias", bias)
        self.register_buffer("input_scale", input_scale)
        self.rht_seed = int(seed) % (2**63)
        generator = torch.Generator().manual_seed(self.rht_seed)
        self.register_buffer("signs", (torch.randint(0, 2, (128,), generator=generator)*2-1).float().to(q.device))
        self.register_buffer("sr_seed", torch.tensor(int(seed) & 0x7fffffff, device=q.device, dtype=torch.int32))
        from . import _packed_cuda as cuda
        with torch.cuda.device(q.device):
            self.register_buffer("backward_scale", cuda.backward_scale(self.quant, self.signs))
            bq = cuda.transpose_rht_quant(self.quant, self.signs, self.backward_scale) if backward_mode == "cache" else (None, None, None)
        self.register_buffer("backward_packed", bq[0])
        self.register_buffer("backward_blocks", bq[1])

    @property
    def quant(self):
        return self.packed, self.block_scale, self.tensor_scale

    @property
    def device(self):
        return self.packed.device

    def multiply(self, x, backward=False):
        from . import _packed_cuda as cuda
        if self.packed.dtype != torch.uint8 or self.block_scale.dtype != torch.float8_e4m3fn or any(
            t.dtype != torch.float32 for t in (self.tensor_scale, self.backward_scale, self.signs)) or self.sr_seed.dtype != torch.int32:
            raise TypeError("Packed weight/scale dtypes changed; move the base by device only and use BF16 autocast")
        if x.dtype != torch.bfloat16 or x.device != self.device or x.ndim != 2:
            raise TypeError("Packed NVFP4 GEMMs require BF16 inputs on the base CUDA device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Packed NVFP4 CUDA graph replay has not been validated; use ordinary eager training")
        logical = self.out_features if backward else self.in_features
        if x.shape[-1] != logical:
            raise ValueError("Input width does not match the frozen projection")
        if not backward and self.input_scale is not None:
            x = (x * self.input_scale).to(x.dtype)
        width = self.packed.shape[0] if backward else self.packed.shape[1]*2
        v = F.pad(x, (0, width-logical)).contiguous() if width != logical else x.contiguous()
        with torch.cuda.device(x.device):
            if backward:
                bq = (self.backward_packed, self.backward_blocks, self.backward_scale) if self.backward_mode == "cache" else cuda.transpose_rht_quant(self.quant, self.signs, self.backward_scale)
                aq = cuda.quantize(v, self.signs, seed=self.sr_seed, sr=True)
                self.sr_seed.add_(1)
                y = cuda.matmul(aq, bq)[:len(x), :self.in_features]
                if self.input_scale is not None:
                    y = (y * self.input_scale).to(x.dtype)
            else:
                y = cuda.matmul(cuda.quantize(v), self.quant)[:len(x), :self.out_features]
                if self.bias is not None:
                    y = y + self.bias.to(y.dtype)
        return y

    def forward(self, x):
        return _BaseFunction.apply(x, self)
