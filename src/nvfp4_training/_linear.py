import torch
from ._backend.linear6 import Linear6


class AutocastLinear(Linear6):
    """Honor BF16 autocast at the NVFP4 boundary, including FP32 norm/embedding outputs."""
    def forward(self, input):
        if input.dtype != torch.bfloat16:
            if input.dtype not in (torch.float32, torch.float16) or not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16:
                raise TypeError("NVFP4 Linear expects BF16 activations or CUDA BF16 autocast.")
            input = input.to(torch.bfloat16)
        return super().forward(input)
