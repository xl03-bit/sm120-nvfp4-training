"""Load only this distribution's backends and use PyTorch's CUDA discovery."""
from functools import lru_cache
import os
from pathlib import Path


@lru_cache(maxsize=1)
def load_runtime():
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 (compute capability 12.0) is required.")
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None or not (Path(CUDA_HOME)/"bin/nvcc").is_file():
        raise RuntimeError("Install CUDA Toolkit 13.0 and set CUDA_HOME before starting Python.")
    os.environ.setdefault("MAX_JOBS", "2")
    from ._backend import engine6, nvfp4_fast_v2, step6
    from ._linear import AutocastLinear
    return engine6, step6.StaticStep, nvfp4_fast_v2, AutocastLinear
