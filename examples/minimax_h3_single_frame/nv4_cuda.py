"""H3 adapter for the same shared kernels used by the recorded 200-step run."""
from bootstrap import ROOT
import os
os.environ.setdefault('PRECISION_LORA_KERNEL_CACHE',str(ROOT/'kernel-cache/general'))
from precision_lora._packed_cuda import dequant,quantize,matmul,rope,transpose_rht_quant,backward_scale
