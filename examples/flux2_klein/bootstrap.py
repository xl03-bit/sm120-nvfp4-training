from pathlib import Path
import os,sys
ROOT=Path(os.environ.get('LORA_EXPERIMENT_ROOT',str(Path(__file__).resolve().parent))).resolve()
LIB=Path(__file__).resolve().parents[2]/'src'
ROOT.mkdir(parents=True,exist_ok=True)
for folder in ('cache','metadata','results'): (ROOT/folder).mkdir(exist_ok=True)
sys.path.insert(0,str(LIB))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
(ROOT/'kernel-cache').mkdir(exist_ok=True)
os.environ.setdefault('PRECISION_LORA_KERNEL_CACHE',str(ROOT/'kernel-cache'))
import torch
torch.set_num_threads(4)

def deterministic():
    configure_attention('efficient')

def configure_attention(backend):
    if backend not in ('efficient','cudnn'):raise ValueError(backend)
    # cuDNN uses a different parallel reduction order. It is an explicit option,
    # validated separately, never a silent fallback from deterministic SDPA.
    torch.use_deterministic_algorithms(backend=='efficient')
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(backend=='cudnn')
    torch.backends.cuda.enable_mem_efficient_sdp(backend=='efficient')
    torch.backends.cuda.enable_math_sdp(False)
