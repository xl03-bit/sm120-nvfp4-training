"""Locate the unchanged H3 recipe's external assets and shared CUDA package."""
from pathlib import Path
import sys,os
CODE_ROOT=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('H3_EXPERIMENT_ROOT',str(CODE_ROOT/'workdir'))).expanduser().resolve()
LIB=CODE_ROOT.parents[1]/'src'
MUSUBI_ROOT=Path(os.environ.get('H3_MUSUBI_ROOT',str(ROOT/'upstream/musubi-tuner'))).expanduser().resolve()
DEPS=Path(os.environ.get('H3_LOCAL_DEPS',str(ROOT/'deps'))).expanduser().resolve()
sys.path[:0]=[str(DEPS),str(MUSUBI_ROOT/'src'),str(LIB)]
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
torch.set_num_threads(4)
