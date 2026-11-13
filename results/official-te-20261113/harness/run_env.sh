#!/usr/bin/env bash
set -euo pipefail
work=/mnt/c/Users/18066/Desktop/workspace/te-nvfp4-compare-20261113
root=/home/lixin/te-nvfp4-compare-20261113
export PATH="$root/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$work/project/src:$work${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="$root/torch_extensions"
export TRITON_CACHE_DIR="$root/triton_cache"
export NVFP4_FINEWEB=/mnt/c/Users/18066/Desktop/workspace/GB206-validation-20261031/public-compute-sources/fineweb/fineweb_train_000001.bin
export LORA_GPT2=/mnt/c/Users/18066/Desktop/workspace/GB206-validation-20261031/gpt2
export MAX_JOBS=3 OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="$root/venv/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$work"
exec "$@"
