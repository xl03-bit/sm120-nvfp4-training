#!/usr/bin/env bash
set -euo pipefail
work=/mnt/c/Users/18066/Desktop/workspace/te-nvfp4-compare-20261113
root=/home/lixin/te-nvfp4-compare-20261113
export PATH="$root/image-venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$work/project/src:$work${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="$root/image_torch_extensions" TRITON_CACHE_DIR="$root/image_triton_cache"
export MAX_JOBS=3 OMP_NUM_THREADS=4 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH="$root/image-venv/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$work"
exec "$@"
