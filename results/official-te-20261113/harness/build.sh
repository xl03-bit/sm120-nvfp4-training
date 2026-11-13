#!/usr/bin/env bash
set -euo pipefail
work=/mnt/c/Users/18066/Desktop/workspace/te-nvfp4-compare-20261113
root=/home/lixin/te-nvfp4-compare-20261113
mkdir -p "$work/logs" "$work/results"
exec > >(tee -a "$work/logs/build.log") 2>&1
export PATH="$root/venv/bin:/usr/local/cuda/bin:$PATH"
python -m pip install --upgrade pip 'setuptools<82' wheel cmake ninja pybind11 pydantic importlib-metadata nvidia-cudnn-frontend apache-tvm-ffi psutil pytest einops onnxscript onnx nvdlfw-inspect
if [ ! -d "$root/TransformerEngine" ]; then
    cp -a "$work/TransformerEngine" "$root/TransformerEngine"
fi
test "$(git -C "$root/TransformerEngine" rev-parse HEAD)" = 464df759d5a4ed549f2ce99b58fe653178a252f5
export CUDA_HOME=/usr/local/cuda
export CUDNN_PATH="$root/venv/lib/python3.10/site-packages/nvidia/cudnn"
export CPATH="$CUDNN_PATH/include:$root/venv/lib/python3.10/site-packages/nvidia/nccl/include"
export NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=120a
export NVTE_WITH_NCCL_EP=0 MAX_JOBS=3 NVTE_BUILD_MAX_JOBS=3 NVTE_BUILD_THREADS_PER_JOB=1
cd "$root/TransformerEngine"
python -m pip install -v --no-build-isolation --no-deps .
python -m pip freeze > "$work/results/environment-pip-freeze.txt"
python -c 'import torch, transformer_engine, transformer_engine.pytorch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); print(transformer_engine.__version__, transformer_engine.__file__)'
