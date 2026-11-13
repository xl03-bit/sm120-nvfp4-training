#!/usr/bin/env bash
set -euo pipefail
work=/mnt/c/Users/18066/Desktop/workspace/te-nvfp4-compare-20261113
root=/home/lixin/te-nvfp4-compare-20261113
exec > >(tee "$work/logs/image-torch-only-build.log") 2>&1
export PATH="$root/image-venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export CUDNN_PATH="$root/venv/lib/python3.10/site-packages/nvidia/cudnn"
export CPATH="$CUDNN_PATH/include:$root/image-venv/lib/python3.10/site-packages/nvidia/nccl/include"
export NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=120a NVTE_WITH_NCCL_EP=0
export MAX_JOBS=3 NVTE_BUILD_MAX_JOBS=3 NVTE_BUILD_THREADS_PER_JOB=1
cd "$root/TransformerEngine/transformer_engine/pytorch"
python setup.py build_ext --inplace
python - <<'PY'
import sysconfig,shutil,glob,json,hashlib
from pathlib import Path
source=list(Path('.').glob('transformer_engine_torch*.so'))
assert len(source)==1,source
site=Path(sysconfig.get_paths()['purelib']);old=list((site/'transformer_engine').rglob('transformer_engine_torch*.so'))
assert len(old)==1,old
shutil.copy2(source[0],old[0])
core=list((site/'transformer_engine').rglob('libtransformer_engine.so'))
out=dict(source=str(source[0].resolve()),destination=str(old[0]),extension_sha256=hashlib.sha256(old[0].read_bytes()).hexdigest(),core_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in core})
Path('/mnt/c/Users/18066/Desktop/workspace/te-nvfp4-compare-20261113/results/image-bindings-install.json').write_text(json.dumps(out,indent=2))
print(out)
PY
