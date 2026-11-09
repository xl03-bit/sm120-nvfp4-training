"""Build an independent PyTorch-only source package and wheel from shared sources."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'dist');args=ap.parse_args()
staging=args.output.resolve()/'precision-lora-0.2.0.dev1'
staging.mkdir(parents=True,exist_ok=True)
shutil.copytree(ROOT/'src/precision_lora',staging/'src/precision_lora',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copyfile(ROOT/'LICENSE',staging/'LICENSE')
shutil.copyfile(ROOT/'NOTICE.md',staging/'NOTICE.md')
shutil.copytree(ROOT/'THIRD_PARTY_LICENSES',staging/'THIRD_PARTY_LICENSES',dirs_exist_ok=True)
(staging/'README.md').write_text((ROOT/'docs/LORA.md').read_text(encoding='utf8').replace('../results/','results/').replace('../examples/','examples/').replace('(PACKED_NVFP4.md)','(docs/PACKED_NVFP4.md)'),encoding='utf8')
for name in ('tests/test_lora.py','tests/test_lora_graph.py','tests/test_packed_lora.py','examples/lora.py','docs/PACKED_NVFP4.md','docs/LORA.md'):
    dst=staging/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
if (ROOT/'results/lora').exists():
    shutil.copytree(ROOT/'results/lora',staging/'results/lora',dirs_exist_ok=True)
shutil.copytree(ROOT/'results/lora-image',staging/'results/lora-image',dirs_exist_ok=True)
shutil.copytree(ROOT/'examples/flux2_klein',staging/'examples/flux2_klein',dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__','models','data','cache','kernel-cache','metadata','results','benchmark-runs'))
shutil.copytree(ROOT/'examples/minimax_h3_single_frame',staging/'examples/minimax_h3_single_frame',dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('__pycache__','workdir'))
(staging/'pyproject.toml').write_text('''[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
[project]
name = "precision-lora"
version = "0.2.0.dev1"
description = "LoRA with unchanged base weight formats, fused accumulation and CUDA graph replay"
readme = "README.md"
license = {file = "LICENSE"}
requires-python = ">=3.10"
dependencies = ["torch>=2.6"]
[tool.setuptools]
package-dir = {"" = "src"}
license-files = ["LICENSE", "NOTICE.md", "THIRD_PARTY_LICENSES/*.txt"]
[tool.setuptools.packages.find]
where = ["src"]
''',encoding='utf8')
temporary=args.output.resolve()/'.build-tmp';temporary.mkdir(exist_ok=True)
env=dict(os.environ,TEMP=str(temporary),TMP=str(temporary),TMPDIR=str(temporary),PIP_DISABLE_PIP_VERSION_CHECK='1')
subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-index','--no-cache-dir','--no-build-isolation','--wheel-dir',str(args.output.resolve()),str(staging)],check=True,env=env)
archive=Path(str(staging)+'.zip')
# Keep one canonical source copy, excluding setuptools' temporary build copies.
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(staging.rglob('*')):
        rel=p.relative_to(staging)
        if p.is_file() and not any(a in ('build','__pycache__') or a.endswith('.egg-info') for a in rel.parts):
            z.write(p,rel.as_posix())
print(archive)
