# SM120 NVFP4 Training

[Measurement records](results/benchmarks.json) · [Attribution](NOTICE.md)

Train PyTorch Linear layers on native E2M1 Tensor Cores with software stochastic rounding and H16 random Hadamard transforms. The tested machine is a single RTX 5060 Ti 16GB under WSL2 Ubuntu 22.04. Other SM120 cards need their own verification.

## Install and verify

Use an existing Linux or WSL Python with the verified dependencies. The tested stack uses Python 3.10, PyTorch 2.11.0+cu130, TorchAO 0.17.0, Triton 3.6.0, NumPy, Ninja, a compatible NVIDIA driver, CUDA Toolkit 13.0 with nvcc and a C++ compiler. If CUDA discovery fails, set `CUDA_HOME` to the local toolkit directory before starting Python. The install command below adds the package while preserving the installed training dependencies.

```bash
python -m pip install --no-deps .
python -m nvfp4_training list
python -m nvfp4_training verify
python -m nvfp4_training check
python examples/train.py --profile balanced
```

`verify` compiles the CUDA extensions and checks quantization against a separately built reference, paired gradients, stochastic-rounding statistics and continuous updates. It records the accepted backward modes in the user's cache. Re-run after a fingerprint or device change. `check` exercises each profile with aligned/padded shapes, eager/graph execution, transpose recomputation and checkpoint recovery. No dataset is needed for these commands.

## Use in a training loop

```python
import torch
from torch import nn
from nvfp4_training import enable_nvfp4, train_step, check_finite

model = enable_nvfp4(nn.Linear(257, 193), profile="balanced")
x = torch.randn(2, 64, 257, device="cuda")
y = torch.randn(2, 64, 193, device="cuda")

def objective(model, x, y):
    return (model(x).float() - y).square().mean()

loss = train_step(model, objective, x, y)
check_finite(model)
```

Start with a fresh trainable FP32 model with ordinary `nn.Linear` layers on one device. The converter attaches the matching AdamW as `model.nvfp4_optimizer`. For fixed shapes, use `with capture_step(model, objective, x, y, steps=N) as graph:` and call `graph.step(next_x, next_y)` once per real update. An eager step can precede capture. After closing the graph, use `save_checkpoint(path, model)` and restore with `load_checkpoint(path, fresh_converted_model)`. Save the sampler position separately. Global gradient clipping, cross-step accumulation, distributed training and custom Linear subclasses need their own integration. The FP8-master profile requires one deferred weight-gradient contribution per compressed projection per backward.

## Measurements on one RTX 5060 Ti

Complete forward, loss, backward and AdamW update, 12 warmup steps and five rounds of 30 synchronized steps. Peak memory is PyTorch allocated GiB, measured from initialization through training. The table is from the original validated implementation, with transpose recomputation off; all round times and three recomputation variants are in [the results](results/benchmarks.json).

| Profile | Master / moments | 453M, 2×64 ms / GiB | FineWeb 354M, 2×512 ms / GiB |
|---|---|---:|---:|
| `balanced` | FP16 / 8+8 bits | 19.23 / 2.049 | 40.08 / 2.562 |
| `short128` | FP16 / 8+8 bits, 128-element blocks | 18.49 / 2.069 | 39.78 / 2.580 |
| `conservative` | BF16 + signed residual / 8+8 bits | 20.30 / 2.487 | 41.82 / 2.865 |
| `same_as_v5` | FP32 / INT8 + block-scaled FP16 | 28.81 / 3.351 | 46.27 / 3.447 |
| `experimental_m4v4` | FP16 / 4+4 bits | 19.20 / 1.687 | 38.84 / 2.266 |
| `experimental_fp8_master` | scaled E4M3 / 4+4 bits | 19.05 / 1.265 | 39.89 / 1.985 |

`remat=True` re-encodes the transposed weight during backward. In the 453M workload the FP8-master profile with recomputation measured **18.01 ms and 1.031 GiB**. Tied embedding/output weights stay FP32. Weight packing uses separate 1×16 scales along the two contraction directions. NVIDIA Transformer Engine's default 16×16 shared weight scaling is a different recipe.

```bash
python -m nvfp4_training bench --task 453m --profile short128
python -m nvfp4_training bench --task 453m --profile short128 --precision bf16
# For the language-model benchmark, set NVFP4_FINEWEB to the external shard.
python -m nvfp4_training bench --task fineweb --profile experimental_m4v4 --remat
```

The FineWeb benchmark expects [this GPT-2-tokenized shard](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/blob/889765ea1f903759787add96995d81171b632d0c/fineweb_train_000001.bin), SHA-256 `771fa4a99b9fe0946ffb6e848b4ba5c6a9b0fe87860ebf03bc2c1c7e45f8178e`. It verifies the digest before use.

### Validation loss

At 8,192 steps and 8,388,608 training tokens, validation cross-entropy was **5.85269** for balanced NVFP4, **5.87184** for 4-bit moments and **5.89044** for FP8 master + 4-bit moments. Balanced BF16 reached **5.84488**. The FP8-master BF16 control produced nonfinite training loss after its last finite evaluation at step 6,144. The plot and [paired-window results](results/validation.json) retain that missing endpoint. These are single-seed runs on a 354M causal LM.

![Six validation traces and their paired differences](results/validation.png)

### Where the numbers come from

The table and curves come from the saved RTX 5060 Ti runs. The 453M task uses synthetic, GPU-resident inputs, bidirectional attention and MSE. FineWeb uses causal attention, a tied vocabulary head and token cross-entropy. At step 8,192 the balanced FP4–BF16 difference was +0.00781 CE on the same held-out windows. See the [validation and packaging record](results/README.md) for the scope of the checks.

On a new machine, `verify` tests all three backward paths against the reference. It records `fast_pair` for both aligned and padded inputs after the tests pass. This preserves the weight-packing configuration used for the table and long runs.

## License and attribution

The project's original contributions are licensed under the [MIT License](LICENSE). The implementation builds on [Elliot Arledge's SM120 work](https://github.com/Infatoshi/nvfp4-sm120). The upstream MIT license and copyright notice are included. See [NOTICE](NOTICE.md) for component provenance.
