# SM120 NVFP4 Training

**Use native NVFP4 to accelerate training and reduce GPU memory use on SM120 GPUs.** This PyTorch package combines FP4 Tensor Core forward/backward computation with compact weight and AdamW storage, plus configurable caching and recomputation for frozen backbones and LoRA.

Built on [Infatoshi/nvfp4-sm120](https://github.com/Infatoshi/nvfp4-sm120), it implements software stochastic rounding (SR) and random Hadamard transforms (RHT). Six full-parameter training profiles let you choose how much memory to spend on weights and AdamW state. Frozen-weight and LoRA paths offer additional storage choices.

Official NVIDIA Transformer Engine also supports RHT, SR and native FP4 on the tested SM120 GPU in the main revision measured on **2026-11-13**. The project focuses on measured training throughput, memory use and model capacity. See the [official-default comparison](#official-transformer-engine-comparison-2026-11-13) for the pinned version, configuration differences, official wins and capacity failures.

## What does this mean?

On the measured **RTX 5060 Ti 16GB**:

- FLUX.2 Klein 4B LoRA at 1024px took **2.306 s/update with 6.626 GiB allocated**, versus **3.405 s and 9.516 GiB** for the matched BF16 run: about **1.48× throughput and 30.4% less allocation**.
- The FLUX recompute option used **4.669 GiB**, **50.9% less allocation than BF16**, at **2.360 s/update**.
- A **4.06B-parameter language model completed 2,048 full-parameter updates** with **12.75 GiB allocated**, using compact weights/moments and transpose recomputation.

These are complete training-update measurements, not isolated GEMM rates. Benefits depend on shape, attention, precision and storage policy. Allocated memory is not total physical VRAM usage, and these single-GPU results do not establish equivalent performance on other cards. The sections below retain loss results and each measurement's scope.

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

For frozen backbones used in inference or differentiable objectives, an opt-in
[frozen-weight path](docs/frozen.md) packs weights once, supports checked disk
caches and captures forward/backward with replay-safe stochastic rounding.
The guide explains the problem, solution, supported workloads and integration steps.
The existing six profiles below remain the path for trainable base weights.

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

## LoRA and unchanged-precision adaptation

`prepare_lora` trains both low-rank matrices while freezing the base. Choose
`mode="preserve"` to retain the model's FP32, BF16 or FP16 weights unchanged, or
`mode="nvfp4"` to prepack frozen projections for native FP4 GEMMs with SR and RHT.
Both use FP32 adapters and ordinary AdamW. Optional block checkpointing reduces
activation memory, and CUDA Graph replay reduces launch overhead for fixed shapes.

For already quantized NVFP4 checkpoints, `mode="packed_nvfp4"` preserves the source
weight payload and offers GPU recomputation or caching of the backward layout.
See [packed NVFP4 LoRA](docs/PACKED_NVFP4.md) for formats, tested compatibility and
memory choices. This new mode uses eager training; the existing modes keep their
CUDA Graph support.

```python
from nvfp4_training import prepare_lora
model, optimizer = prepare_lora(model, mode="preserve", rank=16, alpha=16)
# Use the normal loss.backward() / optimizer.step() loop.
```

The unchanged-precision implementation is also available independently as
`precision_lora`, with only PyTorch as a dependency. See the [LoRA API guide](docs/LORA.md)
for target selection, save/load, merging, independent packaging and examples.
The [LoRA measurements](results/lora/README.md) include three workload shapes and
512 FineWeb updates of a real pretrained GPT-2, with matched data and loss curves.

### Image-model LoRA on a 16 GB GPU

**FLUX.2 Klein Base 4B completed 1,000 NVFP4 LoRA updates at 1024px.** In the matched
40-update benchmark, native NVFP4 reduced full update time from **3.405 to 2.306 s**
and peak allocated memory from **9.516 to 6.626 GiB**. The memory-saving recompute
option took **2.360 s** and **4.669 GiB**—about half the BF16 training allocation.
All modes used the same data, rank, optimizer, checkpointing and attention backend.

**Training reference — one of the four photographs used to learn this toy:**

<img src="results/lora-image/reference/monster-toy-00.jpg" alt="Original training photograph of the red monster toy, showing its raised eyes, yellow hair and belly, and four rope limbs" width="280">

Photo by Varun Jampani, Google DreamBooth, CC BY 4.0.
[Original image, source and license](results/lora-image/reference/README.md).

**Generated beach scene after 100, 200, 300, 500 and 1000 updates:**

![Subject learning after 100, 200, 300, 500 and 1000 updates](results/lora-image/beach-progress.png)

See the [image/H3 experiment report](results/lora-image/README.md) for loss and memory
curves, numerical checks, optimization history and the scope of the H3 tests.
The [H3 single-frame example](examples/minimax_h3_single_frame/README.md) includes
the scripts and recorded configuration of the completed 200-step LoRA experiment.
The [FLUX.2 example](examples/flux2_klein/README.md) includes training, inference,
adapter export and the matched benchmark. These additions leave the six
full-parameter profiles and the existing frozen-weight interfaces unchanged.

## Official Transformer Engine comparison (2026-11-13)

Tested official **TE `2.22.0.dev0+464df759`**, pinned to
[`464df759d5a4ed549f2ce99b58fe653178a252f5`](https://github.com/NVIDIA/TransformerEngine/tree/464df759d5a4ed549f2ce99b58fe653178a252f5),
including SM120 RHT/SR fixes. Actual quantizer/graph checks and native FP4 kernel
evidence confirm RHT, SR and forward/backward FP4 execution on this RTX 5060 Ti.
The earlier statement that official SM120 NVFP4 requires disabling RHT/SR does
not apply to this tested revision.

The primary comparison uses **official default `NVFP4BlockScaling()`**, with
ordinary FP32 parameters/gradients and official FusedAdam FP32 moments for
full-parameter training; LoRA retains BF16 frozen weights and FP32 adapters/AdamW
state. Original model sizes, batches, objectives and timing methods are retained.
No project compressed states, transpose-recomputation implementation or persistent
quantized-weight cache were added to the official default path. This compares
complete implementations with different storage policies; it is not a claim
about TE's best possible tuned configuration or FP4 GEMM alone.

Project values below come from the existing Git archives, without rerunning the
project baseline. The short full-parameter archive predates the packaging fixes.

| Workload / project baseline | Project s/update | Official s/update | Official time change | Project allocated GiB | Official allocated GiB | Official whole-device GiB |
|---|---:|---:|---:|---:|---:|---:|
| 453M, `balanced`, 2×64 | 0.01923 | 0.06153 | +220.0% | 2.049 | 6.896 | 7.846 |
| FineWeb 354M short, `balanced`, 2×512 | 0.04008 | 0.07099 | +77.1% | 2.562 | 5.650 | 6.514 |
| FineWeb 8,192 updates, `balanced`, final segment | 0.04082 | 0.07180 | +75.9% | 2.697 | 5.844 | 7.147 |
| FLUX 40 updates, packed cache | 2.30552 | 2.38321 | +3.37% | 6.626 | 9.732 | 10.779 |
| FLUX 1,000 updates, 970 shared timed indices | 2.37453 | 2.36624 | **−0.35%** | 6.626 | 9.732 | 10.790 |

Official was also **1.9% faster** in the 2×512, 16-layer LoRA CUDA Graph test
without checkpointing; the other eleven LoRA combinations were slower. FLUX's
970-index long-run median was **0.55% slower** for official TE. Those indices
include the project's previously documented preview overlap at updates 568–601.
Excluding that interval from **both** runs gives 936 matched samples and means of
**2.307914 s (project) versus 2.366685 s (official), +2.55%**. Both views are retained;
the small raw-mean win is not evidence of a repeatable throughput advantage.

| Final validation objective | Project | Official default | Official − project |
|---|---:|---:|---:|
| FineWeb 8,192-step CE, `balanced` | 5.852692 | 5.848919 | −0.003773 |
| GPT-2 LoRA 512-step CE, checkpoint off/on | 3.674122 | 3.704230 | +0.030108 |
| FLUX 1,000-step heldout flow MSE | 0.805012 | 0.812247 | +0.007235 |

The completed official runs remained finite. These are single-seed results;
FLUX's final heldout MSE rose above initialization for both implementations and
does not directly score generated-image identity. Image tests also differ in
OS/Python and source weight representation: official TE dynamically quantizes
the corresponding BF16 base, while the project retains its prequantized payload
and H128 backward recipe. The full record lists these differences.

**Capacity:** official 4.06B initialization loaded the complete model, then OOMed
while allocating default FP32 Adam states. The H3 default BF16 projection-storage
probe also OOMed; it was not a completed H3 training run. No smaller model or
custom CPU offload was substituted, and no official speed/loss is reported for
these failures. They show the capacity benefit of the project's measured storage
choices, without claiming that every possible official configuration would fail.

Whole-device memory is sampled with NVML and includes allocator-external memory
and other GPU users. The historical project runs have no corresponding NVML
archive, so the table does not invent a project whole-device value or difference.
LoRA construction/capture and reserved peaks are separately preserved in the
full tables.

See the [experiment record](results/official-te-20261113/README.md),
[all workload/profile tables](results/official-te-20261113/tables.md), and
[machine-readable comparison](results/official-te-20261113/comparison.json).
Earlier runs combining official computation with project compressed storage are
retained as **auxiliary** results, separate from the official-default comparison.

## Measurements on one RTX 5060 Ti

Complete forward, loss, backward and AdamW update, 12 warmup steps and five rounds of 30 synchronized steps. Peak memory is PyTorch allocated GiB, measured from initialization through training. The table uses the original, pre-fix implementation and disables transpose recomputation. Per-round timings and three recomputation variants are in [the results](results/benchmarks.json).

| Profile | Master / moments | 453M, 2×64 ms / GiB | FineWeb 354M, 2×512 ms / GiB |
|---|---|---:|---:|
| `balanced` | FP16 / 8+8 bits | 19.23 / 2.049 | 40.08 / 2.562 |
| `short128` | FP16 / 8+8 bits, 128-element blocks | 18.49 / 2.069 | 39.78 / 2.580 |
| `conservative` | BF16 + signed residual / 8+8 bits | 20.30 / 2.487 | 41.82 / 2.865 |
| `same_as_v5` | FP32 / INT8 + block-scaled FP16 | 28.81 / 3.351 | 46.27 / 3.447 |
| `experimental_m4v4` | FP16 / 4+4 bits | 19.20 / 1.687 | 38.84 / 2.266 |
| `experimental_fp8_master` | scaled E4M3 / 4+4 bits | 19.05 / 1.265 | 39.89 / 1.985 |

Only `experimental_fp8_master` uses adaptive blocks of 512 for master-weight storage. `balanced`, `short128` and `experimental_m4v4` store their compressed projection weights in FP16. `conservative` stores a BF16 value plus a signed residual, and `same_as_v5` keeps FP32 master weights. The block scales for AdamW moments and for the NVFP4 matrix operands are separate from the FP8 master-weight scale.

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

## 4.06B full-parameter run

**On one GeForce RTX 5060 Ti 16GB, a 4.06B-parameter language model completed 2,048 full-parameter AdamW updates on FineWeb.** Training loss averaged **9.60 over the first 32 updates** and **6.78 over the last 32**. On the same 65,536 held-out tokens, validation loss fell from **11.40 at initialization** to **7.03 at update 1,024** and **6.68 at update 2,048**.

![Per-update training loss, 32-update moving average and fixed-window validation loss](results/fineweb-4b-2048/loss.png)

PyTorch peak GPU memory was **12.75 GiB allocated** and **14.33 GiB reserved**. Weights, activations and AdamW state stayed on the card. With two 512-token sequences per batch, the model saw 2,097,152 training tokens at 297.98 ms per update (3,436 tokens/s). The run used `experimental_fp8_master`, block-by-block initialization and transpose recomputation.

![GPU memory allocated and reserved during training, graph capture and validation](results/fineweb-4b-2048/memory.png)

An earlier run exposed an overflow in the fixed-scale FP8 master weights. Version `0.1.0.dev1` lets each block of 512 weights grow its scale when an update exceeds its range. With the same seed and data, the rerun passed all 19 FP8 weight scans and the model and optimizer health check at step 2,048. [Per-update measurements, fix and reproduction instructions](results/fineweb-4b-2048/README.md)

## License and attribution

The project's original contributions are licensed under the [MIT License](LICENSE). The implementation builds on [Elliot Arledge's SM120 work](https://github.com/Infatoshi/nvfp4-sm120). The upstream MIT license and copyright notice are included. See [NOTICE](NOTICE.md) for component provenance.
