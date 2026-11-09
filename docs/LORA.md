# LoRA training

There are three base-weight modes. All train A and B, keep adapters and AdamW
state in FP32, and leave biases, norms and embeddings frozen.

| Mode | Frozen base | Calculation | Use when |
|---|---|---|---|
| `preserve` | The original FP32, BF16 or FP16 Parameter objects and values | Original precision, or the caller's autocast precision | Keep the model's weight format and avoid quantization error |
| `nvfp4` | Two prepacked FP4 orientations with independent 1×16 scales | Native FP4 base GEMMs, software SR in backward, H16 RHT in both directions | Reduce base memory and accelerate suitable SM120 workloads |
| `packed_nvfp4` | Existing NVFP4 payload and scales; optional backward cache | Native FP4 base GEMMs; H128 RHT and SR in backward | Train LoRA on an already quantized NVFP4 checkpoint |

For an existing quantized checkpoint, see [packed NVFP4 integration](PACKED_NVFP4.md).
Its `recompute` option keeps about 0.5625 bytes per aligned base weight and builds
the backward layout directly on the GPU. `cache` keeps both orientations. The
existing `nvfp4` recipe is unchanged: its forward and backward weights are
independently quantized from the original floating-point base.

`preserve` also works independently as **precision-lora**, requiring only PyTorch.
It does not import Triton, TorchAO or the NVFP4 CUDA extension. No model-wide dtype
conversion occurs inside these interfaces. `nvfp4` deliberately replaces the
selected base projections with quantized buffers; use `preserve` when the original
weight format must remain intact.

`preserve` does **not** use NVFP4: its frozen base forward and dX are floating-point
operations. Retaining BF16 weights means retaining two bytes per base weight on
the GPU. For a pretrained NVFP4 checkpoint, `packed_nvfp4` instead preserves the
checkpoint's packed representation. It does not restore the original BF16 model.

For measured image-model choices, see [the 4B benchmark](../results/lora-image/README.md):

| Choice | Full update | Training peak allocated | What remains in GPU memory |
|---|---:|---:|---|
| BF16 `preserve` | 3.405 s | 9.516 GiB | Original BF16 base |
| `packed_nvfp4`, `cache` | 2.306 s | 6.626 GiB | Source packed base and backward layout |
| `packed_nvfp4`, `recompute` | 2.360 s | 4.669 GiB | Source packed base; transient backward layout |

These are RTX 5060 Ti results for FLUX.2 Klein Base 4B, 1024px, batch 1, rank 16,
checkpointing and cuDNN attention. They are workload measurements, not universal
mode speedups. The original floating-base `nvfp4` mode has separate
[MLP and GPT-2 measurements](../results/lora/README.md).

## Minimal integration

```python
import torch
from precision_lora import prepare_lora, lora_parameters, save_lora
# The same functions are exported by nvfp4_training.

model, optimizer = prepare_lora(
    model,                     # pretrained model, already on the desired device
    mode="preserve",           # or "nvfp4" with the full NVFP4 package
    rank=16, alpha=16, lr=2e-4,
    target_modules=["q_proj", "v_proj"],  # adapt names to your model
)
optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(**batch).loss
loss.backward()
torch.nn.utils.clip_grad_norm_(lora_parameters(model), 1.0, error_if_nonfinite=True)
optimizer.step()
save_lora("adapter.pt", model, optimizer=optimizer, base_id="model-name@revision")
```

For an existing optimizer/trainer, call `enable_lora(model, ...)`, then construct
the optimizer from `lora_parameters(model)`. Normal `backward()`, gradient
accumulation and AMP/GradScaler work. This API is separate from the full-parameter
`train_step()` / `capture_step()` interface. Set a language model's `use_cache=False`
while training, as usual.

Targets are exact module paths or final path components. Without a target list,
all untied ordinary `nn.Linear` layers are adapted except `head`, `lm_head` and
`output_layer`. Invalid target names fail before conversion. Custom Linear and
quantization wrappers need explicit integration. GPT-2's Conv1D projections must
first be converted with their weights transposed; the real-model test includes
this conversion. These checkpoints are this package's format, not PEFT's format.

The supported adapter is standard LoRA with zero dropout and zero-initialized B.
A and B are both trainable. It does not substitute LoRA-FA or change rank to
produce a speed number. Double backward, distributed wrappers and `torch.compile`
are not covered by these tests.

## Speed and memory

The implementation accumulates the adapter output into the private base output,
and its input gradient into the base input gradient, avoiding two full-sized
intermediate tensors. A frozen FP4 base is encoded once at construction, including
its independently quantized transpose. The original full-size master tensor is
then released for each selected projection.

For aligned matrices, the FP4 base has two copies of 0.5-byte weights and two
1-byte-per-16-value scale arrays: about **1.125 bytes per base weight**, plus two
scalar scales. This is not a claim of 4× lower total training memory. Adapters,
optimizer, untargeted weights, activations and allocator reservations also matter.

Use CUDA Graph replay for repeated fixed shapes. Capture restores the adapters,
optimizer and RNG after warmup; only calls to `step()` count as training updates.
The GPU SR counter advances on every FP4 backward replay.

```python
from precision_lora import capture_lora_step

def objective(model, ids, labels):
    return model(input_ids=ids, labels=labels, use_cache=False).loss

with capture_lora_step(model, optimizer, objective, example_ids, example_labels) as graph:
    for ids, labels in loader:  # tensors on the same CUDA device; fixed shapes
        loss = graph.step(ids, labels)
        # Read loss.item() when needed. The returned tensor is reused next step.
```

Graph replay requires fused torch AdamW and a CUDA model. It captures the current
training mode and optimizer hyperparameters. Close and recapture after changing
shape, learning rate, mode or optimizer state. Save checkpoints after closing the
graph. Do not run simultaneous training against a model captured in a graph.
`packed_nvfp4` uses eager execution and explicitly rejects graph capture; use
ordinary checkpointed training for that mode.

For long sequences, add `checkpoint_modules=["model.layers.0", ...]` to
`prepare_lora`. Use your model's actual parent block paths. Non-reentrant PyTorch
checkpointing recomputes block activations and preserves RNG. This trades extra
forward computation for memory without changing A/B training. The caller may
instead use the model's own checkpointing, but should not wrap the same block twice.

Keeping all original base weights in BF16 cannot make those weights occupy less
than two bytes each. `preserve` saves memory by eliminating base gradients and
optimizer state; checkpointing addresses activations. FP4 base storage is a
separate, explicit choice.

## Save, restore and export

`save_lora` writes only A/B, the recipe, optional optimizer state, RNG and FP4 SR
counters. Keep the exact base model and revision separately. `base_id` checks the
provided revision label, not a hash of a remote model.

```python
from precision_lora import load_lora, merge_lora
model, optimizer = prepare_lora(original_base, mode="preserve", rank=16, alpha=16)
load_lora("adapter.pt", model, optimizer=optimizer, base_id="model-name@revision")
# Optional export, after saving your adapter:
merged = merge_lora(model)
```

`merge_lora` is explicit and available for `preserve` only. It changes base values
by adding BA, rounds into the original dtype, and removes adapters. It uses bounded
row chunks instead of constructing a full FP32 delta weight. Merged inference can
differ slightly from the unmerged two-path calculation due to rounding.

## Install the independent package

With PyTorch already installed, from this repository:

```bash
python tools/package_precision_lora.py
python -m pip install --no-deps dist/precision_lora-0.2.0.dev1-py3-none-any.whl
```

This builds from the same `src/precision_lora` used by the NVFP4 package. It creates
no environment and downloads no dependencies. The independent wheel includes the
NVRTC backend for `packed_nvfp4`. The older `mode="nvfp4"` requires the full package
and its verified WSL/Linux stack. Use `PYTHONPATH=/path/to/repo/src` instead when no
installation is desired. `preserve` still imports no CUDA compiler or quantizer.

## Tests and measurements

```bash
python tests/test_lora.py
python tests/test_lora_graph.py
python tests/test_packed_lora.py --output packed-checks.json
python tests/benchmark_lora.py bench --output results/lora/bench.json
# Uses a local GPT-2 checkpoint and the existing verified FineWeb token shard:
LORA_GPT2=/path/to/gpt2 NVFP4_FINEWEB=/path/to/fineweb_train_000001.bin \
  python tests/benchmark_lora.py fineweb --steps 512 --output results/lora/fineweb.json
```

The test suite covers forward/gradient parity, finite differences, padded shapes,
FP32/FP16/BF16 storage, repeated updates, accumulation, clipping, adapter restoration,
frozen-weight integrity, merging and paired-seed FP4 RHT/SR. Graph/eager updates and
step counts are checked separately. The benchmark compares the same seed, rank,
base, data and AdamW state to an ordinary PyTorch LoRA implementation.

See [measured results](../results/lora/README.md) for timings, memory, loss curves
and the limits of the experiment.

The [FLUX.2 Klein example](../examples/flux2_klein/README.md) provides a complete
pretrained-image-model integration, including the source-format mapping and
Diffusers-compatible adapter export. Model-specific prompt/VAE handling belongs
in that integration; the LoRA and packed CUDA kernels are shared with the H3 tests.
