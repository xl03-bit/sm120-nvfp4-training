# LoRA on an existing NVFP4 checkpoint

`packed_nvfp4` uses the checkpoint's original packed forward weights. Both frozen
base GEMMs—forward and the input gradient—run on native FP4 Tensor Cores. The
input-gradient path applies a 128-value random Hadamard transform and software
stochastic rounding. LoRA parameters and AdamW state are FP32; the low-rank branch
computes in BF16.

## Load explicitly, then use the usual interface

```python
from precision_lora import PackedNVFP4Linear, prepare_lora

# Read these frozen tensors from your checkpoint onto the same CUDA device.
# Match the exact packing convention below; dtype alone does not identify it.
model.proj = PackedNVFP4Linear(
    packed, block_scale, tensor_scale,
    shape=(out_features, in_features),
    layout="comfy-high-first-swizzle128x4",
    backward_mode="recompute",  # or "cache"
    bias=bias,                 # optional frozen bias
    input_scale=awq_scale,      # optional frozen AWQ input multiplier, or None
    seed=123,
)
model, optimizer = prepare_lora(
    model, mode="packed_nvfp4", rank=16, alpha=16,
    target_modules=["proj"], lr=2e-4,
)
```

Use ordinary BF16 autocast, `loss.backward()`, clipping and `optimizer.step()`.
`enable_lora`, `lora_parameters`, `save_lora` and `load_lora` work as in the other
modes. Save adapters separately and retain the exact source checkpoint. Loading
an adapter can switch between `recompute` and `cache` with identical SR state.

The accepted source format is uint8 `[N,K/2]`, with the first value in the high
nibble, E4M3 scales per 16 values in the cuBLAS 128×4 swizzled layout, and one FP32
tensor scale. Logical dimensions may be irregular. The loader adds zero padding,
relocates scale bytes when the swizzle stride changes, and crops outputs/gradients.
It verifies that existing padding represents zero. It does not change active
weight or scale bytes. AWQ input multipliers are applied in both forward and dX.

This is an explicit loader interface, not automatic recognition of every model
repository. A model-specific loader must map its projections, biases and AWQ
metadata correctly. Do not call `model.bfloat16()` after loading quantized buffers;
move by device only and select compute precision through autocast.

## Select packed projections with calibration data

For a model with frozen CUDA BF16 `nn.Linear` weights, supply groups of
projections and a scoring function that reuses your calibration samples. This
example scores a classifier. Replace the patterns and forward call with those
for your own model.

```python
from precision_lora import calibrate_packed_layers

# Each batch contains (inputs, class_indices) on the model's device.
batches = tuple(calibration_batches)

def accuracy(candidate):
    correct = total = 0
    for inputs, labels in batches:
        correct += (candidate(inputs).argmax(-1) == labels).sum().item()
        total += labels.numel()
    return correct / total

report = calibrate_packed_layers(
    model,
    groups={"encoder": ["encoder.blocks.*.mlp"], "decoder": ["decoder.blocks.*.mlp"]},
    keep=["decoder.blocks.*.mlp.output"],
    evaluate=accuracy,
    max_drop=0.005,  # half a percentage point of accuracy
)
```

Group and `keep` patterns match module names or ancestors using the same
case-sensitive glob rules as `nvfp4_training.pack_frozen_layers`. Give each
candidate to a single group. `keep` protects matching projections. Select
standard, untied Linear modules that the model calls in its forward pass.

Calibration first scores the original model, then tries each group in the
supplied order alongside all previously accepted groups. It keeps the packed
modules if the score meets `baseline - max_drop`. A rejected group recovers its
original BF16 modules. A packing or scoring exception restores all replacements
from that call. During calibration both the original and trial weights can be
resident on the GPU. Accepted modules release their original weights when the
call returns and no other references remain.

Use a deterministic, read-only scoring function with higher scores indicating
better quality (e.g. negate a loss). Scores are measured in eval mode with
gradients disabled. The report contains `baseline`, `final_score` and a
`groups` list with the matched candidate paths, trial score and decision.
A fully protected group has an empty `modules` list and a `None` score.
Evaluate the resulting model on held-out samples to measure task accuracy,
throughput and memory. Group order and the calibration data can change which
combination fits the budget.

Accepted projections are ordinary `PackedNVFP4Linear` instances with native
forward and input-gradient GEMMs. To pack one known projection directly, use
`PackedNVFP4Linear.from_linear(frozen_linear)`. Both entry points accept
`backward_mode="recompute"` or `"cache"` and an optional `seed`.

## Gated delta-rule prefill

Models that already implement gated delta-rule linear attention can call
`gated_delta_rule` at their prefill call site on SM120. It works with BF16
projections and with calibrated packed projections. Weight packing and the
attention call are independent opt-ins.

```python
import torch
from precision_lora import gated_delta_rule

# Inside a compatible attention module's dense prefill branch on SM120.
# Keep the model's own training, variable-length and token-decode branches.
rule = self.training_delta_rule if torch.is_grad_enabled() else gated_delta_rule
output, next_state = rule(
    query, key, value, g=log_decay, beta=update_strength,
    initial_state=previous_state, output_final_state=use_cache,
    use_qk_l2norm_in_kernel=True,  # use the model's normalization convention
)
```

Q, K and V have shape `[batch, time, heads, width]` and share an FP32, FP16 or
BF16 dtype. Q and K have the same width. The finite log decay and update
strength use `[batch, time, heads]`. After optional Q/K normalization in their
input dtype, each token decays the state by `exp(g)`, predicts V from K and
that state, then adds the outer product of K with `beta * (V - prediction)`.
The output reads the updated state with Q scaled by `1 / sqrt(key_width)`.
Pass padding and head expansion through the model's usual preprocessing.
Concatenated independent sequences need separate states.

The implementation streams blocks of up to 64 tokens. One CUDA block solves
each triangular system in shared memory. A short tail uses its actual length.
Omitting the initial state lets the first block skip the incoming-state products.
The last block computes an outgoing state only when requested. It returns FP32
state with shape `[batch, heads, key_width, value_width]` and output in the
input dtype. FP32 matrix products follow PyTorch's configured matmul precision.
Use the same setting and chunk size for reference comparisons. Check both
output scores and task accuracy on held-out inputs for your model.

## Memory and speed choices

| Backward mode | Persistent aligned base storage | Backward preparation |
|---|---:|---|
| `recompute` | 0.5625 bytes/weight, plus small scales/signs | Fused packed-weight decode, transpose, H128 RHT and quantization on GPU |
| `cache` | 1.125 bytes/weight, plus small scales/signs | Retains the same backward layout in GPU memory |

Recomputation has no dense weight tensor and no CPU weight transfer. The global
backward scale is calculated once directly from the packed source. A transient
packed matrix is still needed for each GEMM. Padding, activations, adapters,
optimizer state and untargeted parameters add to total memory.

These paths are mathematically distinct from the older `nvfp4` mode: an existing
NVFP4 checkpoint has already lost information during quantization. Its transposed
backward approximation must be derived from that checkpoint. It cannot reproduce
an independently quantized transpose of unavailable original BF16 weights.

## Verified coverage

Tests ran on RTX 5060 Ti 16 GB using Windows / PyTorch 2.14.1+cu132, and the existing
WSL Ubuntu / PyTorch 2.11.0+cu130 environment. No packages in those environments
were changed. CUDA 13 NVRTC compiles the small kernels on first use; subsequent
runs reuse a source-and-runtime-keyed cache. No C++ build toolchain is needed.
`PRECISION_LORA_KERNEL_CACHE` can select the cache directory.

The backend checks for SM120 and uses one CUDA device per process. Other GPU
models, multi-GPU training, double backward and `torch.compile` have not been
validated. Packed-mode CUDA Graph replay is rejected explicitly; eager training
and nonreentrant gradient checkpointing are tested. The older modes retain their
existing graph support.

```bash
python tests/test_packed_lora.py --output packed-checks.json
python tests/test_packed_calibration.py
python tests/test_gated_delta.py
```

Coverage includes 128×384, 193×257, 11×19 and 256×512 projections; batches/sequences
1×1, 2×17 and 3×64; noncontiguous inputs; AWQ scaling; paired-seed SR; bitwise
cache/recompute agreement; 64 MLP optimizer steps; 32 steps through a two-block
attention model; checkpointing and exact optimizer/adapter restoration. The
preserve-mode regression suite also passes. This validates interface mechanics;
task quality needs a real-data run for the chosen model and recipe.

H3-specific prompt formatting, video/audio packing, VAE caches and RoPE selection
remain in the H3 experiment integration. They are not global model patches in
this package.

## Pretrained model examples

The [FLUX.2 Klein recipe](../examples/flux2_klein/README.md) loads official NVFP4
weights and exposes `--backward-mode cache` / `recompute`. Its original scalar
activation calibration is metadata, not an AWQ vector. The experiment explicitly
uses dynamic activation scales during training and inference.

The [experiment report](../results/lora-image/README.md) contains the matched BF16
comparison, 1,000-update subject run, H3 engineering measurements and development
fixes. Only the selected beach subject progression is published as generated
imagery; the numerical traces retain all recorded training steps.
