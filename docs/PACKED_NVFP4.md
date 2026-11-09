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
