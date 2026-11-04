# Frozen-weight FP4 integration

This update adds a frozen-weight execution path for inference and differentiable objectives. It packs weights once, stores the packed representation and replays fixed-shape computation through CUDA Graphs. The existing six profiles handle trainable base weights.

## Problem identified

A training-oriented linear layer can encode a weight on every forward pass and encode its transpose again during backward. Frozen weights stay constant, so this work can move out of the execution loop. Repeated host dispatch adds overhead, especially when the same backbone runs many times.

Two details matter for backward replay: upstream trainable layers need input gradients through the frozen backbone, and stochastic rounding needs a changing seed on each replay. A captured Python counter would repeat the same seed.

## What this update changes

- `pack_frozen_layers` packs both weight orientations once and releases the converted layers' high-precision master weights. Storage consists of packed weights, block scales and BF16 biases.
- `save_frozen_cache` and `load_frozen_cache` persist and restore the packed buffers directly. A checksum, schema, model fingerprint and logical shapes identify the matching cache.
- `capture_frozen` captures fixed-shape inference or a complete scalar objective and backward pass. A device seed plus a per-operation offset keeps stochastic rounding dynamic on replay.
- Explicit `include` and `keep` patterns select frozen projections and preserve high-precision boundaries. Conversion checks layer type, frozen status and parameter sharing before modifying the model.

## Where to use it

Use it for frozen-backbone inference, adapter training and upstream modules trained through a frozen backbone. The helper targets fixed shapes, first-order gradients and a single SM120 GPU.

| Module | Supported configuration |
|---|---|
| `nn.Linear` | Standard frozen linear projections |
| `nn.Conv2d` | 1×1 kernel, groups=1, stride=1, padding=0 |
| `nn.MultiheadAttention` | Self-attention, dropout=0, masks=None, is_causal=False, need_weights=False |

MHA is converted as one unit; protecting a descendant preserves the whole module. Projections accept BF16/FP32 activations, use native FP4 matrix operations and return BF16. Normalization, softmax, ordinary convolutions and residual state retain their existing precision. Choose precision boundaries using output and gradient fidelity on the target model.

Measure packing, capture and steady replay separately. Total memory includes packed weights, activations, the graph pool and host metadata. Bounded chunking in the objective callback provides a way to trade activation memory against graph size and throughput. Shared weights, custom functional projections, distributed execution and dynamic shapes need application-level integration.

## Use

For a new project, provide the model, the complete objective callback and a precision-boundary list. Reuse the packing, cache and replay implementation, then check numerical behavior and resource use on a small batch.

Install and run the existing backend verification from the repository README first. Then:

```python
from nvfp4_training import pack_frozen_layers, capture_frozen

# Freeze the backbone before conversion.
model.backbone.requires_grad_(False)
model = pack_frozen_layers(
    model.cuda(), include=("backbone",), keep=("input", "output")
)

def objective(model, inputs, targets):
    return loss_function(model(inputs).float(), targets)

with capture_frozen(model, objective, inputs, targets) as replay:
    loss = replay.replay(next_inputs, next_targets)
    # Optional caller-owned optimizer.step() goes here, outside the graph.
```

`capture_frozen` warms up the forward/backward computation and owns gradient clearing during replay. Optimizer updates belong to the caller, outside the graph. Keep gradient buffers allocated, including between optimizer steps; `zero_grad(set_to_none=True)` replaces buffers that the graph uses.

Keep shapes, dtypes, devices, gradient-input flags, module modes and parameter storage fixed while the graph is active. Run capture sequentially and use replay and the optimizer on the original caller stream. Release old autograd graphs before capture. Use `.clone()` to retain an output across replays.

Inputs are copied into fixed, detached buffers, so capture the **complete differentiable objective** for end-to-end gradients. Those gradients belong to the captured buffers and parameters. For inference, pass `backward=False` and return an output tensor instead of a scalar loss.

For a cache, call `save_frozen_cache(path, model, model_fingerprint=revision)` and later `load_frozen_cache(path, fresh_model, expected_fingerprint=revision)`. Use a checkpoint digest or immutable revision identifier. Keep the cache together with its `.sha256` sidecar; restore adapters, other parameters and optimizer state from their own checkpoints.

Run the dataset-free example and tests:

```bash
python examples/frozen.py
python -m unittest discover -s tests -p test_frozen.py -v
```

The generated-tensor tests cover native-path parity, cache round trips, input gradients, repeated frozen-layer use, dynamic rounding, changed-input replay and conversion checks.
