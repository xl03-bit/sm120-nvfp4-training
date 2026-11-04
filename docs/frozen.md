# Frozen-weight FP4 integration

This opt-in path is for frozen backbones used in inference or inside a differentiable objective. It complements the existing six full-parameter training profiles. It does not replace their optimizer implementation.

## Problem identified

A training-oriented linear layer can encode a weight on every forward pass and encode its transpose again during backward. Reusing that path with a frozen weight leaves avoidable work in the hot loop. Selecting an FP4 arithmetic kernel also does not automatically enable CUDA Graph replay, reduce host dispatch, or compress every activation and non-linear operator.

Capturing a Python stochastic-rounding counter is another integration hazard: replay can repeat the same rounding seed. Frozen parameters also do not imply that input gradients are unnecessary. Upstream trainable layers may still need the Jacobian through the frozen backbone.

## What this update changes

- `pack_frozen_layers` packs both weight orientations once. Packed bytes, block scales and BF16 biases remain; no high-precision master parameter is retained for converted layers.
- `save_frozen_cache` and `load_frozen_cache` persist those buffers without re-quantizing them. Caches have a checksum, schema, model fingerprint and logical-shape checks. Trainable adapters and optimizer state are excluded.
- `capture_frozen` captures fixed-shape inference or a complete scalar objective and backward pass. A device seed plus a per-operation offset keeps stochastic rounding dynamic on replay.
- Explicit `include` and `keep` patterns define precision boundaries. The API refuses selected trainable weights, unsupported selected convolutions and ambiguous shared-module aliases.
- Synthetic tests check native-path parity, cache round trips, input gradients, dynamic rounding and changed-input replay. They do not establish accuracy or speed for an application.

The update adds no application data, model checkpoints, private workload configuration or application benchmark results.

## Appropriate workloads and limits

Supported frozen modules are standard `nn.Linear`, ungrouped unit-stride unpadded 1×1 `nn.Conv2d`, and restricted `nn.MultiheadAttention` self-attention without dropout, attention/padding masks, causal hints or returned attention weights (pass `need_weights=False`). MHA is converted as one unit; protecting a descendant protects the whole MHA. Ordinary convolutions, normalization, softmax and residual state are not made FP4 by this API. Converted projections accept BF16/FP32 activations and return BF16, with the existing native FP4 matrix path and input-gradient rounding.

Use the existing training profiles for weights that must change. This frozen path does not compute their weight gradients. Application-specific functional projections, tied/shared parameters, distributed execution, second derivatives and dynamic shapes require separate integration. Validate numerical behavior after conversion: BF16 input/output boundaries alone do not guarantee that an internal FP4 block preserves the application's score or gradient direction.

CUDA Graph storage and host metadata can offset weight-storage savings. Measure ordinary execution, graph capture and replay separately, reporting both framework allocation and total device memory. Consider bounded chunking in the application callback when per-example subgraphs make capture too large; validate that the chunked computation preserves the objective. No universal memory reduction or speedup is promised.

## Use

Install and run the existing backend verification from the repository README first. Then:

```python
from nvfp4_training import pack_frozen_layers, capture_frozen

# Freeze only the intended backbone before conversion.
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

`capture_frozen` performs warmup forward/backward passes but no optimizer updates. It clears captured gradient buffers on each replay. Do not call `zero_grad(set_to_none=True)` or replace parameters/buffers while the graph is active. Shapes, dtypes, devices, gradient-input flags and module training modes must remain fixed. Use replay and any external optimizer on the original caller stream; concurrent capture is unsupported. Discard old autograd graphs before capturing. Copy returned outputs with `.clone()` if they must survive the next replay.

Inputs are copied into fixed, detached buffers. Capture the **complete differentiable objective** when end-to-end gradients are required; this helper is not an autograd-aware replacement for a submodule inside an outside graph. For pure inference pass `backward=False` and return an output tensor instead of a scalar loss.

For a cache, call `save_frozen_cache(path, model, model_fingerprint=revision)` and later `load_frozen_cache(path, fresh_model, expected_fingerprint=revision)`. Use a checkpoint digest or an immutable revision identifier, restore other parameters separately, and keep both the cache and its `.sha256` sidecar local unless their publication is intentional.

Run the dataset-free example and tests:

```bash
python examples/frozen.py
python -m unittest discover -s tests -p test_frozen.py -v
```

The tests use only generated tensors. Their results cannot substitute for output/gradient fidelity and resource checks on each new model.
