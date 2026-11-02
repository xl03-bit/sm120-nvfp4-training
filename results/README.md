# Measurement provenance

All measurements here were recorded on one RTX 5060 Ti 16GB under WSL2 Ubuntu 22.04, with PyTorch 2.11.0+cu130, CUDA 13.0, TorchAO 0.17.0 and Triton 3.6.0.

| Record | Origin and coverage |
|---|---|
| [4.06B FineWeb run](fineweb-4b-2048/README.md) | 2,048 full-parameter updates. Per-update losses and VRAM, 19 fixed-window evaluations, and a diagnosis of the FP8 overflow found in the initial run. |
| [capacity-4b.json](capacity-4b.json) | Single-GPU capacity run with 4,063,797,760 trainable parameters. The 50-layer model used FP8 master weights for eligible projections, two 4-bit AdamW moments and transpose recomputation. The tied embedding/output weight stayed FP32. The log includes all five training losses, one held-out evaluation and peak allocated and reserved memory. |
| [benchmarks.json](benchmarks.json) | Eighteen full-step runs through the original local training API. Six profiles on both workloads and three additional transpose-recomputation runs per workload. Each record includes all five timed rounds. |
| [validation.json](validation.json) and [plot](validation.png) | Saved 354M FineWeb runs. Seed 1234, 2×512 tokens per step, 64 fixed validation batches. The FP8-master BF16 control ends at its last finite validation. Confidence intervals use paired validation windows within this seed. |
| [profile-checks.json](profile-checks.json) | The original installed API checks for all six profiles, both shape classes and both transpose settings. Each row includes FP4 operation counts and exact checkpoint continuation. |

## Package verification

A preliminary wheel built from the extracted sources compiled on this GPU from empty extension caches. It passed all 24 explicit-seed quantizer comparisons, the multi-seed gradient and convergence checks, and 24 profile/checkpoint cases. Its verifier selected `fast_reuse` on the small optimizer test. That preliminary 453M `short128` run measured 33.99 ms and 1.849 GiB, compared with the archived 18.49 ms and 2.069 GiB for `fast_pair`.

The packaged implementation retains the archived `fast_pair` setting after reference validation. The final packaging change was reviewed using the local records. The original six-profile timing table and the 354M curves report the archived measurements. The 4.06B experiment uses the packaged implementation with the FP8 scale repair in version 0.1.0.dev1.
