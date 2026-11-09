# FLUX.2 Klein Base 4B LoRA

This is the recipe used for the [1,000-update subject experiment](../../results/lora-image/README.md). It loads the official BF16 or NVFP4 base through the same LoRA interface. No model-specific GEMM or transformer-forward patch is used.

Tested: RTX 5060 Ti 16 GB, Windows, Python 3.11, PyTorch 2.14.1+cu132, Diffusers 0.40.0, Transformers 5.16.1, safetensors and Pillow. Keep your working environment; these scripts do not install or upgrade packages. The quantized path needs SM120, CUDA 13 NVRTC and `torch.nn.functional.scaled_mm`. The full-parameter package's older WSL dependency pins are unchanged.

Run from this directory. Optionally set `LORA_EXPERIMENT_ROOT` to a separate data/output directory. Scripts load `precision_lora` from this checkout. Use the **undistilled base**, not the four-step distilled checkpoint.

```bash
hf download black-forest-labs/FLUX.2-klein-base-4B --revision a3b4f4849157f664bdbc776fd7453c2783562f4d --local-dir models/base
hf download black-forest-labs/FLUX.2-klein-base-4b-nvfp4 flux-2-klein-base-4b-nvfp4.safetensors --revision e655535fb8c2c148b47308cfa559a86a73273d55 --local-dir models/nvfp4
python prepare_data.py
python verify_layout.py
python cache_inputs.py --part text
python cache_inputs.py --part vae
python generate.py --mode nvfp4 --attention cudnn --label before --keys short_beach --seeds 7849
python train.py --mode nvfp4 --backward-mode cache --attention cudnn --steps 1000 --save-every 100 --validate-every 100 --label subject-1000
python generate.py --mode nvfp4 --attention cudnn --adapter results/subject-1000/adapter-1000.pt --label after --keys short_beach --seeds 7849
python export_adapter.py results/subject-1000/adapter-1000.pt results/export-1000
```

When using an external experiment directory, place `models/` there too. The files are large; existing snapshots can be reused. The training dataset and model weights are downloaded separately under their upstream terms; one original training photograph is included in the report for visual comparison. `monster_toy` photographs are by Varun Jampani, from [Google DreamBooth](https://github.com/google/dreambooth/tree/4f887af7970a06fc0cd3adaa1d0b368547d6a1d0/dataset/monster_toy). The manifest pins the five source hashes.

Choose `--backward-mode recompute` to save about 1.96 GiB in this workload. For the BF16 control, use `--mode bf16` with the same remaining arguments. Every output label must be new. To continue from step 300, use `--resume path/to/adapter-0300.pt --resume-step 300 --steps 1000`; optimizer step counts are checked. Training caches stay in FP32 after VAE normalization; the VAE and text encoder are frozen and not resident during DiT training.

Both attention choices are explicit. `efficient` enforces deterministic algorithms; `cudnn` is faster on the tested long sequences but not bitwise deterministic. The VAE keeps efficient attention. Do not change attention between modes when comparing throughput or numerical behavior.

For the matched 40-update performance comparison (10 warmups, three groups of 10 measured full updates):

```bash
python benchmark.py --mode preserve --label bf16 --validate
python benchmark.py --mode packed_nvfp4 --backward-mode cache --label fp4-cache --validate
python benchmark.py --mode packed_nvfp4 --backward-mode recompute --label fp4-recompute --validate
```

Timing includes forward, loss, backward, gradient clipping and AdamW. Model loading, compilation, text/VAE caching, validation, image generation and optional profiling are outside step timing. `--profile` records a separate extra update. The source checkpoint's scalar activation calibration is retained as metadata; training uses dynamic per-input scales. It is not interpreted as an AWQ input multiplier.

The public example includes the beach prompt. The recorded training used the same four captions, heldout caption, seed, learning rate and cached training inputs, plus additional evaluation prompts. Adapter export uses Diffusers' safetensors format with rank/alpha metadata; the optimizer/RNG resume checkpoint is separate.
