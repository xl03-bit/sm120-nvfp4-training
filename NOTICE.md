# Attribution and licensing

The reference CUDA quantizer and parts of the reference Linear and Hadamard implementation in `nvfp4_training._verification` are adapted from [Infatoshi/nvfp4-sm120](https://github.com/Infatoshi/nvfp4-sm120/tree/12e9256073e8eccfaa46c1ae7a6e4773c373328e). The optimized quantizer follows its RNG counter mapping and adjacent-level E2M1 stochastic rounding behavior for comparison. The upstream code is copyright © 2026 Elliot Arledge and is licensed under MIT. Its [license text](THIRD_PARTY_LICENSES/Infatoshi-MIT.txt) is included in the source tree and distribution.

The short-contraction MMA code follows the public [NVIDIA SM120 MMA fragment layouts](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm120.hpp). PyTorch, TorchAO, Triton and the CUDA Toolkit are external dependencies under their respective licenses. This is an independent project.

The project's original contributions are distributed under the [MIT License](LICENSE), copyright © 2026 xl03-bit.

The H3 experiments use [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner/tree/f8a1b03794a49239a3539015075f5123d6c07d66) as the external model implementation and numerical reference for Comfy NVFP4 layout decoding. The Flux integration calls the installed Hugging Face Diffusers checkpoint converter and pipeline. These dependencies, model weights and training datasets retain their own licenses; they are not bundled in the LoRA wheel. Dataset attribution and pinned model revisions are listed in the [image/H3 report](results/lora-image/README.md).
