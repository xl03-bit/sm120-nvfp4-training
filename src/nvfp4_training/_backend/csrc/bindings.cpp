#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> quantize_cuda(torch::Tensor x, torch::Tensor signs,
    int64_t rows, int64_t cols, bool sr, bool rht, int64_t seed,
    c10::optional<torch::Tensor> given_scale, bool pair);
torch::Tensor rescale_cuda(torch::Tensor raw, torch::Tensor sa, torch::Tensor sb,
    int64_t rows, int64_t cols, bool output_float32,
    c10::optional<torch::Tensor> bias);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &quantize_cuda, "Fused SM120 NVFP4 quantization");
    m.def("rescale", &rescale_cuda, "Fused output scale, crop, cast and bias");
}
