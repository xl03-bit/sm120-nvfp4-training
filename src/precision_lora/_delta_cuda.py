"""CUDA triangular solve used by the inference-time chunked gated delta rule."""

import torch


SOURCE = r'''
extern "C" __global__ void unit_lower_inverse(float* matrices, int width) {
    // Each block owns an entire matrix. Keep all solved rows on chip.
    __shared__ float tile[64][64];
    int col = threadIdx.x;
    float* matrix = matrices + (long long)blockIdx.x * width * width;
    if (col < width) {
        for (int row = 0; row < width; ++row) {
            tile[row][col] = row > col ? matrix[row * width + col] : 0.0f;
        }
    }
    __syncthreads();

    for (int row = 1; row < width; ++row) {
        float next = 0.0f;
        if (col < row) {
            // Match separate FP32 multiply and short strided-sum operations.
            float sums[4] = {};
            for (int k = (col + 1) & ~3; k < row; k += 4) {
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {
                    int index = k + lane;
                    if (index < row) {
                        float product = __fmul_rn(tile[row][index], tile[index][col]);
                        sums[lane] = __fadd_rn(sums[lane], product);
                    }
                }
            }
            float total = __fadd_rn(__fadd_rn(__fadd_rn(sums[0], sums[1]), sums[2]), sums[3]);
            next = __fadd_rn(tile[row][col], total);
        }
        // All columns must finish reading the original row before replacing it.
        __syncthreads();
        if (col < row) tile[row][col] = next;
        __syncthreads();
    }
    if (col < width) {
        for (int row = 0; row < width; ++row) {
            matrix[row * width + col] = __fadd_rn(tile[row][col], row == col ? 1.0f : 0.0f);
        }
    }
}
'''


def unit_lower_inverse_(matrix):
    """Replace the strict lower triangle L with (I - L)^-1, including its unit diagonal."""
    from ._packed_cuda import launch

    if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1] or not 1 <= matrix.shape[-1] <= 64:
        raise ValueError("Expected square matrices of width 1 through 64")
    if not matrix.is_cuda or matrix.dtype != torch.float32 or not matrix.is_contiguous():
        raise ValueError("Expected contiguous CUDA float32 matrices")
    if matrix.requires_grad:
        raise ValueError("This in-place triangular solve requires an inference intermediate")
    width = matrix.shape[-1]
    count = matrix.numel() // (width * width)
    if count:
        with torch.cuda.device(matrix.device):
            launch("unit_lower_inverse", count, 64, matrix, width)
        torch.autograd.graph.increment_version(matrix)
    return matrix
