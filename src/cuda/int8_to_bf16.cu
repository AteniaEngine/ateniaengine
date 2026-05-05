// M9.0 — INT8 → BF16 per-channel dequant kernel.
//
// Naive baseline used by `examples/bench_int8_w8a16.rs` to gate
// whether INT8 Tensor Cores on RTX 4070 (Ada, sm_89) deliver a real
// speedup vs the M8.4c BF16-resident path. Each thread handles one
// element of a row-major weight matrix [K, N] and computes:
//
//     w_bf16[k, n] = (float)w_int8[k, n] * scales[n]   (truncated to BF16)
//
// `scales` is a per-output-channel (per-column) F32 vector of length
// N — the standard absmax-symmetric quantisation layout for Llama-
// family weight-only INT8. The truncation drops the lower 16 mantissa
// bits, matching the M4.7.2.e BF16 storage path the rest of the
// engine uses.
//
// This is a *separated* dequant — full materialisation of the BF16
// weight in VRAM before the matmul. M9.0-B's fused dequant+GEMM path
// would skip the materialisation; M9.0-A measures the worst case.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void int8_to_bf16_per_channel_kernel(
    const int8_t*  __restrict__ d_int8,
    const float*   __restrict__ d_scales,
    uint16_t*      __restrict__ d_bf16,
    int K,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * N;
    if (idx >= total) return;

    int col = idx % N;                              // [K, N] row-major
    int8_t  q  = d_int8[idx];
    float   f  = (float)q * d_scales[col];
    uint32_t bits = __float_as_uint(f);
    d_bf16[idx] = (uint16_t)(bits >> 16);            // BF16 truncation
}

extern "C" int int8_to_bf16_per_channel_launch_device(
    const void*   d_int8,
    const float*  d_scales,
    void*         d_bf16,
    int           k,
    int           n
) {
    long long total = (long long)k * (long long)n;
    if (total <= 0) return 0;

    int block = 256;
    long long grid_ll = (total + (long long)block - 1) / (long long)block;
    int grid = (grid_ll > 65535 * 1024) ? (65535 * 1024) : (int)grid_ll;
    // For the Llama 13B shapes (max K*N = 5120*32000 ≈ 164M), grid is
    // ~640k blocks of 256 threads — well within sm_89's launch limits.
    if (grid_ll > (long long)grid) {
        // Shouldn't happen for our shapes, but fail loud if it ever does.
        return -1;
    }

    int8_to_bf16_per_channel_kernel<<<grid, block>>>(
        (const int8_t*)d_int8,
        d_scales,
        (uint16_t*)d_bf16,
        k,
        n
    );
    cudaError_t err = cudaGetLastError();
    return (int)err;
}
