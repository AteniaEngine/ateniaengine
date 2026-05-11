// M6 — BF16 → F32 in-VRAM upcast kernel.
//
// Pre-existing `matmul_f32_launch_device` and friends consume F32
// device buffers exclusively; today the host code allocates a fresh
// F32 Vec on CPU (via `bf16_decode_bulk`) and uploads it, peaking at
// 2x the BF16 source size in RAM during the upload window. This
// kernel lets the host upload the BF16 raw bytes (half the PCIe
// traffic, no F32 transient on host), then GPU-side promote to F32
// for the F32 kernels to consume.
//
// The conversion is bit-exact with the host AVX2 `bf16_decode_bulk`:
// `f32_bits = (u32)bf16_bits << 16` — every BF16 is the high 16 bits
// of an F32 with zero mantissa LSBs, so the upcast is lossless and
// reversible. CUDA's `__bfloat162float()` intrinsic does exactly
// that operation in hardware on Ampere+ (RTX 4070 is Ada, supports
// it natively).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include "cuda_common.h"

extern "C" __global__
void bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // `__bfloat162float` is a hardware intrinsic on Ampere+ that
        // expands the high 16 bits with zero-padding mantissa, exactly
        // matching the AVX2 host implementation in
        // `src/simd_kernels/avx2.rs::bf16_decode_bulk`.
        dst[i] = __bfloat162float(src[i]);
    }
}

// Host-launchable wrapper. The Rust side owns the alloc / H↔D copy /
// free lifecycle; this launcher only runs the kernel and synchronizes.
// Returns 0 on success, non-zero on launch / sync failure (the F32
// device-pointer launchers in the codebase return void; this one uses
// an i32 status code so the example test harness can distinguish a
// real failure from a no-op).
extern "C" CUDA_EXPORT
int bf16_to_f32_launch_device(
    const void* d_src_bf16,
    float* d_dst_f32,
    int n
) {
    int block = 256;
    int grid = (n + block - 1) / block;

    bf16_to_f32_kernel<<<grid, block>>>(
        (const __nv_bfloat16*)d_src_bf16,
        d_dst_f32,
        n
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "bf16_to_f32 kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "bf16_to_f32 cudaDeviceSynchronize failed: %s\n",
                cudaGetErrorString(err));
        return 2;
    }

    return 0;
}
