// Kernel interno: opera sobre punteros DEVICE.
#include <cuda_runtime.h>
#include <cstdio>

__global__ void fused_linear_silu_f32_kernel(
    const float* __restrict__ X,   // [M,K]
    const float* __restrict__ W,   // [K,N]
    const float* __restrict__ B,   // [N]
    float* __restrict__ OUT,       // [M,N]
    int M, int K, int N
){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    // MatMul row*K + p  *  p*N + col
    for (int p = 0; p < K; p++) {
        acc += X[row*K + p] * W[p*N + col];
    }

    // Add bias
    acc += B[col];

    // SiLU activation: x * sigmoid(x). Clamp the exponent argument to
    // the safe range of expf for float32 (|x| > ~88 overflows to inf
    // or underflows to zero). Without the clamp, very negative acc
    // silently produces sig = 0 and loses all gradient signal; very
    // positive acc saturates sig = 1 without issue but we clamp both
    // sides for symmetry.
    float neg_acc = -acc;
    float clamped = fmaxf(fminf(neg_acc, 88.0f), -88.0f);
    float sig = 1.0f / (1.0f + expf(clamped));
    OUT[row*N + col] = acc * sig;
}

// Device-pointer variant: assumes the caller owns the device
// memory and is responsible for freeing it. No pool_alloc / pool_free,
// no H<->D memcpy.
//
// Shape contract:
//   d_x  : [M, K]
//   d_w  : [K, N]
//   d_b  : [N]
//   d_out: [M, N]
//
// Returns 0 on success, 2 on kernel launch error, 1 on sync error.
extern "C"
int launch_fused_linear_silu_f32_device_ptrs(
    const float* d_x,
    const float* d_w,
    const float* d_b,
    float* d_out,
    int M,
    int K,
    int N
) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    fused_linear_silu_f32_kernel<<<grid, block>>>(d_x, d_w, d_b, d_out, M, K, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("launch_fused_linear_silu_f32_device_ptrs launch failed: %s\n", cudaGetErrorString(err));
        return 2;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("launch_fused_linear_silu_f32_device_ptrs sync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
