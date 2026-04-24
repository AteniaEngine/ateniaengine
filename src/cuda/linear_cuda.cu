#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__
void linear_f32_kernel(
    const float* __restrict__ A,    // [M,K]
    const float* __restrict__ B,    // [K,N]
    const float* __restrict__ bias, // [N]
    float* __restrict__ C,          // [M,N]
    int M,
    int K,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    // MatMul
    for (int p = 0; p < K; ++p) {
        acc += A[row * K + p] * B[p * N + col];
    }

    // Bias add (bias expected as 1D [N])
    acc += bias[col];

    C[row * N + col] = acc;
}

// Device-pointer variant: assumes the caller already owns the device memory for every
// argument and is responsible for freeing it. No pool_alloc / pool_free,
// no H<->D memcpy. Used by the Rust wire layer when all TensorStorage
// inputs are already `Cuda(TensorGPU)`.
//
// The shape contract is identical to linear_f32_kernel:
//   d_a   : [M, K]
//   d_b   : [K, N]
//   d_bias: [N]
//   d_out : [M, N]
//
// Returns 0 on success, 2 if the kernel launch failed
// (cudaGetLastError after <<<>>>), 1 if cudaDeviceSynchronize failed.
extern "C"
int launch_linear_f32_device_ptrs(
    const float* d_a,
    const float* d_b,
    const float* d_bias,
    float* d_out,
    int M,
    int K,
    int N
) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    linear_f32_kernel<<<grid, block>>>(d_a, d_b, d_bias, d_out, M, K, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("launch_linear_f32_device_ptrs launch failed: %s\n", cudaGetErrorString(err));
        return 2;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("launch_linear_f32_device_ptrs sync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
