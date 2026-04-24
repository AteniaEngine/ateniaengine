#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__
void batch_matmul_kernel(
    const float* A,
    const float* Bmat,
    int batch_size,
    int M,
    int K,
    int N,
    float* Out
) {
    int b   = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || row >= M || col >= N) return;

    float acc = 0.0f;

    int A_base = b * (M * K);
    int B_base = b * (K * N);

    for (int k = 0; k < K; ++k) {
        acc += A[A_base + row * K + k] * Bmat[B_base + k * N + col];
    }

    Out[b * (M * N) + row * N + col] = acc;
}

// Device-pointer variant: assumes the caller owns the device
// memory and is responsible for freeing it. No pool_alloc / pool_free,
// no H<->D memcpy.
//
// Shape contract:
//   d_a  : [B, M, K]
//   d_b  : [B, K, N]
//   d_out: [B, M, N]
//
// Returns 0 on success, 2 on kernel launch error, 1 on sync error.
extern "C"
int launch_batch_matmul_f32_device_ptrs(
    const float* d_a,
    const float* d_b,
    float* d_out,
    int B,
    int M,
    int K,
    int N
) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16, B);

    batch_matmul_kernel<<<grid, block>>>(d_a, d_b, B, M, K, N, d_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("launch_batch_matmul_f32_device_ptrs launch failed: %s\n", cudaGetErrorString(err));
        return 2;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("launch_batch_matmul_f32_device_ptrs sync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
