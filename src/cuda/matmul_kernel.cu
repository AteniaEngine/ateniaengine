#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__
void matmul_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int p = 0; p < K; p++) {
        acc += A[row * K + p] * B[p * N + col];
    }

    C[row * N + col] = acc;
}

// Device-pointer launcher: assumes A, B, C already live in device memory.
// The Rust side (`src/cuda/matmul.rs`) owns the alloc / H↔D copy / free
// cycle via `pool_alloc` + `cudaMemcpy`; this launcher only dispatches
// the kernel and synchronizes.
extern "C" __declspec(dllexport)
void matmul_f32_launch_device(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    matmul_f32_kernel<<<grid, block>>>(A, B, C, M, K, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return;
    }
}
