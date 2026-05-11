#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_common.h"

extern "C" {

// ---------------------------------------------
// Simple vector add (for testing CUDA pipeline)
// ---------------------------------------------
__global__
void vec_add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

CUDA_EXPORT
void vec_add_cuda(const float* a, const float* b, float* out, int n) {
    float *d_a, *d_b, *d_out;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;

    vec_add_kernel<<<grid, block>>>(d_a, d_b, d_out, n);

    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

// -------------------------------------------------
// Simple exported wrappers used by the Rust pool FFI
// -------------------------------------------------
CUDA_EXPORT
void cuda_malloc(void** ptr, size_t bytes) {
    cudaMalloc(ptr, bytes);
}

CUDA_EXPORT
void cuda_free(void* ptr) {
    cudaFree(ptr);
}

}
