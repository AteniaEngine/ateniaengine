// Kernel interno: opera sobre punteros DEVICE.
#include <cuda_runtime.h>
#include <cstdio>

extern "C" void* atenia_pool_alloc(size_t bytes);
extern "C" void  atenia_pool_free(void* ptr, size_t bytes);

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

    // SiLU activation: x * sigmoid(x)
    float sig = 1.0f / (1.0f + expf(-acc));
    OUT[row*N + col] = acc * sig;
}

// Host launcher: recibe punteros de HOST y realiza copias host<->device,
// análogo a launch_linear_f32 en linear_cuda.cu.
extern "C" void launch_fused_linear_silu_f32(
    const float* hX,   // [M,K]
    const float* hW,   // [K,N]
    const float* hB,   // [N]
    float* hOUT,       // [M,N]
    int M, int K, int N
){
    const size_t sizeX   = static_cast<size_t>(M) * K * sizeof(float);
    const size_t sizeW   = static_cast<size_t>(K) * N * sizeof(float);
    const size_t sizeB   = static_cast<size_t>(N) * sizeof(float);
    const size_t sizeOUT = static_cast<size_t>(M) * N * sizeof(float);

    void* rawX = atenia_pool_alloc(sizeX);
    void* rawW = atenia_pool_alloc(sizeW);
    void* rawB = atenia_pool_alloc(sizeB);
    void* rawOUT = atenia_pool_alloc(sizeOUT);

    if (!rawX || !rawW || !rawB || !rawOUT) {
        printf("atenia_pool_alloc failed for fused_linear_silu buffers\n");
        if (rawX) atenia_pool_free(rawX, sizeX);
        if (rawW) atenia_pool_free(rawW, sizeW);
        if (rawB) atenia_pool_free(rawB, sizeB);
        if (rawOUT) atenia_pool_free(rawOUT, sizeOUT);
        return;
    }

    float* dX = static_cast<float*>(rawX);
    float* dW = static_cast<float*>(rawW);
    float* dB = static_cast<float*>(rawB);
    float* dOUT = static_cast<float*>(rawOUT);

    cudaError_t err;

    // Copias host -> device
    err = cudaMemcpy(dX, hX, sizeX, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy hX->dX failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(dW, hW, sizeW, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy hW->dW failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy hB->dB failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    fused_linear_silu_f32_kernel<<<grid, block>>>(dX, dW, dB, dOUT, M, K, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("fused_linear_silu_f32_kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(hOUT, dOUT, sizeOUT, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy dOUT->hOUT failed: %s\n", cudaGetErrorString(err));
    }

cleanup:
    atenia_pool_free(dX, sizeX);
    atenia_pool_free(dW, sizeW);
    atenia_pool_free(dB, sizeB);
    atenia_pool_free(dOUT, sizeOUT);
}
