#include <cuda_runtime.h>
#include <cstdio>

extern "C" void* atenia_pool_alloc(size_t bytes);
extern "C" void  atenia_pool_free(void* ptr, size_t bytes);

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

// Host launcher that recibe punteros de HOST y hace el ciclo completo host<->device.
extern "C"
void launch_linear_f32(
    const float* hA,
    const float* hB,
    const float* hBias,
    float* hC,
    int M,
    int K,
    int N
) {
    const size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
    const size_t sizeB = static_cast<size_t>(K) * N * sizeof(float);
    const size_t sizeBias = static_cast<size_t>(N) * sizeof(float);
    const size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

    void* rawA = atenia_pool_alloc(sizeA);
    void* rawB = atenia_pool_alloc(sizeB);
    void* rawBias = atenia_pool_alloc(sizeBias);
    void* rawC = atenia_pool_alloc(sizeC);

    if (!rawA || !rawB || !rawBias || !rawC) {
        printf("atenia_pool_alloc failed for linear buffers\n");
        if (rawA) atenia_pool_free(rawA, sizeA);
        if (rawB) atenia_pool_free(rawB, sizeB);
        if (rawBias) atenia_pool_free(rawBias, sizeBias);
        if (rawC) atenia_pool_free(rawC, sizeC);
        return;
    }

    float* dA = static_cast<float*>(rawA);
    float* dB = static_cast<float*>(rawB);
    float* dBias = static_cast<float*>(rawBias);
    float* dC = static_cast<float*>(rawC);

    cudaError_t err;

    // Copias host -> device
    err = cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy hA->dA failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy hB->dB failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(dBias, hBias, sizeBias, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy hBias->dBias failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    linear_f32_kernel<<<grid, block>>>(dA, dB, dBias, dC, M, K, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("linear_f32_kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy dC->hC failed: %s\n", cudaGetErrorString(err));
    }

cleanup:
    atenia_pool_free(dA, sizeA);
    atenia_pool_free(dB, sizeB);
    atenia_pool_free(dBias, sizeBias);
    atenia_pool_free(dC, sizeC);
}
