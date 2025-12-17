#include <cuda_runtime.h>
#include <cstdio>

extern "C" void* atenia_pool_alloc(size_t bytes);
extern "C" void  atenia_pool_free(void* ptr, size_t bytes);

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

extern "C"
void launch_batch_matmul_f32(
    const float* hA,
    const float* hB,
    float* hC,
    int B,
    int M,
    int K,
    int N
) {
    const size_t sizeA = static_cast<size_t>(B) * M * K * sizeof(float);
    const size_t sizeB = static_cast<size_t>(B) * K * N * sizeof(float);
    const size_t sizeC = static_cast<size_t>(B) * M * N * sizeof(float);

    void* rawA = atenia_pool_alloc(sizeA);
    void* rawB = atenia_pool_alloc(sizeB);
    void* rawC = atenia_pool_alloc(sizeC);

    if (!rawA || !rawB || !rawC) {
        printf("atenia_pool_alloc failed for batch_matmul buffers\n");
        if (rawA) atenia_pool_free(rawA, sizeA);
        if (rawB) atenia_pool_free(rawB, sizeB);
        if (rawC) atenia_pool_free(rawC, sizeC);
        return;
    }

    float* dA = static_cast<float*>(rawA);
    float* dB = static_cast<float*>(rawB);
    float* dC = static_cast<float*>(rawC);

    cudaError_t err;

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

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16, B);

    batch_matmul_kernel<<<grid, block>>>(dA, dB, B, M, K, N, dC);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("batch_matmul_f32_kernel launch failed: %s\n", cudaGetErrorString(err));
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
    atenia_pool_free(dC, sizeC);
}
