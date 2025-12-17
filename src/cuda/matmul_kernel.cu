#include <cuda_runtime.h>
#include <cstdio>

extern "C" void* atenia_pool_alloc(size_t bytes);
extern "C" void  atenia_pool_free(void* ptr, size_t bytes);

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

// Nuevo launcher que asume que A, B y C ya están en device memory.
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

// Launcher legacy que sigue haciendo malloc/copy/free interno.
// Lo dejamos para compatibilidad con cualquier código previo que lo use.
extern "C"
void launch_matmul_f32(
    const float* hA,
    const float* hB,
    float* hC,
    int M, int K, int N
) {
    const size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
    const size_t sizeB = static_cast<size_t>(K) * N * sizeof(float);
    const size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

    void* rawA = atenia_pool_alloc(sizeA);
    void* rawB = atenia_pool_alloc(sizeB);
    void* rawC = atenia_pool_alloc(sizeC);

    if (!rawA || !rawB || !rawC) {
        printf("atenia_pool_alloc failed for matmul buffers\n");
        if (rawA) atenia_pool_free(rawA, sizeA);
        if (rawB) atenia_pool_free(rawB, sizeB);
        if (rawC) atenia_pool_free(rawC, sizeC);
        return;
    }

    float* dA = static_cast<float*>(rawA);
    float* dB = static_cast<float*>(rawB);
    float* dC = static_cast<float*>(rawC);

    cudaError_t err;

    // Copiar host → device
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

    // Lanzar kernel
    dim3 block2(16, 16);
    dim3 grid2((N + 15) / 16, (M + 15) / 16);

    matmul_f32_kernel<<<grid2, block2>>>(dA, dB, dC, M, K, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Sincronizar
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    // Copiar device → host
    err = cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy dC->hC failed: %s\n", cudaGetErrorString(err));
    }

cleanup:
    atenia_pool_free(dA, sizeA);
    atenia_pool_free(dB, sizeB);
    atenia_pool_free(dC, sizeC);
}
