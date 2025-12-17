#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void launch_matmul_f32(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
);

#ifdef __cplusplus
}
#endif
