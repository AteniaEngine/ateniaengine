#pragma once

#ifdef __cplusplus
extern "C" {
#endif

__declspec(dllexport)
void vec_add_cuda(const float* a, const float* b, float* out, int n);

#ifdef __cplusplus
}
#endif
