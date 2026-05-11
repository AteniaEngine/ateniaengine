#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

// Export macros for Windows/Linux compatibility
#ifdef _WIN32
    #define CUDA_EXPORT __declspec(dllexport)
#else
    #define CUDA_EXPORT
#endif

#endif // CUDA_COMMON_H
