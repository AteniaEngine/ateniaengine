use atenia_engine::gpu::nvrtc::NvrtcCompiler;
use atenia_engine::gpu::kernel::KernelNormalizer;

/// APX 12.0: basic test for kernel normalization + NVRTC compilation.
#[test]
fn apx_12_0_kernel_normalization_basic() {
    // Minimal kernel similar to the one used in MatMulOp.
    let src = r#"
    extern "C" __global__
    void matmul_kernel(const float* A,
                       const float* B,
                       float* C,
                       int N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < N && col < N) {
            float acc = 0.0f;
            for (int kk = 0; kk < N; ++kk) {
                acc += A[row * N + kk] * B[kk * N + col];
            }
            C[row * N + col] = acc;
        }
    }
    "#;

    // Pre-NVRTC normalization.
    let normalized = KernelNormalizer::normalize_kernel(src, "matmul_kernel");

    // Compile with NVRTC using compute_89 architecture (same as MatMulOp).
    let compiler = NvrtcCompiler::new().unwrap();
    let program = compiler
        .compile(&normalized, "matmul_kernel", "compute_89")
        .expect("NVRTC compile failed for normalized kernel");

    let ptx = program.ptx;

    // Verify basic PTX header.
    assert!(ptx.contains(".version"), "PTX should contain .version header");
    assert!(ptx.contains(".target"), "PTX should contain .target header");
    assert!(
        ptx.contains(".address_size 64"),
        "PTX should contain .address_size 64"
    );

    // Verify the kernel entry.
    assert!(
        ptx.contains(".visible .entry matmul_kernel"),
        "PTX should contain visible entry for matmul_kernel"
    );
}
