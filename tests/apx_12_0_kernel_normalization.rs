use atenia_engine::gpu::nvrtc::NvrtcCompiler;
use atenia_engine::gpu::kernel::KernelNormalizer;

/// APX 12.0: test básico de normalización de kernel + compilación NVRTC.
#[test]
fn apx_12_0_kernel_normalization_basic() {
    // Kernel mínimo similar al usado en MatMulOp.
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

    // Normalización previa a NVRTC.
    let normalized = KernelNormalizer::normalize_kernel(src, "matmul_kernel");

    // Compilar con NVRTC usando la arquitectura compute_89 (igual que MatMulOp).
    let compiler = NvrtcCompiler::new().unwrap();
    let program = compiler
        .compile(&normalized, "matmul_kernel", "compute_89")
        .expect("NVRTC compile failed for normalized kernel");

    let ptx = program.ptx;

    // Verificar encabezado básico de PTX.
    assert!(ptx.contains(".version"), "PTX should contain .version header");
    assert!(ptx.contains(".target"), "PTX should contain .target header");
    assert!(
        ptx.contains(".address_size 64"),
        "PTX should contain .address_size 64"
    );

    // Verificar la entrada del kernel.
    assert!(
        ptx.contains(".visible .entry matmul_kernel"),
        "PTX should contain visible entry for matmul_kernel"
    );
}
