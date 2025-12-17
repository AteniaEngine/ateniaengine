use atenia_engine::gpu::nvrtc::NvrtcCompiler;

#[test]
fn test_nvrtc_basic() {
    let compiler = NvrtcCompiler::new().expect("❌ NVRTC not available!");

    let src = r#"
    extern "C" __global__ void test_kernel(float* x) {
        x[0] = 7.0f;
    }
    "#;

    let result = compiler.compile(src, "test_kernel", "compute_89");
    assert!(result.is_ok(), "❌ NVRTC failed to compile even a simple kernel!");
}
