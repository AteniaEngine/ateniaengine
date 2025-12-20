use atenia_engine::gpu::nvrtc::NvrtcCompiler;

#[test]
fn test_nvrtc_compilation() {
    // If NVRTC is not available on the test machine, just exit.
    let compiler = match NvrtcCompiler::new() {
        Ok(c) => c,
        Err(_) => return,
    };

    let kernel = r#"
    extern "C" __global__ void add_one(float* x) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        x[i] += 1.0f;
    }
    "#;

    let res = compiler
        .compile(kernel, "add_one_test", "compute_89")
        .expect("NVRTC failed");

    assert!(res.ptx.len() > 20);
}
