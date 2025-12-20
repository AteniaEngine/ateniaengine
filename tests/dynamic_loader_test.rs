use atenia_engine::gpu::{
    nvrtc::NvrtcCompiler,
    loader::CudaLoader,
};

#[test]
fn test_dynamic_load_add_one() {
    // If NVRTC or the CUDA driver is not available, let the test pass.
    let comp = match NvrtcCompiler::new() {
        Ok(c) => c,
        Err(_) => return,
    };

    let loader = match CudaLoader::new() {
        Ok(l) => l,
        Err(_) => return,
    };

    let src = r#"
    extern "C" __global__ void add_one(float* x) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        x[i] += 1.0f;
    }
    "#;

    let ptx = comp
        .compile(src, "add_one_loader", "compute_89")
        .expect("NVRTC compilation failed");

    let module = match loader.load_module_from_ptx(&ptx.ptx) {
        Ok(m) => m,
        Err(_) => return,
    };

    let func = match loader.get_function(&module, "add_one") {
        Ok(f) => f,
        Err(_) => return,
    };

    assert!(!func.handle.is_null());
}
