use atenia_engine::gpu::loader::{CudaLoader, CudaLoaderError};
use atenia_engine::gpu::nvrtc::NvrtcCompiler;
use atenia_engine::gpu::loader::compat_layer::CompatLoader;

/// APX 12.x: smoke test del cache de módulos PTX usando un kernel "real".
#[test]
fn apx_12_module_cache_smoke() {
    let loader = match CudaLoader::new() {
        Ok(l) => l,
        Err(_) => {
            eprintln!("[APX 12] No CUDA driver available, skipping module cache test");
            return;
        }
    };

    let cuda_src = r#"
    extern "C" __global__
    void test_kernel(float* x, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float v = x[idx];
            x[idx] = v * 2.0f + 1.0f;
        }
    }
    "#;

    let compiler = NvrtcCompiler::new().unwrap();
    let ptx_out = compiler
        .compile(cuda_src, "test_kernel", "compute_89")
        .expect("NVRTC compile failed");

    let ptx = &ptx_out.ptx;

    // First load: si estamos en fallback CPU, saltar el test.
    let m1 = loader.load_module_from_ptx(ptx);

    if matches!(m1, Err(CudaLoaderError::CpuFallback)) || CompatLoader::is_forced_fallback() {
        println!("[TEST] CPU fallback detected  skipping module cache test");
        return;
    }

    assert!(m1.is_ok(), "First load failed");

    // Second load from cache
    let m2 = loader.load_module_from_ptx(ptx);
    assert!(m2.is_ok(), "Second load failed");
}
