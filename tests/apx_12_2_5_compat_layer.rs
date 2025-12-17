use atenia_engine::gpu::loader::compat_layer::CompatLoader;
use atenia_engine::gpu::loader::{CudaLoader, CudaLoaderError};

#[test]
fn apx_12_2_5_fallback_always_loads() {
    let loader = match CudaLoader::new() {
        Ok(l) => l,
        Err(_) => {
            eprintln!("[APX 12.2.5] No CUDA driver available, skipping compat layer test");
            return;
        }
    };

    let ptx = r#"
        .version 8.5
        .target sm_89
        .address_size 64
        .visible .entry empty_kernel() { ret; }
    "#;

    let result = CompatLoader::try_all_paths(&loader, ptx);

    assert!(
        result.is_ok() || matches!(result, Err(CudaLoaderError::CpuFallback)),
        "Compat loader MUST always succeed (GPU or fallback)"
    );
}
