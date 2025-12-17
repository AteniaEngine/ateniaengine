use atenia_engine::gpu::runtime::GpuRuntime;

#[test]
fn test_runtime_initialization() {
    // Si no hay driver CUDA, no fallamos el test.
    let rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(_) => return,
    };

    assert!(!rt.default_stream.is_null());
}
