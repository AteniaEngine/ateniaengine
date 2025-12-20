use atenia_engine::gpu::runtime::GpuRuntime;

#[test]
fn test_runtime_initialization() {
    // If no CUDA driver is available, we do not fail the test.
    let rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(_) => return,
    };

    assert!(!rt.default_stream.is_null());
}
