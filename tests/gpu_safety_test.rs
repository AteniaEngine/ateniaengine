use atenia_engine::gpu::safety::GpuSafety;

#[test]
fn test_safety_detection() {
    assert!(GpuSafety::check(0, "OK"));
    assert!(!GpuSafety::check(700, "Invalid launch"));
    assert!(GpuSafety::should_fallback(700));
    assert!(!GpuSafety::should_fallback(2));
}
