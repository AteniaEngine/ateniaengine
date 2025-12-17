use atenia_engine::gpu::arch::CudaArchDetector;

#[test]
fn test_arch_detection() {
    let det = match CudaArchDetector::new() {
        Ok(d) => d,
        Err(_) => return,
    };

    let arch = match det.arch_flag() {
        Ok(a) => a,
        Err(_) => return,
    };

    assert!(
        arch == "compute_89" ||
        arch == "compute_80" ||
        arch == "compute_75" ||
        arch == "compute_61"
    );
}
