use atenia_engine::hal::detect_cuda;

#[test]
fn test_cuda_detection() {
    assert!(detect_cuda());
}
