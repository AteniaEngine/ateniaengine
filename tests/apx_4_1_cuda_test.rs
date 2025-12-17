use atenia_engine::cuda::*;

#[test]
fn test_cuda_vec_add() {
    if !cuda_available() {
        eprintln!("CUDA not available — skipping test");
        return;
    }

    let a = vec![1.0f32; 1024];
    let b = vec![2.0f32; 1024];

    let out = vec_add_gpu(&a, &b);

    assert_eq!(out.len(), 1024);
    for v in out {
        assert!((v - 3.0).abs() < 1e-6);
    }
}
