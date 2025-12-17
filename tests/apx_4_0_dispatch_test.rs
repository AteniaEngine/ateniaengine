use atenia_engine::apx4::gpu_dispatch::{dispatch_matmul, ApxExecTarget};

#[test]
fn test_apx_4_0_cpu_fallback() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 0.0, 0.0, 1.0];
    let mut out = vec![0.0; 4];

    dispatch_matmul(&a, &b, 2, 2, 2, &mut out, ApxExecTarget::Auto);

    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
}
