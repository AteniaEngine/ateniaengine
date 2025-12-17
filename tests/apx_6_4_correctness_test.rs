use atenia_engine::apx3_8::{device_context::DeviceContext, kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8};
use atenia_engine::apx6_4::matmul_4x8_avx2;
use atenia_engine::tensor::Device;

fn run_pair(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.19).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.23).cos()).collect();

    let ctx = DeviceContext::new(Device::CPU);
    let mut out_base = vec![0.0f32; m * n];
    dispatch_matmul_apx3_8(&a, &b, &mut out_base, m, k, n, &ctx);

    let mut out_64 = vec![0.0f32; m * n];
    matmul_4x8_avx2(a.as_ptr(), b.as_ptr(), out_64.as_mut_ptr(), m, k, n);

    (out_base, out_64)
}

#[test]
fn apx_6_4_correctness_small() {
    let sizes = [(32usize, 32usize, 32usize), (64usize, 64usize, 64usize)];

    for (m, k, n) in sizes {
        let (base, micro) = run_pair(m, k, n);
        for i in 0..base.len() {
            assert!(
                (base[i] - micro[i]).abs() < 1e-4,
                "mismatch at {}: base={} micro={}",
                i,
                base[i],
                micro[i]
            );
        }
    }
}
