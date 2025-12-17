use atenia_engine::apx3_8::{device_context::DeviceContext, kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8};
use atenia_engine::apx6_5::matmul_tiled_6_5;
use atenia_engine::tensor::Device;

fn run_pair(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.41).cos()).collect();

    let ctx = DeviceContext::new(Device::CPU);
    let mut out_base = vec![0.0f32; m * n];
    dispatch_matmul_apx3_8(&a, &b, &mut out_base, m, k, n, &ctx);

    let mut out_65 = vec![0.0f32; m * n];
    matmul_tiled_6_5(&a, &b, &mut out_65, m, k, n);

    (out_base, out_65)
}

#[test]
fn apx_6_5_correctness_small() {
    let sizes = [
        (16usize, 16usize, 16usize),
        (32usize, 32usize, 32usize),
        (64usize, 64usize, 64usize),
    ];

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
