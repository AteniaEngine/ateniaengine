use atenia_engine::apx3_8::{device_context::DeviceContext, kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8};
use atenia_engine::apx6_3::tiled_avx2::matmul_tiled_avx2;
use atenia_engine::tensor::Device;

fn run_baseline(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.17).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.29).cos()).collect();

    let ctx = DeviceContext::new(Device::CPU);
    let mut out_base = vec![0.0f32; m * n];
    dispatch_matmul_apx3_8(&a, &b, &mut out_base, m, k, n, &ctx);

    let mut out_tiled = vec![0.0f32; m * n];
    matmul_tiled_avx2(&a, &b, &mut out_tiled, m, k, n);

    (out_base, out_tiled)
}

#[test]
fn apx_6_3_correctness_small_sizes() {
    let sizes = [(16usize, 16usize, 16usize), (32, 32, 32), (64, 64, 64)];

    for (m, k, n) in sizes {
        let (base, tiled) = run_baseline(m, k, n);
        for i in 0..base.len() {
            assert!(
                (base[i] - tiled[i]).abs() < 1e-4,
                "mismatch at {}: base={} tiled={}",
                i,
                base[i],
                tiled[i]
            );
        }
    }
}
