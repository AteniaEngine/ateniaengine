use atenia_engine::matmul_dispatcher::matmul_dispatch;
use atenia_engine::apx3_8::{device_context::DeviceContext, kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8};
use atenia_engine::tensor::Device;

fn reference_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let ctx = DeviceContext::new(Device::CPU);
    let mut out = vec![0.0f32; m * n];
    dispatch_matmul_apx3_8(a, b, &mut out, m, k, n, &ctx);
    out
}

#[test]
fn tiled_matches_reference_small_sizes() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "6.1");
    }

    let m = 16;
    let k = 16;
    let n = 16;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.02).collect();

    let mut out_tiled = vec![0.0f32; m * n];
    matmul_dispatch(&a, &b, &mut out_tiled, m, k, n);

    let out_ref = reference_matmul(&a, &b, m, k, n);

    for (i, (x, y)) in out_tiled.iter().zip(out_ref.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(diff < 1e-4, "mismatch at {}: tiled={} ref={} diff={}", i, x, y, diff);
    }
}

#[test]
fn dispatcher_uses_apx3_8_when_mode_below_6_1() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "5.4");
    }

    let m = 8;
    let k = 8;
    let n = 8;
    let a: Vec<f32> = vec![1.0; m * k];
    let b: Vec<f32> = vec![1.0; k * n];

    let mut out_dispatch = vec![0.0f32; m * n];
    matmul_dispatch(&a, &b, &mut out_dispatch, m, k, n);

    let out_ref = reference_matmul(&a, &b, m, k, n);

    assert_eq!(out_dispatch, out_ref);
}

