use std::time::Instant;

use atenia_engine::apx3_8::{device_context::DeviceContext, kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8};
use atenia_engine::apx6_4::matmul_4x8_avx2;
use atenia_engine::apx6_6_auto_tiling::{AutoTilingSelector, KernelKind};
use atenia_engine::matmul_dispatcher::matmul_dispatch as auto_dispatch;
use atenia_engine::tensor::Device;

fn baseline_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, u128) {
    let ctx = DeviceContext::new(Device::CPU);
    let mut out = vec![0.0f32; m * n];
    let t0 = Instant::now();
    dispatch_matmul_apx3_8(a, b, &mut out, m, k, n, &ctx);
    let dt = t0.elapsed().as_micros();
    (out, dt)
}

fn micro_6_4(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, u128) {
    let mut out = vec![0.0f32; m * n];
    let t0 = Instant::now();
    matmul_4x8_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m, k, n);
    let dt = t0.elapsed().as_micros();
    (out, dt)
}

fn auto_6_6(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, u128, KernelKind) {
    let mut out = vec![0.0f32; m * n];
    let t0 = Instant::now();

    let kind = AutoTilingSelector::choose_kernel(m, n, k);
    match kind {
        KernelKind::Baseline38 => {
            let ctx = DeviceContext::new(Device::CPU);
            dispatch_matmul_apx3_8(a, b, &mut out, m, k, n, &ctx);
        }
        KernelKind::Tiled63B | KernelKind::Micro64 => {
            auto_dispatch(a, b, &mut out, m, k, n);
        }
    }

    let dt = t0.elapsed().as_micros();
    (out, dt, kind)
}

fn bench_size(m: usize, k: usize, n: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.173).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.293).cos()).collect();

    let (out_base, t_base) = baseline_matmul(&a, &b, m, k, n);
    let (out_64, t_64) = micro_6_4(&a, &b, m, k, n);
    let (out_66, t_66, kind) = auto_6_6(&a, &b, m, k, n);

    for i in 0..out_base.len() {
        assert!((out_base[i] - out_64[i]).abs() < 1e-4);
        assert!((out_base[i] - out_66[i]).abs() < 1e-4);
    }

    println!(
        "[APX 6.6 BENCH] size={} -> base={}us | 6.4={}us | 6.6={}us | selected={:?}",
        m, t_base, t_64, t_66, kind
    );
}

#[test]
fn apx_6_6_benchmark() {
    let sizes = [128usize, 256, 512, 1024];
    for &s in &sizes {
        bench_size(s, s, s);
    }
}
