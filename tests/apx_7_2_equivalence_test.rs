use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::config::get_runtime_flags;

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |acc, d| if d > acc { d } else { acc })
}

#[test]
fn apx_7_2_all_strategies_match_seq() {
    let m = 256;
    let k = 256;
    let n = 256;

    let a = Tensor::randn(&[m, k], Device::CPU);
    let b = Tensor::randn(&[k, n], Device::CPU);

    // Baseline secuencial: usar el propio matmul del engine con flags
    // en modo secuencial (sin PEX ni WS).
    {
        let mut flags = get_runtime_flags();
        flags.enable_pex = false;
        flags.enable_workstealing = false;
    }
    let seq = a.matmul(&b);

    // PEX v1 y PEX v2 WS usan los wrappers ya validados en 7.0 y 7.1.
    let pex = a.matmul_parallel(&b);
    let ws  = a.matmul_parallel_ws(&b);

    assert!(max_abs_diff(&seq, &pex) < 1e-5);
    assert!(max_abs_diff(&seq, &ws) < 1e-5);
}
