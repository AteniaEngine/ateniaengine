use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::config::get_runtime_flags;

#[inline(always)]
fn now<F, T>(f: F) -> f64
where
    F: FnOnce() -> T,
{
    let t0 = std::time::Instant::now();
    let _ = f();
    t0.elapsed().as_secs_f64() * 1000.0
}

#[test]
fn apx_7_1_ws_benchmark() {
    let a = Tensor::randn(&[1024, 1024], Device::CPU);
    let b = Tensor::randn(&[1024, 1024], Device::CPU);

    {
        let mut flags = get_runtime_flags();
        flags.enable_pex = false;
        flags.enable_workstealing = false;
    }

    let seq = now(|| a.matmul(&b));
    let ws = now(|| a.matmul_parallel_ws(&b));

    println!("[APX 7.1] seq={:.3} ms ws={:.3} ms", seq, ws);
}
