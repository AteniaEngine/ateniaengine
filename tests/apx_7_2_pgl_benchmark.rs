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
fn apx_7_2_pgl_benchmark() {
    let a = Tensor::randn(&[1024, 1024], Device::CPU);
    let b = Tensor::randn(&[1024, 1024], Device::CPU);

    // Secuencial pura
    let seq = now(|| {
        let mut flags = get_runtime_flags();
        flags.enable_pex = false;
        flags.enable_workstealing = false;
        drop(flags);
        // Matmul secuencial usando la ruta estándar del engine
        a.matmul(&b)
    });

    // PEX v1 (wrapper actual)
    let pex = now(|| a.matmul_parallel(&b));

    // PEX v2 WS (wrapper actual)
    let ws = now(|| a.matmul_parallel_ws(&b));

    // Modo auto PGL (APX 7.2 activo)
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.2");
    }
    {
        let mut flags = get_runtime_flags();
        flags.enable_pex = false;
        flags.enable_workstealing = false;
    }
    let auto = now(|| a.matmul(&b));

    println!(
        "[APX 7.2 PGL] seq={:.3} ms | pex={:.3} ms | ws={:.3} ms | auto={:.3} ms",
        seq, pex, ws, auto
    );
}
