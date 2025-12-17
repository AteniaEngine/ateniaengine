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
fn apx_7_2_pgl_multisize_benchmark() {
    // Aseguramos modo 7.2 para que PGL esté activo en la ruta "auto".
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.2");
    }

    // Warm-up: pagamos costes de inicialización fuera de las medidas.
    {
        let a = Tensor::randn(&[256, 256], Device::CPU);
        let b = Tensor::randn(&[256, 256], Device::CPU);

        let mut flags = get_runtime_flags();
        flags.enable_pex = false;
        flags.enable_workstealing = false;
        drop(flags);

        let _ = a.matmul(&b);
        let _ = a.matmul_parallel(&b);
        let _ = a.matmul_parallel_ws(&b);
    }

    // En debug limitamos a 2048 para evitar timeouts; 4096 se puede
    // activar manualmente en ejecuciones --release.
    let sizes = [256_usize, 512, 1024, 1536, 2048 /*, 4096*/];

    const REPS: usize = 3;

    for &n in &sizes {
        let a = Tensor::randn(&[n, n], Device::CPU);
        let b = Tensor::randn(&[n, n], Device::CPU);

        // Baseline secuencial: matmul del engine con flags sin PEX/WS.
        let mut best_seq = f64::INFINITY;
        let mut best_pex = f64::INFINITY;
        let mut best_ws = f64::INFINITY;
        let mut best_auto = f64::INFINITY;

        for _ in 0..REPS {
            let t_seq = now(|| {
                let mut flags = get_runtime_flags();
                flags.enable_pex = false;
                flags.enable_workstealing = false;
                drop(flags);
                a.matmul(&b)
            });
            if t_seq < best_seq {
                best_seq = t_seq;
            }

            let t_pex = now(|| a.matmul_parallel(&b));
            if t_pex < best_pex {
                best_pex = t_pex;
            }

            let t_ws = now(|| a.matmul_parallel_ws(&b));
            if t_ws < best_ws {
                best_ws = t_ws;
            }

            let t_auto = now(|| {
                let mut flags = get_runtime_flags();
                flags.enable_pex = false;
                flags.enable_workstealing = false;
                drop(flags);
                a.matmul(&b)
            });
            if t_auto < best_auto {
                best_auto = t_auto;
            }
        }

        println!(
            "N={:4}  seq={:8.3} ms | pex={:8.3} ms | ws={:8.3} ms | auto={:8.3} ms",
            n, best_seq, best_pex, best_ws, best_auto
        );
    }
}
