use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::apx7::adaptive_pgl::{ADAPTIVE_BUCKETS, bucket_for};

#[inline(always)]
fn now_ms<F, T>(f: F) -> f64
where
    F: FnOnce() -> T,
{
    let t0 = std::time::Instant::now();
    let _ = f();
    t0.elapsed().as_secs_f64() * 1000.0
}

fn run_once(n: usize, adaptive: bool) -> f64 {
    let a = Tensor::randn(&[n, n], Device::CPU);
    let b = Tensor::randn(&[n, n], Device::CPU);

    if adaptive {
        now_ms(|| a.matmul_adaptive(&b))
    } else {
        now_ms(|| a.matmul(&b))
    }
}

#[test]
fn apx_7_3_adaptive_before_after_benchmark() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.3");
    }

    let sizes = [256_usize, 512, 1024];

    // Fase 1: medir "antes" (PGL 7.2 heurístico, sin adaptive flag).
    let debug = std::env::var("ATENIA_DEBUG").ok().as_deref() == Some("1");
    if debug {
        println!("[APX 7.3] BEFORE adaptive training:");
    }
    for &n in &sizes {
        let t = run_once(n, false);
        if debug {
            println!("  N={:4} -> matmul (no adaptive) = {:8.3} ms", n, t);
        }
    }

    // Fase 2: entrenamiento adaptativo usando matmul_adaptive.
    if debug {
        println!("[APX 7.3] Training adaptive PGL...");
    }
    for _ in 0..5 {
        for &n in &sizes {
            let _ = run_once(n, true);
        }
    }

    // Leer estadísticas aprendidas.
    let buckets = ADAPTIVE_BUCKETS.read().unwrap();
    for &n in &sizes {
        let b = bucket_for(n);
        if let Some((avg_seq, avg_pex, avg_ws)) = buckets[b].avg() {
            if debug {
                println!(
                    "  Bucket for N~{:4}: avg_seq={:8.3} ms | avg_pex={:8.3} ms | avg_ws={:8.3} ms",
                    n, avg_seq, avg_pex, avg_ws
                );
            }
        } else if debug {
            println!("  Bucket for N~{:4}: not enough samples yet", n);
        }
    }

    // Fase 3: medir "después" con adaptive activo.
    if debug {
        println!("[APX 7.3] AFTER adaptive training:");
    }
    for &n in &sizes {
        let t = run_once(n, true);
        if debug {
            println!("  N={:4} -> matmul_adaptive = {:8.3} ms", n, t);
        }
    }
}
