//! APX 7.2 - Parallel GEMM Layer (PGL)
//! No modifica matemática ni backward.
//! Selecciona la mejor estrategia de ejecución paralela.

#[derive(Clone, Copy, Debug)]
pub enum PGLStrategy {
    Seq,
    Pex,
    WorkStealing,
}

#[derive(Clone, Copy, Debug)]
pub struct PGLDecision {
    pub strategy: PGLStrategy,
}

pub fn decide_pgl(m: usize, k: usize, n: usize, threads: usize) -> PGLDecision {
    // APX 7.3: si hay estadísticas adaptativas disponibles para el bucket
    // correspondiente, dejamos que el runtime elija en función de los
    // promedios observados. Las decisiones son efímeras (RAM) y nunca se
    // persisten.
    if crate::apx_mode_at_least("7.3") {
        let bucket = crate::apx7::adaptive_pgl::bucket_for(n);
        let guard = crate::apx7::adaptive_pgl::ADAPTIVE_BUCKETS.read().unwrap();
        if let Some((avg_seq, avg_pex, avg_ws)) = guard[bucket].avg() {
            let mut best = avg_seq;
            let mut strategy = PGLStrategy::Seq;

            if avg_pex < best {
                best = avg_pex;
                strategy = PGLStrategy::Pex;
            }
            if avg_ws < best {
                strategy = PGLStrategy::WorkStealing;
            }

            return PGLDecision { strategy };
        }
    }

    // Fallback: heurística estática de APX 7.2 basada en FLOPs y número
    // de hilos, derivada de los benchmarks en hardware real.
    let flops = (m * k * n) as f64;

    if flops < 5e7 {
        return PGLDecision { strategy: PGLStrategy::Seq };
    }

    if flops < 5e8 {
        if threads <= 8 {
            return PGLDecision { strategy: PGLStrategy::Pex };
        } else {
            return PGLDecision { strategy: PGLStrategy::WorkStealing };
        }
    }

    if threads >= 8 {
        PGLDecision { strategy: PGLStrategy::WorkStealing }
    } else {
        PGLDecision { strategy: PGLStrategy::Pex }
    }
}
