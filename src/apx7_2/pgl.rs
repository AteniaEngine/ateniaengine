//! APX 7.2 - Parallel GEMM Layer (PGL)
//! Does not modify math nor backward.
//! Selects the best parallel execution strategy.

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
    // APX 7.3: if adaptive stats are available for the corresponding bucket,
    // let the runtime choose based on observed averages. Decisions are
    // ephemeral (RAM) and are never persisted.
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

    // Fallback: APX 7.2 static heuristic based on FLOPs and thread count,
    // derived from benchmarks on real hardware.
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
