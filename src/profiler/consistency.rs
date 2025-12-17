// APX 12.12 — GPU Consistency Scanner
// Nota: en este repo no existe run_small_matmul_for_benchmark() expuesta,
// así que usamos un stub mínimo que devuelve una latencia sintética fija.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuConsistency {
    Stable,
    JitterLow,
    JitterMedium,
    JitterHigh,
    Unstable,
}

pub struct ConsistencyReport {
    pub avg_ms: f32,
    pub max_ms: f32,
    pub jitter: f32,
    pub state: GpuConsistency,
}

pub struct ConsistencyScanner;

impl ConsistencyScanner {
    pub fn scan() -> ConsistencyReport {
        let iterations = 50;
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let t = run_small_matmul_for_benchmark_stub();
            times.push(t);
        }

        let avg = times.iter().sum::<f32>() / iterations as f32;
        let max = times.iter().cloned().fold(0.0_f32, f32::max);
        let jitter = max - avg;

        let state = if jitter < avg * 0.10 {
            GpuConsistency::Stable
        } else if jitter < avg * 0.25 {
            GpuConsistency::JitterLow
        } else if jitter < avg * 0.50 {
            GpuConsistency::JitterMedium
        } else if jitter < avg * 1.00 {
            GpuConsistency::JitterHigh
        } else {
            GpuConsistency::Unstable
        };

        ConsistencyReport {
            avg_ms: avg,
            max_ms: max,
            jitter,
            state,
        }
    }
}

// Stub mínimo: en una integración completa esto llamaría a
// gpu::ops::matmul::run_small_matmul_for_benchmark().
fn run_small_matmul_for_benchmark_stub() -> f32 {
    1.0
}
