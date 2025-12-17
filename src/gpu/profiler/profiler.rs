use std::time::Instant;

use crate::gpu::tensor::{GpuTensorManager, TensorGPU};
use crate::gpu::ops::matmul::MatMulOp;

/// Registro simple de latencia por tamaño de matmul cuadrado.
pub struct LatencyRecord {
    pub size: usize,
    pub ms: f32,
}

pub struct GpuProfiler;

impl GpuProfiler {
    /// Perfila matmuls cuadradas N x N para una lista de tamaños.
    ///
    /// Si no hay GPU disponible (GpuTensorManager::new() falla), devuelve vec![]
    /// para permitir early-return en tests sin CUDA.
    pub fn profile_matmul_sizes(sizes: &[usize]) -> Vec<LatencyRecord> {
        let mgr = match GpuTensorManager::new() {
            Ok(m) => m,
            Err(_) => {
                eprintln!("[PROF] No CUDA available, skipping GPU profiler");
                return Vec::new();
            }
        };

        let mut results = Vec::new();

        for &n in sizes {
            // Crear matrices A, B, C en GPU.
            let elems = n * n;
            let a_host: Vec<f32> = (0..elems).map(|i| (i % 13) as f32).collect();
            let b_host: Vec<f32> = (0..elems).map(|i| (i % 7) as f32).collect();

            let a_gpu: TensorGPU = match mgr.from_cpu_vec(&a_host, n, n) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let b_gpu: TensorGPU = match mgr.from_cpu_vec(&b_host, n, n) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let c_gpu: TensorGPU = match mgr.from_cpu_vec(&vec![0.0f32; elems], n, n) {
                Ok(t) => t,
                Err(_) => continue,
            };

            // Medir 5 lanzamientos y promediar.
            let runs = 5;
            let mut acc_ms = 0.0f32;

            for _ in 0..runs {
                let start = Instant::now();
                MatMulOp::run(&a_gpu.ptr, &b_gpu.ptr, &c_gpu.ptr, n, n, n);
                let elapsed = start.elapsed();
                acc_ms += elapsed.as_secs_f32() * 1000.0;
            }

            let avg_ms = acc_ms / runs as f32;
            eprintln!("[PROF] size={} avg={:.4} ms", n, avg_ms);

            results.push(LatencyRecord { size: n, ms: avg_ms });
        }

        results
    }
}
