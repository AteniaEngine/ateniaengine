use crate::gpu::memory::{GpuMemoryEngine, GpuMemoryError, GpuPtr};

#[derive(Debug, Clone)]
pub struct StabilityMapEntry {
    pub offset: usize,
    pub read_ms: f32,
}

#[derive(Debug, Clone)]
pub struct GpuStabilityMap {
    pub entries: Vec<StabilityMapEntry>,
}

pub struct StabilityScanner {}

impl StabilityScanner {
    pub fn scan(total_bytes: usize, step: usize) -> Result<GpuStabilityMap, GpuMemoryError> {
        // Trabajamos en bytes hacia fuera, pero el engine actual copia en f32.
        // Suponemos que total_bytes y step son múltiplos de 4 (se cumple en los tests).
        let mem = GpuMemoryEngine::new()?;

        // 1) Alloc buffer completo en VRAM (en bytes)
        let base = mem.alloc(total_bytes)?;

        let step_f32 = step / 4;

        // 2) Touch memory (write) con copias H->D secuenciales
        let host_buf = vec![1.0f32; step_f32];
        let mut offset = 0usize;

        while offset < total_bytes {
            let slice_ptr = GpuPtr {
                ptr: base.ptr + offset as u64,
                size: step,
            };
            mem.copy_htod(&slice_ptr, &host_buf)?;
            offset += step;
        }

        // 3) Measure read latencies D->H secuenciales
        let mut entries = Vec::new();
        let mut readback = vec![0.0f32; step_f32];

        offset = 0;
        while offset < total_bytes {
            let slice_ptr = GpuPtr {
                ptr: base.ptr + offset as u64,
                size: step,
            };

            let t0 = std::time::Instant::now();
            mem.copy_dtoh(&slice_ptr, &mut readback)?;
            let dt = t0.elapsed().as_secs_f32() * 1000.0;

            entries.push(StabilityMapEntry {
                offset,
                read_ms: dt,
            });

            offset += step;
        }

        mem.free(&base)?;

        Ok(GpuStabilityMap { entries })
    }
}
