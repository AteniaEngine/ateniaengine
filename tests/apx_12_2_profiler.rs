use atenia_engine::gpu::profiler::{GpuProfiler, LatencyRecord};
use atenia_engine::gpu::loader::compat_layer::CompatLoader;

/// APX 12.2: test básico del GPU Capability Profiler.
#[test]
fn apx_12_2_profiler_basic() -> Result<(), Box<dyn std::error::Error>> {
    // APX 12.x: este test es un sanity check del profiler.
    // Si el loader marcó CpuFallback, se skippea (no es un fallo del profiler).
    if CompatLoader::is_forced_fallback() {
        if std::env::var("ATENIA_DEBUG").ok().as_deref() == Some("1") {
            println!("[TEST] CPU fallback detected - skipping profiler test");
        }
        return Ok(());
    }

    // Forzar perfiles explícitos para tamaños 16 y 512
    let sizes = [16usize, 32, 64, 128, 256, 512];
    let result: Vec<LatencyRecord> = GpuProfiler::profile_matmul_sizes(&sizes);

    if result.is_empty() {
        if std::env::var("ATENIA_DEBUG").ok().as_deref() == Some("1") {
            eprintln!("[APX 12.2] No CUDA available, skipping profiler test");
        }
        return Ok(());
    }

    // Buscar latencias específicas para N=16 y N=512
    let lat16 = result
        .iter()
        .find(|r| r.size == 16)
        .map(|r| r.ms)
        .ok_or("missing latency record for size 16")?;

    let lat512 = result
        .iter()
        .find(|r| r.size == 512)
        .map(|r| r.ms)
        .ok_or("missing latency record for size 512")?;

    if std::env::var("ATENIA_DEBUG").ok().as_deref() == Some("1") {
        println!("[TEST] measured N=16 = {:.4} ms", lat16);
        println!("[TEST] measured N=512 = {:.4} ms", lat512);
    }

    // Sanity check: el profiler debe producir latencias válidas.
    // No asumimos monotonicidad estricta (puede haber jitter/overhead/cachés).
    assert!(lat16.is_finite() && lat16 >= 0.0, "invalid latency for size 16: {lat16}");
    assert!(lat512.is_finite() && lat512 >= 0.0, "invalid latency for size 512: {lat512}");

    Ok(())
}
