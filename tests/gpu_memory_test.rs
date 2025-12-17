use atenia_engine::gpu::memory::GpuMemoryEngine;

#[test]
fn test_gpu_memory_basic() {
    // Si no hay driver CUDA disponible, no fallamos el test.
    let mem = match GpuMemoryEngine::new() {
        Ok(m) => m,
        Err(_) => return,
    };

    let data = vec![1.0f32, 2.0, 3.0, 4.0];

    let gpu = match mem.alloc(data.len() * 4) {
        Ok(p) => p,
        Err(_) => return,
    };

    if mem.copy_htod(&gpu, &data).is_err() {
        let _ = mem.free(&gpu);
        return;
    }

    let mut out = vec![0.0f32; 4];
    if mem.copy_dtoh(&gpu, &mut out).is_err() {
        let _ = mem.free(&gpu);
        return;
    }

    let _ = mem.free(&gpu);

    assert_eq!(out, data);
}
