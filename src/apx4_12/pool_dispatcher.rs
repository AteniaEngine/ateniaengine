use crate::apx4_12::pool_can_alloc;

/// Dispatcher genérico que decide si ejecutar una op en GPU usando el
/// MemoryPool APX 4.12 o caer a la ruta CPU antes de lanzar el forward.
pub fn try_gpu_with_pool<F, C>(bytes_needed: usize, gpu_op: F, cpu_fallback: C)
where
    F: FnOnce(),
    C: FnOnce(),
{
    if pool_can_alloc(bytes_needed) {
        gpu_op();
    } else {
        cpu_fallback();
    }
}
