use crate::apx4_12::pool_can_alloc;

/// Generic dispatcher that decides whether to run an op on GPU using the
/// APX 4.12 MemoryPool or fall back to the CPU path before launching forward.
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
