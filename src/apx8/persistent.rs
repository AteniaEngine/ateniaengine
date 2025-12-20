// APX 8.5 — GPU Persistence Layer
// Does not modify backward, CPU kernels, nor model math.
// GPU is only an optional persistent cache; CPU remains the truth.

use std::sync::atomic::{AtomicU64, Ordering};

static GLOBAL_STEP: AtomicU64 = AtomicU64::new(0);

pub fn next_global_step() -> u64 {
    GLOBAL_STEP.fetch_add(1, Ordering::Relaxed)
}

pub fn current_global_step() -> u64 {
    GLOBAL_STEP.load(Ordering::Relaxed)
}

#[derive(Debug, Clone, PartialEq)]
pub struct GPUPersistenceInfo {
    pub reuse_score: u32,      // how many times it was reused on GPU
    pub last_used_step: u64,   // engine global step
    pub tensor_bytes: usize,   // size
    pub pinned: bool,          // whether it must stay no matter what
}
