// APX 8.5 — GPU Persistence Layer
// No modifica backward, kernels CPU ni la matemática del modelo.
// GPU es sólo una caché persistente opcional; CPU sigue siendo la verdad.

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
    pub reuse_score: u32,      // cuántas veces se reutilizó en GPU
    pub last_used_step: u64,   // paso global del engine
    pub tensor_bytes: usize,   // tamaño
    pub pinned: bool,          // si debe quedarse sí o sí
}
