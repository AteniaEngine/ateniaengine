// src/engine/fingerprint.rs
// Execution Fingerprinting v1

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct ExecFingerprint {
    pub kernel: String,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
    pub params: usize,
}

impl ExecFingerprint {
    pub fn new(
        kernel: impl Into<String>,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        params: usize,
    ) -> Self {
        Self {
            kernel: kernel.into(),
            grid,
            block,
            shared_mem,
            params,
        }
    }

    pub fn hash64(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.kernel.hash(&mut h);
        self.grid.hash(&mut h);
        self.block.hash(&mut h);
        self.shared_mem.hash(&mut h);
        self.params.hash(&mut h);
        h.finish()
    }
}
