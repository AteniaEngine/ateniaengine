#![allow(dead_code)]

/// A logical description of how model bytes are mapped in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySegment {
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryMap {
    pub artifact_id: String,
    pub total_size_bytes: u64,
    pub loaded_bytes: u64,
    pub segments: Vec<MemorySegment>,
}

impl MemoryMap {
    pub fn fully_loaded(&self) -> bool {
        self.loaded_bytes == self.total_size_bytes
    }
}
