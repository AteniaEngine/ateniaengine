#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    ComputeHeavy,
    MemoryBound,
    Small,
    Serial,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelProfile {
    pub name: String,
    pub kind: KernelKind,
    pub estimated_flops: u64,
    pub estimated_bytes: u64,
}

impl KernelProfile {
    pub fn is_gpu_friendly(&self) -> bool {
        match self.kind {
            KernelKind::ComputeHeavy => true,
            KernelKind::MemoryBound => true,
            KernelKind::Small => false,
            KernelKind::Serial => false,
        }
    }
}
