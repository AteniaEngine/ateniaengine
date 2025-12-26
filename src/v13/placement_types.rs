#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementTarget {
    Cpu,
    Gpu,
    Ram,
    Vram,
    Ssd,
}

#[derive(Debug, Clone)]
pub struct PlacementDecision {
    pub target: PlacementTarget,
    pub device_id: Option<String>, // e.g. "gpu0" when target is Vram/Gpu
    pub reason: String,            // human-readable, for debugging/logs
}

#[derive(Debug, Clone)]
pub struct TensorProfile {
    pub num_elements: u64,
    pub element_size_bytes: u32,
    pub estimated_compute_cost: Option<f32>, // abstract units
}

impl TensorProfile {
    pub fn total_size_bytes(&self) -> u64 {
        self.num_elements * self.element_size_bytes as u64
    }
}
