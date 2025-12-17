use crate::tensor::DType;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceTarget {
    CPU,
    GPU,
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub op_name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device_chosen: DeviceTarget,
    pub duration_us: u64,
    pub vram_before: u64,
    pub vram_after: u64,
    pub fallback: bool,
}
