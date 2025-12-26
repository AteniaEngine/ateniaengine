#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    Vram,
    Ram,
    Ssd,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionKind {
    None,
    Rle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionMeta {
    pub kind: CompressionKind,
    pub original_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StorageBacking {
    None,
    Ram(Vec<u8>),
    SsdFile { path: String, compression: Option<CompressionMeta> },
    VramHandle { key: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryFootprint {
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierStatus {
    pub total_bytes: Option<u64>,
    pub free_bytes: Option<u64>,
    pub pressure: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemorySnapshot {
    pub vram: TierStatus,
    pub ram: TierStatus,
    pub ssd: TierStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MoveError {
    Unsupported(String),
    IoError(String),
    BackendUnavailable(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MovePlan {
    pub from: MemoryTier,
    pub to: MemoryTier,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorResidence {
    pub id: TensorId,
    pub tier: MemoryTier,
    pub footprint: MemoryFootprint,
    pub backing: StorageBacking,
}

impl MemoryFootprint {
    pub fn validate_len(&self, len: usize) -> Result<(), MoveError> {
        if self.bytes == len as u64 {
            Ok(())
        } else {
            Err(MoveError::Unsupported(
                "Byte length mismatch between footprint and data".to_string(),
            ))
        }
    }
}
