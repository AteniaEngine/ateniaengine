#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamKind {
    Cpu,
    Gpu,
    SsdPrefetch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskKind {
    Compute { name: String },
    Transfer { name: String },
    Io { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamTask {
    pub id: u64,
    pub stream: StreamKind,
    pub kind: TaskKind,
    pub estimated_cost: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamConfig {
    pub advanced_streams_supported: bool,
}
