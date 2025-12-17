// src/engine/timeline/event.rs
// Launch Execution Timeline - Event structure

#[derive(Debug, Clone)]
pub struct TimelineEvent {
    pub name: String,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
    pub params: usize,
    pub timestamp_us: u128,
}

impl TimelineEvent {
    pub fn new(
        name: impl Into<String>,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        params: usize,
        timestamp_us: u128,
    ) -> Self {
        Self {
            name: name.into(),
            grid,
            block,
            shared_mem,
            params,
            timestamp_us,
        }
    }
}
