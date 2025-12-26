#[derive(Debug, Clone)]
pub struct LearningSnapshot {
    pub entries: Vec<LearningEntrySnapshot>,
    pub summary: LearningSummary,
}

#[derive(Debug, Clone, Copy)]
pub enum BackendKind {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone)]
pub struct LearningEntrySnapshot {
    pub context: LearningContextSnapshot,
    pub recommended_backend: BackendKind,
    pub stats: LearningStatsSnapshot,
}

#[derive(Debug, Clone, Copy)]
pub struct LearningContextSnapshot {
    pub gpu_available: bool,
    pub vram_band: u8,
    pub ram_band: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct LearningStatsSnapshot {
    pub episodes: u32,
    pub successes: u32,
    pub drift_events: u32,
    pub average_score: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct LearningSummary {
    pub total_entries: usize,
    pub total_episodes: u64,
    pub gpu_preference_ratio: f32,
}

// Snapshot construction is implemented on SelfTrainer in self_trainer.rs.
