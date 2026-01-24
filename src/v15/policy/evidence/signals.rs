#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum PolicySignalKind {
    RecentRecovery,
    HighMemoryPressure,
    FragmentationWarning,
    StableLatency,
    PreOomSignal,
}

/// A single, passive evidence signal derived from lower layers (e.g. APX 14).
/// All scores are normalized in [0.0, 1.0].
#[derive(Debug, Clone, PartialEq)]
pub struct PolicySignal {
    pub kind: PolicySignalKind,
    /// Normalized severity or strength of the signal.
    pub score: f32,
}

impl PolicySignal {
    pub fn is_normalized(&self) -> bool {
        (0.0..=1.0).contains(&self.score)
    }
}
