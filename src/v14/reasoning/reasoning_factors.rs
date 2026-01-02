#![allow(dead_code)]

use crate::v14::memory::pressure_snapshot::{MemoryRiskLevel, PressureSnapshot};

#[derive(Debug, Clone, PartialEq)]
pub struct ReasoningFactors {
    pub memory_snapshot: Option<PressureSnapshot>,
    pub memory_risk: Option<MemoryRiskLevel>,
    pub fragmentation_ratio: Option<f64>,
    pub device_available: Option<bool>,
    pub recent_decisions_count: Option<u64>,
}

impl ReasoningFactors {
    pub fn new(
        memory_snapshot: Option<PressureSnapshot>,
        memory_risk: Option<MemoryRiskLevel>,
        fragmentation_ratio: Option<f64>,
        device_available: Option<bool>,
        recent_decisions_count: Option<u64>,
    ) -> Self {
        ReasoningFactors {
            memory_snapshot,
            memory_risk,
            fragmentation_ratio,
            device_available,
            recent_decisions_count,
        }
    }
}
