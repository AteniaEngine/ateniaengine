#![allow(dead_code)]

use super::memory_layer::MemoryLayer;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryRiskLevel {
    Safe,
    Warning,
    Critical,
    PreOOM,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureSnapshot {
    pub layer: MemoryLayer,
    pub used_bytes: u64,
    pub capacity_bytes: u64,
    pub pressure_ratio: f64,
    pub fragmentation_ratio: f64,
    pub risk_level: MemoryRiskLevel,
    pub timestamp: u64,
}

impl PressureSnapshot {
    pub fn new(
        layer: MemoryLayer,
        used_bytes: u64,
        capacity_bytes: u64,
        fragmentation_ratio: f64,
        timestamp: u64,
    ) -> Self {
        let pressure_ratio = if capacity_bytes == 0 {
            0.0
        } else {
            (used_bytes as f64) / (capacity_bytes as f64)
        };

        let risk_level = classify_risk(pressure_ratio);

        PressureSnapshot {
            layer,
            used_bytes,
            capacity_bytes,
            pressure_ratio,
            fragmentation_ratio,
            risk_level,
            timestamp,
        }
    }
}

pub fn classify_risk(pressure_ratio: f64) -> MemoryRiskLevel {
    if pressure_ratio >= 0.98 {
        MemoryRiskLevel::PreOOM
    } else if pressure_ratio >= 0.9 {
        MemoryRiskLevel::Critical
    } else if pressure_ratio >= 0.75 {
        MemoryRiskLevel::Warning
    } else {
        MemoryRiskLevel::Safe
    }
}
