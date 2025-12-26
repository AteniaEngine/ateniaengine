use std::sync::{Mutex, OnceLock};

use crate::v13::memory_types::{MemoryTier, TensorId};

#[derive(Debug, Clone)]
pub enum CheckpointDrift {
    MissingBackend {
        desired: MemoryTier,
    },
    TierDowngrade {
        desired: MemoryTier,
        restored: MemoryTier,
    },
    PlanMismatch {
        summary: String,
    },
}

#[derive(Debug, Clone)]
pub struct DriftReport {
    pub entry_id: TensorId,
    pub drifts: Vec<CheckpointDrift>,
}

static DRIFT_COLLECTOR: OnceLock<Mutex<Vec<DriftReport>>> = OnceLock::new();

fn collector() -> &'static Mutex<Vec<DriftReport>> {
    DRIFT_COLLECTOR.get_or_init(|| Mutex::new(Vec::new()))
}

pub(crate) fn push_report(report: DriftReport) {
    if let Ok(mut guard) = collector().lock() {
        guard.push(report);
    }
}

pub(crate) fn clear_reports() {
    if let Ok(mut guard) = collector().lock() {
        guard.clear();
    }
}

pub fn take_all_for_test() -> Vec<DriftReport> {
    match collector().lock() {
        Ok(mut guard) => {
            let reports = guard.clone();
            guard.clear();
            reports
        }
        Err(_) => Vec::new(),
    }
}
