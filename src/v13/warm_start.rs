use crate::v13::checkpoint::drift::DriftReport;
use crate::v13::checkpoint::{HybridCheckpoint, WarmStartAction, WarmStartDecision, WarmStartPlan};
use crate::v13::hybrid_memory::HybridMemoryManager;
use crate::v13::memory_types::{MemoryTier, TensorId};

fn tier_to_str(tier: MemoryTier) -> &'static str {
    match tier {
        MemoryTier::Vram => "Vram",
        MemoryTier::Ram => "Ram",
        MemoryTier::Ssd => "Ssd",
        MemoryTier::Cpu => "Cpu",
    }
}

fn find_drift_for_id<'a>(
    id: &str,
    drifts: &'a [DriftReport],
) -> Option<&'a DriftReport> {
    let needle = TensorId(id.to_string());
    for report in drifts {
        if report.entry_id == needle {
            return Some(report);
        }
    }
    None
}

fn pick_action_for_entry(
    id: &str,
    is_grad: bool,
    current: MemoryTier,
    desired: Option<MemoryTier>,
    drift: Option<&DriftReport>,
    gpu_available: bool,
) -> WarmStartDecision {
    // Drift-aware downgrade has priority.
    if let Some(report) = drift {
        for d in &report.drifts {
            if let crate::v13::checkpoint::drift::CheckpointDrift::TierDowngrade { desired, restored } = d {
                let reason = format!(
                    "Tier downgraded from {} to {} during restore drift",
                    tier_to_str(*desired),
                    tier_to_str(*restored),
                );
                return WarmStartDecision {
                    id: id.to_string(),
                    is_grad,
                    current,
                    desired: Some(*desired),
                    action: WarmStartAction::DegradeSafe { to: *restored },
                    reason,
                };
            }
        }
    }

    match desired {
        Some(MemoryTier::Vram) => {
            let (action, reason) = if gpu_available {
                (
                    WarmStartAction::HintPromote { to: MemoryTier::Vram },
                    "Desired VRAM and GPU available".to_string(),
                )
            } else {
                (
                    WarmStartAction::DegradeSafe { to: MemoryTier::Ram },
                    "Desired VRAM but GPU unavailable".to_string(),
                )
            };
            return WarmStartDecision {
                id: id.to_string(),
                is_grad,
                current,
                desired: Some(MemoryTier::Vram),
                action,
                reason,
            };
        }
        Some(MemoryTier::Ram) => {
            if matches!(current, MemoryTier::Ssd) {
                let action = WarmStartAction::HintPromote { to: MemoryTier::Ram };
                let reason = "Prefer RAM for faster access when safe".to_string();
                return WarmStartDecision {
                    id: id.to_string(),
                    is_grad,
                    current,
                    desired,
                    action,
                    reason,
                };
            } else {
                let action = WarmStartAction::Keep;
                let reason = "Desired tier already matches current placement".to_string();
                return WarmStartDecision {
                    id: id.to_string(),
                    is_grad,
                    current,
                    desired,
                    action,
                    reason,
                };
            }
        }
        Some(other) => {
            // For Cpu/Ssd desired tiers, default to keep with a neutral reason.
            let action = WarmStartAction::Keep;
            let reason = format!(
                "Desired tier is {} and current placement is {}, no change planned",
                tier_to_str(other),
                tier_to_str(current),
            );
            return WarmStartDecision {
                id: id.to_string(),
                is_grad,
                current,
                desired,
                action,
                reason,
            };
        }
        None => {
            let action = WarmStartAction::Keep;
            let reason = "No desired tier hint".to_string();
            return WarmStartDecision {
                id: id.to_string(),
                is_grad,
                current,
                desired: None,
                action,
                reason,
            };
        }
    }
}

pub fn build_warm_start_plan(
    mem: &HybridMemoryManager,
    checkpoint: &HybridCheckpoint,
    drift: &[DriftReport],
    gpu_available: bool,
) -> WarmStartPlan {
    let mut decisions: Vec<WarmStartDecision> = Vec::new();

    for entry in &checkpoint.entries {
        let current = match mem.get_tier(&entry.id) {
            Some(t) => t,
            None => entry.tier,
        };

        let desired = mem.get_desired_tier_hint(&entry.id).or(entry.desired_tier);

        let drift_for_id = find_drift_for_id(&entry.id, drift);

        let decision = pick_action_for_entry(
            &entry.id,
            entry.is_grad,
            current,
            desired,
            drift_for_id,
            gpu_available,
        );

        decisions.push(decision);
    }

    // Deterministic ordering: (is_grad, id).
    decisions.sort_by(|a, b| {
        let ag = if a.is_grad { 1u8 } else { 0u8 };
        let bg = if b.is_grad { 1u8 } else { 0u8 };
        ag.cmp(&bg).then_with(|| a.id.cmp(&b.id))
    });

    let mut keep = 0usize;
    let mut promote = 0usize;
    let mut degrade = 0usize;

    for d in &decisions {
        match d.action {
            WarmStartAction::Keep => keep += 1,
            WarmStartAction::HintPromote { .. } => promote += 1,
            WarmStartAction::DegradeSafe { .. } => degrade += 1,
        }
    }

    let summary = format!("warm_start: keep={} promote={} degrade={}", keep, promote, degrade);

    WarmStartPlan { decisions, summary }
}

pub fn apply_warm_start_plan_summaries(mem: &mut HybridMemoryManager, plan: &WarmStartPlan) {
    for d in &plan.decisions {
        mem.set_last_plan_summary(&d.id, Some(d.reason.clone()));
        mem.set_desired_tier_hint(&d.id, d.desired);
    }
}
