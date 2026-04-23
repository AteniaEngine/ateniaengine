//! APX v20 M3-e.6 — unit tests for `cpu_saturated_externally`.
//!
//! The function decides whether a `GuardAction::Degrade` verdict
//! should be vetoed because the CPU is saturated by external
//! processes rather than by Atenia. Its inputs are two fields on
//! `GuardConditions` (`cpu_pressure_total`, `cpu_pressure_self`),
//! and its output is a boolean consumed by the Degrade arm in
//! `Graph::check_guard_before_node`.
//!
//! Table of expected behavior (from the M3-e.6 design doc):
//!
//! | total | self | self/total | expected veto |
//! |-------|------|-----------|---------------|
//! | 0.10  | 0.05 | 0.50      | false (idle, nothing to veto) |
//! | 0.70  | 0.65 | 0.93      | false (Atenia is the cause) |
//! | 0.80  | 0.15 | 0.19      | **true** (external pressure — skip) |
//! | 0.95  | 0.60 | 0.63      | false (both loaded; Atenia is half the problem) |
//!
//! Additional cases covered: either field absent (`None`), threshold
//! boundary behavior, and share boundary behavior.

use atenia_engine::amg::reactive::{
    cpu_saturated_externally, CPU_PRESSURE_TOTAL_THRESHOLD, CPU_SELF_CONTRIBUTION_MIN,
};
use atenia_engine::v16::guards::guard_conditions::GuardConditions;

/// Construct `GuardConditions` with only the CPU-pressure fields set;
/// everything else zeroed. Lets each test focus on a single (total,
/// self_) pair without the other fields influencing anything.
fn cpu_only(total: f32, self_: f32) -> GuardConditions {
    GuardConditions::new(0.0, 0, false, false).with_cpu_pressure(total, self_)
}

// =============================================================
// 4 canonical scenarios from the M3-e.6 design table
// =============================================================

#[test]
fn scenario_idle_no_veto() {
    // System is idle. No CPU pressure to react to; Degrade should
    // proceed unimpeded regardless of who is doing what.
    let c = cpu_only(0.10, 0.05);
    assert!(
        !cpu_saturated_externally(&c),
        "idle system must not veto Degrade, got veto=true"
    );
}

#[test]
fn scenario_atenia_is_the_cause_no_veto() {
    // System loaded (70%), and Atenia accounts for almost all of
    // it (65% of the total). Atenia is the source of pressure, so
    // migrating to CPU is the right response — veto must NOT fire.
    let c = cpu_only(0.70, 0.65);
    assert!(
        !cpu_saturated_externally(&c),
        "Atenia-caused pressure must not veto Degrade, got veto=true"
    );
}

#[test]
fn scenario_external_pressure_veto_fires() {
    // System is loaded (80%), but Atenia is only 15% of it. The
    // other 65% comes from external processes (Chrome, games,
    // etc.). Migrating Atenia's work to CPU would worsen the
    // external workload. Veto must fire.
    let c = cpu_only(0.80 + 0.001, 0.15);
    assert!(
        cpu_saturated_externally(&c),
        "external-caused pressure must veto Degrade, got veto=false"
    );
}

#[test]
fn scenario_both_loaded_no_veto() {
    // System at 95%, Atenia at 60% (share ~0.63). Atenia is a big
    // part of the problem — migrating helps even if others are
    // also loaded. Veto must NOT fire; let the reaction strategy
    // try to migrate.
    let c = cpu_only(0.95, 0.60);
    assert!(
        !cpu_saturated_externally(&c),
        "shared-load scenario must not veto Degrade, got veto=true"
    );
}

// =============================================================
// Absence of the CPU signal (fail-open)
// =============================================================

#[test]
fn test_both_cpu_fields_absent_does_not_veto() {
    // No CPU probe attached, or probe failed — CPU fields are None
    // on GuardConditions. The reaction must proceed (fail-open):
    // absence of a signal must not block a migration the memory
    // guard has already requested.
    let c = GuardConditions::new(0.9, 0, false, false);
    assert_eq!(c.cpu_pressure_total, None);
    assert_eq!(c.cpu_pressure_self, None);
    assert!(
        !cpu_saturated_externally(&c),
        "absent CPU signal must be fail-open (no veto), got veto=true"
    );
}

#[test]
fn test_only_total_present_does_not_veto() {
    // Partial state should not happen through the SignalBus
    // (it suppresses half-populated readings), but the helper must
    // still be robust if another code path ever produces one.
    let mut c = GuardConditions::new(0.9, 0, false, false);
    c.cpu_pressure_total = Some(0.95);
    c.cpu_pressure_self = None;
    assert!(
        !cpu_saturated_externally(&c),
        "partial CPU signal must be fail-open (no veto)"
    );
}

#[test]
fn test_only_self_present_does_not_veto() {
    let mut c = GuardConditions::new(0.9, 0, false, false);
    c.cpu_pressure_total = None;
    c.cpu_pressure_self = Some(0.10);
    assert!(
        !cpu_saturated_externally(&c),
        "partial CPU signal must be fail-open (no veto)"
    );
}

// =============================================================
// Threshold / boundary behavior
// =============================================================

#[test]
fn test_total_at_threshold_is_not_pressure() {
    // Total exactly equal to the threshold is NOT "above" — the
    // check is strict `>`. Guarantees the semantics are consistent
    // with `SimpleMemoryPressureGuard`, which also uses strict `>`.
    let c = cpu_only(CPU_PRESSURE_TOTAL_THRESHOLD, 0.01);
    assert!(
        !cpu_saturated_externally(&c),
        "total == threshold must be treated as 'not pressure' (strict >), got veto=true"
    );
}

#[test]
fn test_total_just_above_threshold_triggers() {
    // A hair above the threshold is enough, provided share is low.
    let c = cpu_only(CPU_PRESSURE_TOTAL_THRESHOLD + 0.001, 0.01);
    assert!(
        cpu_saturated_externally(&c),
        "total just above threshold + low share must veto"
    );
}

#[test]
fn test_share_at_boundary_does_not_veto() {
    // share == CPU_SELF_CONTRIBUTION_MIN: the check is strict `<`,
    // so "share equal to the floor" means Atenia's contribution is
    // borderline — we do NOT veto, letting migration try.
    // Construct total=0.90, self=0.45 → share=0.50 (exactly).
    let c = cpu_only(0.90, 0.90 * CPU_SELF_CONTRIBUTION_MIN);
    let share = c.cpu_pressure_self.unwrap() / c.cpu_pressure_total.unwrap();
    assert!((share - CPU_SELF_CONTRIBUTION_MIN).abs() < 1e-5);
    assert!(
        !cpu_saturated_externally(&c),
        "share == min must be treated as 'enough Atenia' (strict <), got veto=true"
    );
}

#[test]
fn test_share_just_below_boundary_triggers() {
    // share just below the floor: Atenia is definitely not the
    // cause. Veto.
    let c = cpu_only(0.90, 0.90 * (CPU_SELF_CONTRIBUTION_MIN - 0.01));
    assert!(
        cpu_saturated_externally(&c),
        "share just below min + total above threshold must veto"
    );
}

// =============================================================
// Interaction with other GuardConditions fields (orthogonality)
// =============================================================

#[test]
fn test_other_fields_do_not_affect_decision() {
    // The helper reads only the two CPU fields. Memory pressure,
    // recent failures, latency spike, and pre-OOM signal must
    // not influence the veto decision — that logic belongs to
    // other guards, not here.
    let mut c = GuardConditions::new(0.95, 99, true, true); // all bad
    c.cpu_pressure_total = Some(0.10); // but CPU is fine
    c.cpu_pressure_self = Some(0.05);
    assert!(
        !cpu_saturated_externally(&c),
        "CPU-veto decision must ignore non-CPU fields"
    );
}
