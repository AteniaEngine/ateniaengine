//! APX v19 Group 1 integration: FailureCounter → SignalBus → both
//! Guard conditions and Policy evidence.
//!
//! Demonstrates that recording failures through the SignalBus-exposed
//! counter flows into both output paths:
//! - `GuardConditions.recent_failures` reflects the in-window count.
//! - `PolicyEvidenceSnapshot.signals` includes `RecentRecovery` when the
//!   last failure falls within the "just recovered" window.
//!
//! This test only verifies the branch where RecentRecovery is **absent**
//! (elapsed < 10s), because the lower bound of the recovery window is
//! hardcoded at 10 seconds in `SignalBus::collect_policy_evidence`.
//! Exercising the branch where RecentRecovery fires would require either
//! a real 10-second sleep or a parameterized SignalBus (scope creep for
//! APX v19 Group 1). A dedicated slow test or a later refactor can
//! cover that branch.

use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::v15::policy::evidence::signals::PolicySignalKind;

#[test]
fn test_failure_counter_drives_guard_and_policy_signals() {
    let bus = SignalBus::new();
    let counter = bus.failure_counter();

    for _ in 0..5 {
        counter.record_failure();
    }

    // --- Guard conditions path ---
    let conditions = match bus.collect_guard_conditions() {
        Some(c) => c,
        None => {
            eprintln!("SKIPPED: memory probe unavailable");
            return;
        }
    };
    assert_eq!(
        conditions.recent_failures, 5,
        "recent_failures must reflect the 5 failures we just recorded"
    );

    // --- Policy evidence path ---
    let evidence = bus
        .collect_policy_evidence()
        .expect("probe worked moments ago, must still work");
    let has_recovery = evidence
        .all_signals()
        .iter()
        .any(|s| s.kind == PolicySignalKind::RecentRecovery);
    assert!(
        !has_recovery,
        "RecentRecovery must not fire within 10s of a failure"
    );
}
