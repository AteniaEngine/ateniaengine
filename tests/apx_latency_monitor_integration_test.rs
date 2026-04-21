//! APX v19 Group 2 integration: LatencyMonitor → SignalBus → both
//! Guard conditions and Policy evidence.
//!
//! Demonstrates that recording latencies through the SignalBus-exposed
//! monitor flows into both output paths:
//! - `GuardConditions.latency_spike` reflects recent-spike state.
//! - `PolicyEvidenceSnapshot.signals` includes `StableLatency` when the
//!   monitor reports a reliable, spike-free baseline.

use std::time::Duration;

use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::v15::policy::evidence::signals::PolicySignalKind;

#[test]
fn test_latency_monitor_drives_guard_and_policy_signals() {
    let bus = SignalBus::new();

    // --- Initial state: no samples yet ---
    let initial_conditions = match bus.collect_guard_conditions() {
        Some(c) => c,
        None => {
            eprintln!("SKIPPED: memory probe unavailable");
            return;
        }
    };
    assert_eq!(
        initial_conditions.latency_spike, false,
        "latency_spike must be false before any samples"
    );

    let initial_evidence = bus
        .collect_policy_evidence()
        .expect("probe worked moments ago, must still work");
    assert!(
        !initial_evidence
            .all_signals()
            .iter()
            .any(|s| s.kind == PolicySignalKind::StableLatency),
        "StableLatency must be absent with zero samples"
    );

    // --- Enough stable samples: StableLatency appears, spike stays false ---
    let monitor = bus.latency_monitor();
    for _ in 0..15 {
        monitor.record_latency(Duration::from_millis(50));
    }

    let stable_conditions = bus
        .collect_guard_conditions()
        .expect("probe must still work");
    assert_eq!(
        stable_conditions.latency_spike, false,
        "latency_spike must remain false under stable samples"
    );

    let stable_evidence = bus
        .collect_policy_evidence()
        .expect("probe must still work");
    assert!(
        stable_evidence
            .all_signals()
            .iter()
            .any(|s| s.kind == PolicySignalKind::StableLatency),
        "StableLatency must appear with 15 stable samples"
    );

    // --- One obvious spike: latency_spike flips on, StableLatency vanishes ---
    monitor.record_latency(Duration::from_millis(500));

    let spiked_conditions = bus
        .collect_guard_conditions()
        .expect("probe must still work");
    assert_eq!(
        spiked_conditions.latency_spike, true,
        "latency_spike must be true after a 10x spike"
    );

    let spiked_evidence = bus
        .collect_policy_evidence()
        .expect("probe must still work");
    assert!(
        !spiked_evidence
            .all_signals()
            .iter()
            .any(|s| s.kind == PolicySignalKind::StableLatency),
        "StableLatency must vanish once a recent spike is detected"
    );
}
