#[path = "../src/v14/mod.rs"]
mod v14;

use v14::failure::failure_event::FailureSeverity;
use v14::failure::failure_kind::FailureKind;
use v14::failure::recovery_action::RecoveryAction;
use v14::failure::recovery_record::RecoveryResult;
use v14::failure::failure_trace::FailureTrace;

#[test]
fn records_prefailure_and_recovery_with_stable_json() {
    let mut trace = FailureTrace::new();

    trace.record_failure_with_recovery(
        FailureKind::OutOfMemoryRisk,
        "VRAM usage reached 95%".to_string(),
        Some("gpu0".to_string()),
        Some("tensor_A".to_string()),
        Some("kernel_X".to_string()),
        FailureSeverity::Warning,
        RecoveryAction::MoveTensorToRAM,
        "Moved tensor_A to RAM to avoid OOM".to_string(),
        RecoveryResult::Avoided,
    );

    let records = trace.records();
    assert_eq!(records.len(), 1);
    let rec = &records[0];
    assert_eq!(rec.failure_event.kind, FailureKind::OutOfMemoryRisk);
    assert_eq!(rec.failure_event.severity, FailureSeverity::Warning);
    assert_eq!(rec.result, RecoveryResult::Avoided);
    assert_eq!(rec.timestamp, 0);
    assert_eq!(rec.failure_event.timestamp, 0);

    let json1 = trace.export_json();
    let json2 = trace.export_json();
    assert_eq!(json1, json2);

    assert!(json1.starts_with('['));
    assert!(json1.ends_with(']'));
    assert!(json1.contains("\"kind\":\"OutOfMemoryRisk\""));
    assert!(json1.contains("\"severity\":\"Warning\""));
    assert!(json1.contains("\"action_taken\":\"MoveTensorToRAM\""));
    assert!(json1.contains("\"result\":\"Avoided\""));
}

#[test]
fn maintains_temporal_order_and_reset() {
    let mut trace = FailureTrace::new();

    trace.record_failure_with_recovery(
        FailureKind::TransferFailure,
        "PCIe transfer stalled".to_string(),
        Some("gpu0".to_string()),
        None,
        None,
        FailureSeverity::Warning,
        RecoveryAction::Retry,
        "Retried transfer".to_string(),
        RecoveryResult::Recovered,
    );

    trace.record_failure_with_recovery(
        FailureKind::DeviceUnavailable,
        "GPU device lost".to_string(),
        Some("gpu1".to_string()),
        None,
        None,
        FailureSeverity::Critical,
        RecoveryAction::FallbackToCPU,
        "Switched to CPU execution".to_string(),
        RecoveryResult::Degraded,
    );

    let records = trace.records();
    assert_eq!(records.len(), 2);
    assert!(records[0].timestamp < records[1].timestamp);

    trace.reset();
    assert_eq!(trace.records().len(), 0);

    // Tras reset, los timestamps vuelven a empezar en cero.
    trace.record_failure_with_recovery(
        FailureKind::Unknown,
        "Unknown error".to_string(),
        None,
        None,
        None,
        FailureSeverity::Info,
        RecoveryAction::None,
        "No action taken".to_string(),
        RecoveryResult::Failed,
    );
    let records = trace.records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].timestamp, 0);
}
