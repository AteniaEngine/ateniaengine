#[path = "../src/v14/mod.rs"]
mod v14;

use v14::memory::memory_layer::MemoryLayer;
use v14::memory::pressure_snapshot::{MemoryRiskLevel, PressureSnapshot};
use v14::reasoning::decision_event::DecisionEventKind;
use v14::reasoning::reasoning_factors::ReasoningFactors;
use v14::reasoning::decision_reasoner::DecisionReasoner;

#[test]
fn records_decision_with_factors_and_exports_stable_json() {
    let mut reasoner = DecisionReasoner::new();

    let snapshot = PressureSnapshot::new(
        MemoryLayer::VRAM,
        900,
        1000,
        0.2,
        0,
    );

    let factors = ReasoningFactors::new(
        Some(snapshot),
        Some(MemoryRiskLevel::Critical),
        Some(0.2),
        Some(true),
        Some(3),
    );

    reasoner.record_decision(
        DecisionEventKind::DeviceSelection,
        "dec1".to_string(),
        "tensor_A".to_string(),
        factors,
        Some("CPUFallback".to_string()),
        42,
    );

    let records = reasoner.records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].event.id, "dec1");
    assert_eq!(records[0].event.object_id, "tensor_A");
    assert_eq!(records[0].timestamp, 0);
    assert_eq!(records[0].event.timestamp, 0);

    let json1 = reasoner.export_json();
    let json2 = reasoner.export_json();
    assert_eq!(json1, json2);

    // Comprobación estructural mínima
    assert!(json1.starts_with('['));
    assert!(json1.ends_with(']'));
    assert!(json1.contains("\"id\":\"dec1\""));
    assert!(json1.contains("\"kind\":\"DeviceSelection\""));
    assert!(json1.contains("\"object_id\":\"tensor_A\""));
    assert!(json1.contains("\"justification_code\":42"));
}

#[test]
fn maintains_temporal_order_and_allows_reset() {
    let mut reasoner = DecisionReasoner::new();

    let factors_empty = ReasoningFactors::new(None, None, None, None, None);

    reasoner.record_decision(
        DecisionEventKind::DeviceSelection,
        "d1".to_string(),
        "obj1".to_string(),
        factors_empty.clone(),
        None,
        1,
    );
    reasoner.record_decision(
        DecisionEventKind::TensorMovement,
        "d2".to_string(),
        "obj2".to_string(),
        factors_empty.clone(),
        None,
        2,
    );

    let records = reasoner.records();
    assert_eq!(records.len(), 2);
    assert!(records[0].timestamp < records[1].timestamp);

    reasoner.reset();
    assert_eq!(reasoner.records().len(), 0);

    // Tras reset, los timestamps vuelven a empezar en 0.
    reasoner.record_decision(
        DecisionEventKind::KernelPlacement,
        "d3".to_string(),
        "obj3".to_string(),
        factors_empty,
        None,
        3,
    );
    let records = reasoner.records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].timestamp, 0);
}
