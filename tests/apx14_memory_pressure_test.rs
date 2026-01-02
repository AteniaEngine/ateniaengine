#[path = "../src/v14/mod.rs"]
mod v14;

use v14::memory::memory_layer::MemoryLayer;
use v14::memory::pressure_snapshot::{PressureSnapshot, MemoryRiskLevel};
use v14::memory::pressure_analyzer::{MemoryPressureAnalyzer, PressureTrend};
use v14::memory::fragmentation::compute_fragmentation_ratio;

#[test]
fn pressure_ratio_and_risk_level_are_correct() {
    let snapshot_safe = PressureSnapshot::new(
        MemoryLayer::RAM,
        20,
        100,
        0.0,
        0,
    );
    assert!((snapshot_safe.pressure_ratio - 0.2).abs() < 1e-9);
    assert_eq!(snapshot_safe.risk_level, MemoryRiskLevel::Safe);

    let snapshot_warning = PressureSnapshot::new(
        MemoryLayer::RAM,
        80,
        100,
        0.0,
        1,
    );
    assert!((snapshot_warning.pressure_ratio - 0.8).abs() < 1e-9);
    assert_eq!(snapshot_warning.risk_level, MemoryRiskLevel::Warning);

    let snapshot_critical = PressureSnapshot::new(
        MemoryLayer::RAM,
        95,
        100,
        0.0,
        2,
    );
    assert!((snapshot_critical.pressure_ratio - 0.95).abs() < 1e-9);
    assert_eq!(snapshot_critical.risk_level, MemoryRiskLevel::Critical);

    let snapshot_pre_oom = PressureSnapshot::new(
        MemoryLayer::RAM,
        99,
        100,
        0.0,
        3,
    );
    assert!((snapshot_pre_oom.pressure_ratio - 0.99).abs() < 1e-9);
    assert_eq!(snapshot_pre_oom.risk_level, MemoryRiskLevel::PreOOM);
}

#[test]
fn trend_detection_is_deterministic() {
    let mut analyzer = MemoryPressureAnalyzer::new(10);

    // Stable
    analyzer.record(PressureSnapshot::new(MemoryLayer::VRAM, 50, 100, 0.0, 0));
    analyzer.record(PressureSnapshot::new(MemoryLayer::VRAM, 50, 100, 0.0, 1));
    let result = analyzer.analyze();
    assert_eq!(result.trend, Some(PressureTrend::Stable));

    // Upward trend
    analyzer.reset();
    analyzer.record(PressureSnapshot::new(MemoryLayer::VRAM, 30, 100, 0.0, 0));
    analyzer.record(PressureSnapshot::new(MemoryLayer::VRAM, 60, 100, 0.0, 1));
    analyzer.record(PressureSnapshot::new(MemoryLayer::VRAM, 90, 100, 0.0, 2));
    let result = analyzer.analyze();
    assert_eq!(result.trend, Some(PressureTrend::Up));
    assert_eq!(analyzer.history().len(), 3);
    assert_eq!(result.latest.unwrap().timestamp, 2);

    // Downward trend
    analyzer.reset();
    analyzer.record(PressureSnapshot::new(MemoryLayer::VRAM, 90, 100, 0.0, 0));
    analyzer.record(PressureSnapshot::new(MemoryLayer::VRAM, 60, 100, 0.0, 1));
    analyzer.record(PressureSnapshot::new(MemoryLayer::VRAM, 30, 100, 0.0, 2));
    let result = analyzer.analyze();
    assert_eq!(result.trend, Some(PressureTrend::Down));
    assert_eq!(analyzer.history().len(), 3);
    assert_eq!(result.latest.unwrap().timestamp, 2);
}

#[test]
fn fragmentation_is_reproducible() {
    let ratio = compute_fragmentation_ratio(100, 100);
    assert!((ratio - 0.0).abs() < 1e-9);

    let ratio = compute_fragmentation_ratio(100, 50);
    assert!((ratio - 0.5).abs() < 1e-9);

    let ratio = compute_fragmentation_ratio(100, 10);
    assert!((ratio - 0.9).abs() < 1e-9);

    // total_free = 0 should be well-defined and deterministic
    let ratio = compute_fragmentation_ratio(0, 10);
    assert!((ratio - 0.0).abs() < 1e-9);

    // Ensure SSD layer is constructible and behaves like the others.
    let snapshot_ssd = PressureSnapshot::new(
        MemoryLayer::SSD,
        50,
        100,
        0.25,
        0,
    );
    assert_eq!(snapshot_ssd.layer, MemoryLayer::SSD);
}
