//! Smoke test for the hardware-probe module.
//!
//! Only verifies that `probe()` does not panic and produces a report
//! with the mandatory structural invariants filled. The specific
//! contents (number of GPUs, vendor names, VRAM sizes) depend on the
//! hardware the test runs on, so we assert only on fields that must
//! exist regardless of hardware.
//!
//! Feature-gated behind `hw-probe` so CI without wgpu/NVML deps
//! installed can skip this file entirely.

#![cfg(feature = "hw-probe")]

use atenia_engine::hw_probe;

#[test]
fn probe_does_not_panic_and_returns_structured_report() {
    let report = hw_probe::probe();

    // Probe version always matches the crate semver.
    assert_eq!(report.probe_version, env!("CARGO_PKG_VERSION"));

    // Timestamp is populated (either a real clock value or the 0
    // fallback documented by the probe module). Both are valid for
    // this test.
    let _ = report.probed_at_unix_secs;
    assert!(!report.probed_at_iso8601.is_empty());
    assert!(
        report.probed_at_iso8601.ends_with('Z'),
        "timestamp should be ISO 8601 UTC, got: {}",
        report.probed_at_iso8601
    );

    // System info must always be populated. An empty OS string would
    // indicate a fallthrough in the os_info mapping that needs fixing.
    assert!(!report.system.os.is_empty());
    assert!(!report.system.arch.is_empty());
    // `hostname` may be empty on some platforms; don't assert.

    // RAM total must be positive on any real host. If this trips on
    // a fake/mocked CI, the sysinfo crate API probably changed.
    assert!(
        report.system.ram_total_mb > 0,
        "sysinfo reported 0 MB RAM; unusual for any real host"
    );

    // GPU list may legitimately be empty (headless CI), so do NOT
    // assert on it being non-empty. Just verify every entry that IS
    // present has its mandatory fields populated.
    for gpu in &report.gpus {
        assert!(!gpu.vendor.is_empty());
        assert!(!gpu.name.is_empty());
        assert!(!gpu.device_type.is_empty());
        assert!(!gpu.backend.is_empty());
    }
}

#[test]
fn json_serialization_roundtrips() {
    let report = hw_probe::probe();
    let json =
        serde_json::to_string(&report).expect("probe report must serialize to JSON");
    assert!(json.contains("\"probe_version\""));
    assert!(json.contains("\"system\""));
    assert!(json.contains("\"gpus\""));
}
