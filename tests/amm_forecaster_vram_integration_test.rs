//! Integration: MemoryForecaster consuming the real VRAM probe.
//!
//! Each test skips with a visible message if `nvidia-smi` is not available.
//! The failure-path mechanics (one-shot warning, flag toggling) are covered
//! by unit tests inside `src/amm/forecaster.rs` where the private failure
//! path can be exercised directly without subprocess mocking.

use std::process::Command;

use atenia_engine::amm::forecaster::MemoryForecaster;

fn nvidia_smi_available() -> bool {
    match Command::new("nvidia-smi").arg("--version").output() {
        Ok(out) => out.status.success(),
        Err(_) => false,
    }
}

fn skip_if_unavailable(name: &str) -> bool {
    if !nvidia_smi_available() {
        eprintln!("SKIPPED: nvidia-smi not available ({})", name);
        return true;
    }
    false
}

#[test]
fn test_available_vram_returns_value() {
    if skip_if_unavailable("test_available_vram_returns_value") {
        return;
    }

    let f = MemoryForecaster::new();
    let v = f.available_vram_bytes();

    assert!(v.is_some(), "expected Some(bytes) on NVIDIA host");
    assert!(v.unwrap() > 0, "free VRAM must be > 0 bytes");
    assert!(
        !f.vram_probe_failed_once(),
        "probe failure flag should remain false after a successful call"
    );
}

#[test]
fn test_external_pressure_is_reasonable() {
    if skip_if_unavailable("test_external_pressure_is_reasonable") {
        return;
    }

    // current_bytes = 0 means "Atenia accounts for nothing", so external
    // pressure should equal the driver-reported used VRAM.
    let f = MemoryForecaster::new();
    let p = f.external_memory_pressure_bytes();

    assert!(p.is_some(), "expected Some(bytes) on NVIDIA host");

    // Plausibility bound: no consumer card exceeds 100 GiB of VRAM.
    let v = p.unwrap();
    let upper = 100u64 * 1024 * 1024 * 1024;
    assert!(
        v < upper,
        "external pressure unexpectedly high: {} bytes (> {} bytes)",
        v,
        upper
    );
}

#[test]
fn test_probe_failure_flag_stays_false_after_success() {
    if skip_if_unavailable("test_probe_failure_flag_stays_false_after_success") {
        return;
    }

    let f = MemoryForecaster::new();
    let _ = f.available_vram_bytes();
    let _ = f.external_memory_pressure_bytes();
    let _ = f.available_vram_bytes();

    assert!(
        !f.vram_probe_failed_once(),
        "multiple successful probe calls must not set the failure flag"
    );
}

#[test]
fn test_existing_byte_counter_unaffected() {
    // This test does not require nvidia-smi: it only validates that adding
    // the VRAM capability did not change the semantics of the static counter.
    let mut f = MemoryForecaster::new();
    assert_eq!(f.current_bytes, 0);
    assert_eq!(f.predicted_next_bytes, 0);
    f.current_bytes = 12_345;
    assert!(!f.is_over_limit(100_000));
    f.predicted_next_bytes = 200_000;
    assert!(f.is_over_limit(100_000));
}
