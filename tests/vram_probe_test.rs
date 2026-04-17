//! Unit-ish tests for the NVIDIA VRAM probe.
//!
//! These tests hit `nvidia-smi` as a real subprocess. If `nvidia-smi` is not
//! available on the host (no NVIDIA driver, non-NVIDIA machine), each test
//! returns early with a visible "SKIPPED" line.

use std::process::Command;

use atenia_engine::amm::vram_probe::{read_nvidia_vram_free_bytes, VramProbeError};

const MIB: u64 = 1024 * 1024;

/// Returns true if `nvidia-smi --version` executes successfully.
fn nvidia_smi_available() -> bool {
    match Command::new("nvidia-smi").arg("--version").output() {
        Ok(out) => out.status.success(),
        Err(_) => false,
    }
}

fn skip_if_unavailable(test_name: &str) -> bool {
    if !nvidia_smi_available() {
        eprintln!("SKIPPED: nvidia-smi not available ({})", test_name);
        return true;
    }
    false
}

#[test]
fn test_reads_positive_value() {
    if skip_if_unavailable("test_reads_positive_value") {
        return;
    }

    let bytes = read_nvidia_vram_free_bytes().expect("probe should succeed on NVIDIA host");
    assert!(bytes > 0, "free VRAM must be > 0 bytes, got {}", bytes);
}

#[test]
fn test_value_within_reasonable_bounds() {
    if skip_if_unavailable("test_value_within_reasonable_bounds") {
        return;
    }

    let bytes = read_nvidia_vram_free_bytes().expect("probe should succeed on NVIDIA host");

    let lower = 100 * MIB;                // sanity: > 100 MB
    let upper = (8 * 1024 + 512) * MIB;   // 8.5 GiB, margin over an 8 GB card

    assert!(
        bytes > lower,
        "free VRAM unexpectedly low: {} bytes (< {} bytes)",
        bytes, lower
    );
    assert!(
        bytes < upper,
        "free VRAM unexpectedly high: {} bytes (> {} bytes); \
         this bound assumes the dev GPU is an 8 GB class card",
        bytes, upper
    );
}

#[test]
fn test_multiple_calls_consistent() {
    if skip_if_unavailable("test_multiple_calls_consistent") {
        return;
    }

    let a = read_nvidia_vram_free_bytes().expect("first probe failed");
    let b = read_nvidia_vram_free_bytes().expect("second probe failed");

    let diff = if a > b { a - b } else { b - a };
    let max_diff = 500 * MIB;

    assert!(
        diff < max_diff,
        "consecutive probes differ by {} bytes (> {} bytes); \
         expected stable VRAM between back-to-back calls",
        diff, max_diff
    );
}

#[test]
fn test_error_type_is_debug_and_eq() {
    // Compile-time sanity: the error type satisfies the contracts tests rely on.
    let e = VramProbeError::NvidiaSmiNotFound;
    let _ = format!("{:?}", e);
    assert_eq!(e, VramProbeError::NvidiaSmiNotFound);
}
