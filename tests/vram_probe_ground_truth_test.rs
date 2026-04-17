//! Ground-truth test: the value reported by `read_nvidia_vram_free_bytes()`
//! must match an independent, direct `nvidia-smi` invocation within a small
//! tolerance (natural overhead between the two calls).

use std::process::Command;

use atenia_engine::amm::vram_probe::read_nvidia_vram_free_bytes;

const MIB: u64 = 1024 * 1024;

fn nvidia_smi_available() -> bool {
    match Command::new("nvidia-smi").arg("--version").output() {
        Ok(out) => out.status.success(),
        Err(_) => false,
    }
}

/// Independent read of free VRAM via `nvidia-smi`, used as ground truth.
/// Mirrors the same CLI invocation the probe performs internally, but is
/// written separately so the test is not self-referential.
fn reference_free_vram_bytes() -> u64 {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .expect("nvidia-smi invocation failed");

    assert!(out.status.success(), "nvidia-smi returned non-zero");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .map(|l| l.trim())
        .find(|l| !l.is_empty())
        .expect("no data line in nvidia-smi output");

    let mib: u64 = line.parse().expect("failed to parse MiB value");
    mib * MIB
}

#[test]
fn test_matches_nvidia_smi() {
    if !nvidia_smi_available() {
        eprintln!("SKIPPED: nvidia-smi not available (test_matches_nvidia_smi)");
        return;
    }

    let probe = read_nvidia_vram_free_bytes().expect("probe failed");
    let reference = reference_free_vram_bytes();

    let diff = if probe > reference {
        probe - reference
    } else {
        reference - probe
    };

    let tolerance = 50 * MIB;

    assert!(
        diff < tolerance,
        "probe vs ground truth mismatch: probe={} bytes, reference={} bytes, \
         diff={} bytes (tolerance={} bytes)",
        probe, reference, diff, tolerance
    );
}
