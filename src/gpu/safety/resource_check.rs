//! M6 — Resource availability check before significant GPU operations.
//!
//! After the May 2 BSOD during a 13B smoke (RAM at 99% used + GPU
//! dispatch active under `ATENIA_APX_MODE=4.19` triggered a Windows
//! WDDM bugcheck), GPU operations on this dev box must verify the
//! machine has enough free RAM and VRAM headroom **before** issuing
//! `cudaMalloc`/`cudaMemcpy` calls. Without this gate the system
//! can enter pagefile thrashing or driver instability under
//! sustained pressure.
//!
//! # Scope
//!
//! This module provides a synchronous probe used at upload time
//! (e.g., `WeightStore::upload_resident_layers` future caller) and
//! before any large CUDA allocation. It is NOT called per-matmul —
//! the cost of probing free RAM is small but non-zero, and the
//! per-matmul hot path uses cached `cuda_available()` and the
//! shape-class router instead.
//!
//! # Policy (M6 baseline; tunable in future milestones)
//!
//! | Free RAM      | Decision                          |
//! |---------------|-----------------------------------|
//! | < 8 GiB       | `DegradeToCpu` (no GPU at all)    |
//! | 8–12 GiB      | `DegradeToLayers(N)` where        |
//! |               | N = floor((free - 8 GiB) / per)  |
//! | ≥ 12 GiB      | check VRAM next                   |
//!
//! VRAM layer:
//! - `free_vram >= required_vram + 1 GiB` → `Proceed`
//! - else → `DegradeToLayers(M)` where M is reduced to fit.
//!
//! The 1 GiB VRAM headroom protects working buffers (activation
//! upload, output download, KV cache slice).
//!
//! # Telemetry
//!
//! Every decision logs to stderr prefixed with `[ATENIA]` so the
//! operator can see the policy fire without needing extra flags.
//! In `--silent` builds the log is suppressed via
//! `crate::apx_is_silent()`.

use std::process::Command;

/// Snapshot of the machine state used for the safety decision.
/// Bytes-precise; the human-readable log uses GiB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceCheck {
    pub free_ram_bytes: u64,
    pub free_vram_bytes: u64,
    pub required_ram_bytes: u64,
    pub required_vram_bytes: u64,
}

/// Outcome of the safety decision. Three variants in escalation
/// order: `Proceed` is the happy path; `DegradeToLayers` shrinks
/// the requested upload count to a smaller residency that fits
/// the remaining headroom; `DegradeToCpu` aborts the upload
/// entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyDecision {
    Proceed,
    /// Maximum number of layers the system can safely host given
    /// the current free RAM/VRAM. Caller must clamp its planned
    /// residency count to this value.
    DegradeToLayers(usize),
    /// Free RAM is below the absolute minimum (8 GiB). No GPU
    /// dispatch should be attempted — the caller must keep all
    /// weights on CPU.
    DegradeToCpu,
}

const MIN_RAM_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const SAFE_RAM_BYTES: u64 = 12 * 1024 * 1024 * 1024;
const VRAM_HEADROOM_BYTES: u64 = 1024 * 1024 * 1024;

/// Probe free physical RAM via [`sysinfo::System`]. Returns 0 on
/// failure (the caller will see this as "no RAM available" and
/// degrade to CPU). The first call refreshes the system info; the
/// caller is expected to drop the `System` instance after use so
/// the probe stays cheap.
pub fn probe_free_ram_bytes() -> u64 {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory()
}

/// Probe **total** physical RAM via [`sysinfo::System`]. Unlike
/// [`probe_free_ram_bytes`] this does not depend on the current
/// working set — it returns the box's installed memory. Consumed
/// by the M7.2 adaptive-headroom planner to decide when a model's
/// raw size overwhelms the host (13B+ on a 32 GiB box).
/// Returns 0 on failure.
pub fn probe_total_ram_bytes() -> u64 {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.total_memory()
}

/// Probe free VRAM via `nvidia-smi --query-gpu=memory.free`. Spawns
/// a child process; cost is ~50-300 ms on Windows. Intended for
/// use at upload time (once per session), not in a hot loop.
/// Returns 0 if nvidia-smi is unavailable, malformed output, or
/// any other failure — caller treats this as "no VRAM" and
/// degrades to CPU.
pub fn probe_free_vram_bytes() -> u64 {
    let out = match Command::new("nvidia-smi")
        .args(&[
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return 0,
    };
    let s = String::from_utf8_lossy(&out.stdout);
    // Expected output: "8064" (MiB). Trim trailing newline / spaces
    // and parse.
    s.trim()
        .lines()
        .next()
        .and_then(|l| l.trim().parse::<u64>().ok())
        .map(|mib| mib * 1024 * 1024)
        .unwrap_or(0)
}

/// Decide whether a GPU operation requesting `required_ram_mb` of
/// host-side staging memory and `required_vram_mb` of device-side
/// resident memory should proceed.
///
/// The function probes the live machine state and applies the
/// policy described in the module docs. Logs the decision to
/// stderr unless `apx_is_silent()` is set.
pub fn check_before_gpu_operation(required_ram_mb: u64, required_vram_mb: u64) -> SafetyDecision {
    let free_ram = probe_free_ram_bytes();
    let free_vram = probe_free_vram_bytes();
    let required_ram = required_ram_mb * 1024 * 1024;
    let required_vram = required_vram_mb * 1024 * 1024;

    let snapshot = ResourceCheck {
        free_ram_bytes: free_ram,
        free_vram_bytes: free_vram,
        required_ram_bytes: required_ram,
        required_vram_bytes: required_vram,
    };

    let decision = decide(snapshot);

    log_decision(snapshot, decision);

    decision
}

/// Pure decision function — no I/O, no logging. Exposed for unit
/// testing the policy logic without spawning processes.
pub fn decide(snapshot: ResourceCheck) -> SafetyDecision {
    let ResourceCheck {
        free_ram_bytes,
        free_vram_bytes,
        required_ram_bytes: _,
        required_vram_bytes,
    } = snapshot;

    // Tier 1: hard floor. Below 8 GiB free RAM the system is too
    // close to the pagefile cliff to safely add GPU operations.
    if free_ram_bytes < MIN_RAM_BYTES {
        return SafetyDecision::DegradeToCpu;
    }

    // Tier 2: warning band. Between 8 and 12 GiB free RAM, allow
    // GPU but cap residency to what the available headroom can
    // host without crossing the 8 GiB floor.
    //
    // The "bytes per layer" estimate uses the F32 FFN-down size
    // (270 MB) as a conservative per-layer footprint. A more
    // accurate per-model estimate would require pipeline-side
    // knowledge; the safety gate's job is to be conservative,
    // not precise.
    let bytes_per_layer_estimate: u64 = 270 * 1024 * 1024;
    if free_ram_bytes < SAFE_RAM_BYTES {
        let headroom = free_ram_bytes - MIN_RAM_BYTES;
        let layers = (headroom / bytes_per_layer_estimate) as usize;
        return SafetyDecision::DegradeToLayers(layers);
    }

    // Tier 3: VRAM check. Even with abundant RAM, if VRAM cannot
    // host the requested residency plus a 1 GiB working-buffer
    // headroom, scale back.
    if free_vram_bytes < required_vram_bytes + VRAM_HEADROOM_BYTES {
        // Compute how many layers the available VRAM can host
        // (after reserving the headroom). Divides by the F32
        // FFN-down per-layer estimate.
        let usable_vram = free_vram_bytes.saturating_sub(VRAM_HEADROOM_BYTES);
        let layers = (usable_vram / bytes_per_layer_estimate) as usize;
        return SafetyDecision::DegradeToLayers(layers);
    }

    SafetyDecision::Proceed
}

fn log_decision(snapshot: ResourceCheck, decision: SafetyDecision) {
    if crate::apx_is_silent() {
        return;
    }
    let ram_gib = snapshot.free_ram_bytes as f64 / (1024.0_f64.powi(3));
    let vram_gib = snapshot.free_vram_bytes as f64 / (1024.0_f64.powi(3));
    let req_ram_mib = snapshot.required_ram_bytes / (1024 * 1024);
    let req_vram_mib = snapshot.required_vram_bytes / (1024 * 1024);
    match decision {
        SafetyDecision::Proceed => {
            eprintln!(
                "[ATENIA] Safety check: {:.2} GiB RAM free, {:.2} GiB VRAM free \
                 (need {} MiB RAM, {} MiB VRAM) — Proceed",
                ram_gib, vram_gib, req_ram_mib, req_vram_mib
            );
        }
        SafetyDecision::DegradeToLayers(n) => {
            eprintln!(
                "[ATENIA] Safety check: {:.2} GiB RAM free, {:.2} GiB VRAM free \
                 (need {} MiB RAM, {} MiB VRAM) — DegradeToLayers({}) \
                 (insufficient headroom for full request)",
                ram_gib, vram_gib, req_ram_mib, req_vram_mib, n
            );
        }
        SafetyDecision::DegradeToCpu => {
            eprintln!(
                "[ATENIA] Safety check: {:.2} GiB RAM free (minimum 8 GiB \
                 required for GPU operations) — DegradeToCpu",
                ram_gib
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(free_ram_gib: u64, free_vram_gib: u64, req_vram_gib: u64) -> ResourceCheck {
        ResourceCheck {
            free_ram_bytes: free_ram_gib * 1024 * 1024 * 1024,
            free_vram_bytes: free_vram_gib * 1024 * 1024 * 1024,
            required_ram_bytes: 0,
            required_vram_bytes: req_vram_gib * 1024 * 1024 * 1024,
        }
    }

    #[test]
    fn six_gib_ram_degrades_to_cpu() {
        let s = snapshot(6, 8, 4);
        assert_eq!(decide(s), SafetyDecision::DegradeToCpu);
    }

    #[test]
    fn ten_gib_ram_degrades_to_layers() {
        // free 10 GiB → headroom 2 GiB → 2048 / 270 ≈ 7 layers.
        let s = snapshot(10, 8, 4);
        match decide(s) {
            SafetyDecision::DegradeToLayers(n) => {
                assert!(n >= 6 && n <= 8, "expected ~7 layers, got {}", n);
            }
            other => panic!("expected DegradeToLayers, got {:?}", other),
        }
    }

    #[test]
    fn fifteen_gib_ram_with_ample_vram_proceeds() {
        // free 15 GiB RAM (above 12 GiB safe band) AND
        // 8 GiB VRAM free with 4 GiB request + 1 GiB headroom = 5 GiB
        // needed → 8 GiB free comfortably above → Proceed.
        let s = snapshot(15, 8, 4);
        assert_eq!(decide(s), SafetyDecision::Proceed);
    }

    #[test]
    fn fifteen_gib_ram_but_low_vram_degrades_layers() {
        // RAM ok, but VRAM short: free 3 GiB, want 4 GiB + 1 GiB
        // headroom = 5 GiB needed → degrade. Usable = 3-1 = 2 GiB
        // → 2048 / 270 ≈ 7 layers.
        let s = snapshot(15, 3, 4);
        match decide(s) {
            SafetyDecision::DegradeToLayers(n) => {
                assert!(n >= 6 && n <= 8, "expected ~7 layers, got {}", n);
            }
            other => panic!("expected DegradeToLayers (VRAM-bound), got {:?}", other),
        }
    }

    #[test]
    fn eight_gib_ram_exactly_at_floor_degrades_to_layers_with_zero() {
        // At exactly 8 GiB, headroom = 0, so layers = 0. The caller
        // is expected to treat DegradeToLayers(0) as effectively
        // CpuOnly. We don't collapse the two variants here so the
        // log surface stays informative.
        let s = snapshot(8, 8, 4);
        match decide(s) {
            SafetyDecision::DegradeToLayers(0) => {}
            other => panic!("expected DegradeToLayers(0), got {:?}", other),
        }
    }

    #[test]
    fn below_floor_skips_vram_check() {
        // Even with abundant VRAM, if RAM is below 8 GiB the
        // result is DegradeToCpu — VRAM is never checked.
        let s = snapshot(4, 100, 4);
        assert_eq!(decide(s), SafetyDecision::DegradeToCpu);
    }
}
