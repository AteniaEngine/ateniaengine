//! Real system-RAM snapshots via the `sysinfo` crate.
//!
//! Parallel to `vram_probe` but targets system memory (not GPU VRAM).
//! Intentionally narrow: read-only, single-call semantics, no integration
//! with forecasters/guards/policies (that lives in `forecaster.rs`).
//!
//! ## Trait-based injection (M3-e.11.3)
//!
//! Since M3-e.11.3 the production path is exposed through the
//! [`RamProbeApi`] trait so `SignalBus` can carry the probe as
//! `Option<Arc<dyn RamProbeApi>>` and tests can inject mocks. The
//! pre-existing [`read_system_ram_snapshot`] free function is kept
//! verbatim; the trait's production struct [`RamProbe`] delegates
//! to it. Normalizes VRAM and RAM probes to the same pattern used
//! by all probes added in M3-e.6+.

use std::sync::Mutex;
use sysinfo::System;

/// Snapshot of system RAM state at a single point in time. All fields in bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RamSnapshot {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub used_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RamProbeError {
    /// `sysinfo` returned unusable memory data (e.g. zero total, or
    /// available exceeding total). Indicates the underlying OS API failed.
    SysinfoFailed(String),
}

/// Reads a full RAM snapshot (total / available / used) from the OS.
///
/// Uses `sysinfo::System::new()` + `refresh_memory()` — the lightest path
/// that populates only memory fields (no processes, CPU, or disk).
pub fn read_system_ram_snapshot() -> Result<RamSnapshot, RamProbeError> {
    let mut sys = System::new();
    sys.refresh_memory();

    let total = sys.total_memory();
    let available = sys.available_memory();

    if total == 0 {
        return Err(RamProbeError::SysinfoFailed(
            "sysinfo reported total_memory() == 0".to_string(),
        ));
    }
    if available > total {
        return Err(RamProbeError::SysinfoFailed(format!(
            "available ({}) > total ({})",
            available, total
        )));
    }

    Ok(RamSnapshot {
        total_bytes: total,
        available_bytes: available,
        used_bytes: total - available,
    })
}

// =========================================================================
// M3-e.11.3 — trait-based injection surface
// =========================================================================

/// Abstract interface over a RAM probe. Mirror of [`VramProbeApi`]
/// from the sibling module — tests inject fakes that return canned
/// snapshots; production uses [`RamProbe`], which delegates to
/// [`read_system_ram_snapshot`].
pub trait RamProbeApi: Send + Sync {
    fn snapshot(&self) -> Result<RamSnapshot, RamProbeError>;
}

/// Production RAM probe. Stateless; construction is free (no
/// subprocess, no warmup).
pub struct RamProbe {
    /// Serializes sysinfo instantiation across threads, for the
    /// same reason the other probes hold a `Mutex<()>` — keeps
    /// parallel tests from racing on the sysinfo init path.
    lock: Mutex<()>,
}

impl RamProbe {
    pub fn new() -> Self {
        Self {
            lock: Mutex::new(()),
        }
    }
}

impl Default for RamProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl RamProbeApi for RamProbe {
    fn snapshot(&self) -> Result<RamSnapshot, RamProbeError> {
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        read_system_ram_snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reads_total_positive() {
        let s = read_system_ram_snapshot().expect("RAM probe should succeed");
        assert!(s.total_bytes > 0);
    }

    #[test]
    fn test_available_leq_total() {
        let s = read_system_ram_snapshot().expect("RAM probe should succeed");
        assert!(s.available_bytes <= s.total_bytes);
    }

    #[test]
    fn test_used_consistency() {
        let s = read_system_ram_snapshot().expect("RAM probe should succeed");
        assert_eq!(s.used_bytes, s.total_bytes - s.available_bytes);
    }

    #[test]
    fn test_snapshot_reasonable() {
        let s = read_system_ram_snapshot().expect("RAM probe should succeed");
        // Any modern machine has more than 1 GiB of RAM.
        let one_gib = 1024u64 * 1024 * 1024;
        assert!(
            s.total_bytes > one_gib,
            "total RAM unexpectedly low: {} bytes",
            s.total_bytes
        );
    }
}
