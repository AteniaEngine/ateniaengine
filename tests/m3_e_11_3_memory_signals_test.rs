//! APX v20 M3-e.11.3 — integration tests for the vram/ram split on
//! `GuardConditions`.
//!
//! Memory pressure was collapsed into a single aggregate field
//! (`memory_pressure = max(vram, ram)`) since the first real-telemetry
//! commit. M3-e.11.3 exposes the two tiers as independent
//! `Option<f32>` fields without breaking the aggregate — the max is
//! preserved for backwards compatibility, and consumers that need
//! the discrimination (M3-e.11.5 dual-pressure `DeepDegrade` promotion
//! logic) can read the split directly.
//!
//! Five scenarios cover the invariants:
//!
//! 1. Both probes succeed: three fields populated with the correct
//!    values and the max invariant.
//! 2. VRAM-only: RAM probe errors, VRAM keeps flowing. Memory
//!    aggregate equals VRAM.
//! 3. RAM-only: mirror image.
//! 4. Both fail: `collect_guard_conditions` returns `None`
//!    (fail-open contract).
//! 5. `memory_pressure == max(vram, ram)`: sanity check against
//!    symmetric values where the max lies on one side.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use atenia_engine::amm::ram_probe::{RamProbeApi, RamProbeError, RamSnapshot};
use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::amm::vram_probe::{VramProbeApi, VramProbeError, VramSnapshot};

// --- mock probes --------------------------------------------------

/// Deterministic VRAM probe — returns either a canned snapshot or
/// an injected error on every call. Counts calls so tests can
/// assert cache / invocation semantics.
struct FixedVramProbe {
    result: Result<VramSnapshot, VramProbeError>,
    calls: AtomicU64,
}

impl FixedVramProbe {
    fn ok(total_bytes: u64, free_bytes: u64) -> Self {
        Self {
            result: Ok(VramSnapshot {
                total_bytes,
                free_bytes,
                used_bytes: total_bytes.saturating_sub(free_bytes),
            }),
            calls: AtomicU64::new(0),
        }
    }
    fn err() -> Self {
        Self {
            result: Err(VramProbeError::NvidiaSmiNotFound),
            calls: AtomicU64::new(0),
        }
    }
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

impl VramProbeApi for FixedVramProbe {
    fn snapshot(&self) -> Result<VramSnapshot, VramProbeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.result.clone()
    }
}

/// Deterministic RAM probe — mirrors `FixedVramProbe`.
struct FixedRamProbe {
    result: Result<RamSnapshot, RamProbeError>,
    calls: AtomicU64,
}

impl FixedRamProbe {
    fn ok(total_bytes: u64, available_bytes: u64) -> Self {
        Self {
            result: Ok(RamSnapshot {
                total_bytes,
                available_bytes,
                used_bytes: total_bytes.saturating_sub(available_bytes),
            }),
            calls: AtomicU64::new(0),
        }
    }
    fn err() -> Self {
        Self {
            result: Err(RamProbeError::SysinfoFailed("injected failure".into())),
            calls: AtomicU64::new(0),
        }
    }
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

impl RamProbeApi for FixedRamProbe {
    fn snapshot(&self) -> Result<RamSnapshot, RamProbeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.result.clone()
    }
}

/// Build a `SignalBus` with only the two memory probes injected —
/// every other probe is `None` so the test isolates the tier logic
/// from noise introduced by foreground / battery / CPU / GPU-util.
fn bus_with_memory_probes(
    vram: Option<Arc<dyn VramProbeApi>>,
    ram: Option<Arc<dyn RamProbeApi>>,
) -> SignalBus {
    SignalBus::with_probes(None, None, None, None, vram, ram)
}

/// Compute expected pressure from snapshot bytes, for cross-check
/// inside tests. `total > 0` assumed (the tests construct the
/// snapshots, so this is safe).
fn expected_pressure(total: u64, free_or_avail: u64) -> f32 {
    1.0 - (free_or_avail as f32 / total as f32)
}

// --- tests -------------------------------------------------------

#[test]
fn test_vram_and_ram_pressure_populated_when_both_probes_work() {
    // VRAM at 80% used (2 GiB free of 10 GiB), RAM at 50% used
    // (8 GiB available of 16 GiB). Three fields populated.
    let vram_total = 10u64 * 1024 * 1024 * 1024;
    let vram_free = 2u64 * 1024 * 1024 * 1024;
    let ram_total = 16u64 * 1024 * 1024 * 1024;
    let ram_avail = 8u64 * 1024 * 1024 * 1024;

    let vram = Arc::new(FixedVramProbe::ok(vram_total, vram_free));
    let ram = Arc::new(FixedRamProbe::ok(ram_total, ram_avail));
    let bus = bus_with_memory_probes(Some(vram.clone()), Some(ram.clone()));

    let c = bus
        .collect_guard_conditions()
        .expect("both probes succeed → Some");

    let expected_vram = expected_pressure(vram_total, vram_free); // 0.80
    let expected_ram = expected_pressure(ram_total, ram_avail); // 0.50
    let expected_max = expected_vram.max(expected_ram);

    let got_vram = c.vram_pressure.expect("vram_pressure populated");
    let got_ram = c.ram_pressure.expect("ram_pressure populated");

    assert!((got_vram - expected_vram).abs() < 1e-5, "vram {}", got_vram);
    assert!((got_ram - expected_ram).abs() < 1e-5, "ram {}", got_ram);
    assert!(
        (c.memory_pressure - expected_max).abs() < 1e-5,
        "memory_pressure not max: got {}, max(vram,ram) = {}",
        c.memory_pressure,
        expected_max
    );
    assert_eq!(vram.calls(), 1);
    assert_eq!(ram.calls(), 1);
}

#[test]
fn test_vram_pressure_only_when_ram_probe_fails() {
    // VRAM OK (40% used), RAM probe errors. `vram_pressure` flows,
    // `ram_pressure` is None, `memory_pressure` equals the vram
    // value.
    let vram_total = 10u64 * 1024 * 1024 * 1024;
    let vram_free = 6u64 * 1024 * 1024 * 1024;

    let vram = Arc::new(FixedVramProbe::ok(vram_total, vram_free));
    let ram = Arc::new(FixedRamProbe::err());
    let bus = bus_with_memory_probes(Some(vram.clone()), Some(ram.clone()));

    let c = bus
        .collect_guard_conditions()
        .expect("vram alone is enough for Some");

    let expected_vram = expected_pressure(vram_total, vram_free);
    assert!((c.vram_pressure.unwrap() - expected_vram).abs() < 1e-5);
    assert_eq!(c.ram_pressure, None);
    // The single-survivor aggregate equals the surviving tier.
    assert!((c.memory_pressure - expected_vram).abs() < 1e-5);

    // The ram probe was still invoked — failure isolation does not
    // mean silent bypass. One call per cache-miss cycle, as always.
    assert_eq!(ram.calls(), 1);
}

#[test]
fn test_ram_pressure_only_when_vram_probe_fails() {
    // Mirror image: VRAM probe errors, RAM OK.
    let ram_total = 32u64 * 1024 * 1024 * 1024;
    let ram_avail = 8u64 * 1024 * 1024 * 1024; // 75% used

    let vram = Arc::new(FixedVramProbe::err());
    let ram = Arc::new(FixedRamProbe::ok(ram_total, ram_avail));
    let bus = bus_with_memory_probes(Some(vram.clone()), Some(ram.clone()));

    let c = bus
        .collect_guard_conditions()
        .expect("ram alone is enough for Some");

    let expected_ram = expected_pressure(ram_total, ram_avail);
    assert_eq!(c.vram_pressure, None);
    assert!((c.ram_pressure.unwrap() - expected_ram).abs() < 1e-5);
    assert!((c.memory_pressure - expected_ram).abs() < 1e-5);
    assert_eq!(vram.calls(), 1);
}

#[test]
fn test_both_none_when_both_probes_fail() {
    // Neither probe can speak — the aggregate `memory_pressure`
    // has no data to derive from, so `collect_guard_conditions`
    // returns `None` (fail-open for the whole call).
    let vram = Arc::new(FixedVramProbe::err());
    let ram = Arc::new(FixedRamProbe::err());
    let bus = bus_with_memory_probes(Some(vram.clone()), Some(ram.clone()));

    let result = bus.collect_guard_conditions();
    assert!(
        result.is_none(),
        "both probes fail → call returns None; got Some"
    );

    // Both probes were consulted before we bailed.
    assert_eq!(vram.calls(), 1);
    assert_eq!(ram.calls(), 1);
}

#[test]
fn test_memory_pressure_is_max_of_vram_and_ram() {
    // vram = 0.3 (30% used), ram = 0.7 (70% used) → max = 0.7 on
    // the ram side. Flips the winner relative to the first test
    // so the max-invariant is exercised in both directions.
    let vram_total = 100u64;
    let vram_free = 70u64; // 30% used → pressure 0.3
    let ram_total = 100u64;
    let ram_avail = 30u64; // 70% used → pressure 0.7

    let vram = Arc::new(FixedVramProbe::ok(vram_total, vram_free));
    let ram = Arc::new(FixedRamProbe::ok(ram_total, ram_avail));
    let bus = bus_with_memory_probes(Some(vram.clone()), Some(ram.clone()));

    let c = bus.collect_guard_conditions().expect("both ok");

    let v = c.vram_pressure.unwrap();
    let r = c.ram_pressure.unwrap();
    assert!((v - 0.3).abs() < 1e-5);
    assert!((r - 0.7).abs() < 1e-5);
    // max is on the ram side
    assert!(
        (c.memory_pressure - r).abs() < 1e-5,
        "expected memory_pressure ({}) == ram ({})",
        c.memory_pressure,
        r
    );
    // And sanity: it's strictly greater than vram.
    assert!(
        c.memory_pressure > v,
        "memory_pressure ({}) must exceed vram ({}) when ram is higher",
        c.memory_pressure,
        v
    );
}
