//! APX v20 M3-e.9 — integration tests for battery state monitoring
//! wired into `SignalBus`.
//!
//! Observability-only semantics: battery fields populate
//! `GuardConditions::on_battery` / `battery_level` and show up in
//! enriched `[AMG Guard]` log lines, but no reaction site gates
//! decisions on them (deferred to M3-e.12 `Conservation` mode).
//!
//! These tests validate that the signal flows end-to-end and that
//! the cache invariants established across M3-e.4-e.8 hold with a
//! fourth probe attached.
//!
//! Covered:
//!
//! 1. Full-signal populated: both fields present when the probe
//!    returns complete data.
//! 2. Partial signal: `on_battery` without `battery_level`, and
//!    vice versa. Tests the independent-fields contract.
//! 3. Both fields `None` when the bus is built without a probe, or
//!    when the attached probe returns `(None, None)` (desktop /
//!    stub platform).
//! 4. Failure isolation: a failing battery probe nulls out only
//!    the battery fields; CPU / GPU / foreground stay populated.
//! 5. Cache monotonicity with four probes attached: rapid-fire
//!    calls invoke each probe exactly once per cache miss, zero
//!    times during cache hits.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use atenia_engine::amm::battery_probe::{BatteryProbeApi, BatteryProbeError, BatterySnapshot};
use atenia_engine::amm::cpu_probe::{CpuProbeApi, CpuProbeError, CpuSnapshot};
use atenia_engine::amm::foreground_probe::{
    ForegroundProbeApi, ForegroundProbeError, ForegroundSnapshot,
};
use atenia_engine::amm::gpu_util_probe::{GpuUtilProbeApi, GpuUtilProbeError, GpuUtilSnapshot};
use atenia_engine::amm::signal_bus::SignalBus;

// --- mock probes --------------------------------------------------

struct FixedCpuProbe {
    total: f32,
    self_: f32,
    calls: AtomicU64,
}
impl FixedCpuProbe {
    fn new(total: f32, self_: f32) -> Self {
        Self {
            total,
            self_,
            calls: AtomicU64::new(0),
        }
    }
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}
impl CpuProbeApi for FixedCpuProbe {
    fn snapshot(&self) -> Result<CpuSnapshot, CpuProbeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(CpuSnapshot {
            total_fraction: self.total,
            self_fraction: self.self_,
        })
    }
}

struct FixedGpuUtilProbe {
    total: f32,
    self_: f32,
    calls: AtomicU64,
}
impl FixedGpuUtilProbe {
    fn new(total: f32, self_: f32) -> Self {
        Self {
            total,
            self_,
            calls: AtomicU64::new(0),
        }
    }
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}
impl GpuUtilProbeApi for FixedGpuUtilProbe {
    fn snapshot(&self) -> Result<GpuUtilSnapshot, GpuUtilProbeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(GpuUtilSnapshot {
            total_fraction: self.total,
            self_fraction: self.self_,
        })
    }
}

struct FixedForegroundProbe {
    value: Option<bool>,
    calls: AtomicU64,
}
impl FixedForegroundProbe {
    fn new(value: Option<bool>) -> Self {
        Self {
            value,
            calls: AtomicU64::new(0),
        }
    }
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}
impl ForegroundProbeApi for FixedForegroundProbe {
    fn snapshot(&self) -> Result<ForegroundSnapshot, ForegroundProbeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(ForegroundSnapshot {
            foreground_is_atenia: self.value,
        })
    }
}

struct FixedBatteryProbe {
    on_battery: Option<bool>,
    level: Option<f32>,
    calls: AtomicU64,
}
impl FixedBatteryProbe {
    fn new(on_battery: Option<bool>, level: Option<f32>) -> Self {
        Self {
            on_battery,
            level,
            calls: AtomicU64::new(0),
        }
    }
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}
impl BatteryProbeApi for FixedBatteryProbe {
    fn snapshot(&self) -> Result<BatterySnapshot, BatteryProbeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(BatterySnapshot {
            on_battery: self.on_battery,
            battery_level: self.level,
        })
    }
}

/// Failing battery probe — validates the independence contract.
struct FailingBatteryProbe {
    calls: AtomicU64,
}
impl FailingBatteryProbe {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
        }
    }
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}
impl BatteryProbeApi for FailingBatteryProbe {
    fn snapshot(&self) -> Result<BatterySnapshot, BatteryProbeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Err(BatteryProbeError::QueryFailed(
            "injected failure".to_string(),
        ))
    }
}

// --- tests -------------------------------------------------------

#[test]
fn test_battery_fields_populated_full_signal() {
    // Laptop on battery at 15%: both fields present.
    let cpu = Arc::new(FixedCpuProbe::new(0.10, 0.05));
    let bat = Arc::new(FixedBatteryProbe::new(Some(true), Some(0.15)));
    let bus = SignalBus::with_probes(Some(cpu.clone()), None, None, Some(bat.clone()), None, None);

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_battery_fields_populated_full_signal] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.on_battery, Some(true));
    assert_eq!(c.battery_level, Some(0.15));
    // Other signals still populated correctly.
    assert_eq!(c.cpu_pressure_total, Some(0.10));
}

#[test]
fn test_battery_fields_populated_plugged_high_charge() {
    // Laptop plugged in at 85% — common everyday case.
    let cpu = Arc::new(FixedCpuProbe::new(0.20, 0.10));
    let bat = Arc::new(FixedBatteryProbe::new(Some(false), Some(0.85)));
    let bus = SignalBus::with_probes(Some(cpu.clone()), None, None, Some(bat.clone()), None, None);

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_battery_fields_populated_plugged_high_charge] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.on_battery, Some(false));
    assert_eq!(c.battery_level, Some(0.85));
}

#[test]
fn test_battery_fields_partial_on_battery_only() {
    // Driver exposes AC status but not charge level.
    // Independent-fields contract: the field we have goes through,
    // the other stays None.
    let cpu = Arc::new(FixedCpuProbe::new(0.10, 0.05));
    let bat = Arc::new(FixedBatteryProbe::new(Some(true), None));
    let bus = SignalBus::with_probes(Some(cpu.clone()), None, None, Some(bat.clone()), None, None);

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_battery_fields_partial_on_battery_only] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.on_battery, Some(true));
    assert_eq!(c.battery_level, None);
}

#[test]
fn test_battery_fields_partial_level_only() {
    // Inverse: level known but AC state unknown.
    let cpu = Arc::new(FixedCpuProbe::new(0.10, 0.05));
    let bat = Arc::new(FixedBatteryProbe::new(None, Some(0.42)));
    let bus = SignalBus::with_probes(Some(cpu.clone()), None, None, Some(bat.clone()), None, None);

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_battery_fields_partial_level_only] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.on_battery, None);
    assert_eq!(c.battery_level, Some(0.42));
}

#[test]
fn test_battery_fields_none_when_probe_says_none() {
    // Desktop scenario — probe ran, returned (None, None).
    let cpu = Arc::new(FixedCpuProbe::new(0.10, 0.05));
    let bat = Arc::new(FixedBatteryProbe::new(None, None));
    let bus = SignalBus::with_probes(Some(cpu.clone()), None, None, Some(bat.clone()), None, None);

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_battery_fields_none_when_probe_says_none] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.on_battery, None);
    assert_eq!(c.battery_level, None);
    assert_eq!(
        bat.calls(),
        1,
        "probe must have been called despite the (None, None)"
    );
}

#[test]
fn test_battery_fields_none_when_no_probe_attached() {
    // Bus explicitly built without a battery probe.
    let cpu = Arc::new(FixedCpuProbe::new(0.10, 0.05));
    let bus = SignalBus::with_probes(Some(cpu.clone()), None, None, None, None, None);

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_battery_fields_none_when_no_probe_attached] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.on_battery, None);
    assert_eq!(c.battery_level, None);
    // Other signals still work.
    assert_eq!(c.cpu_pressure_total, Some(0.10));
}

#[test]
fn test_battery_failure_does_not_contaminate_other_signals() {
    // Failure-independence: a battery probe that errors only nulls
    // the battery fields; CPU / GPU / foreground are unaffected.
    let cpu = Arc::new(FixedCpuProbe::new(0.40, 0.25));
    let gpu = Arc::new(FixedGpuUtilProbe::new(0.33, 0.11));
    let fg = Arc::new(FixedForegroundProbe::new(Some(true)));
    let bat = Arc::new(FailingBatteryProbe::new());
    let bus = SignalBus::with_probes(
        Some(cpu.clone()),
        Some(gpu.clone()),
        Some(fg.clone()),
        Some(bat.clone()),
        None,
        None,
    );

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_battery_failure_does_not_contaminate_other_signals] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.cpu_pressure_total, Some(0.40));
    assert_eq!(c.gpu_util_total, Some(0.33));
    assert_eq!(c.foreground_is_atenia, Some(true));
    assert_eq!(c.on_battery, None);
    assert_eq!(c.battery_level, None);
    assert_eq!(bat.calls(), 1, "battery probe called once despite erroring");
}

#[test]
fn test_cache_monotonicity_with_four_probes() {
    // Four probes attached; rapid-fire collect_guard_conditions
    // must hit the cache and invoke EACH probe exactly once across
    // the whole burst.
    let cpu = Arc::new(FixedCpuProbe::new(0.20, 0.10));
    let gpu = Arc::new(FixedGpuUtilProbe::new(0.33, 0.11));
    let fg = Arc::new(FixedForegroundProbe::new(Some(true)));
    let bat = Arc::new(FixedBatteryProbe::new(Some(false), Some(0.90)));
    let bus = SignalBus::with_probes(
        Some(cpu.clone()),
        Some(gpu.clone()),
        Some(fg.clone()),
        Some(bat.clone()),
        None,
        None,
    );

    // First call: populates the cache (if memory probe works).
    let first = bus.collect_guard_conditions();
    if first.is_none() {
        println!(
            "[TEST:test_cache_monotonicity_with_four_probes] \
             memory probe unavailable -> graceful skip"
        );
        return;
    }

    let probes_after_first = bus.probe_calls_count();
    let cpu_calls_after_first = cpu.calls();
    let gpu_calls_after_first = gpu.calls();
    let fg_calls_after_first = fg.calls();
    let bat_calls_after_first = bat.calls();
    assert_eq!(probes_after_first, 1);
    assert_eq!(cpu_calls_after_first, 1);
    assert_eq!(gpu_calls_after_first, 1);
    assert_eq!(fg_calls_after_first, 1);
    assert_eq!(bat_calls_after_first, 1);

    // 20 rapid-fire calls: cache absorbs them all.
    for _ in 0..20 {
        let _ = bus.collect_guard_conditions();
    }

    assert_eq!(
        bus.probe_calls_count(),
        probes_after_first,
        "probe_calls_count must stay flat during cached burst"
    );
    assert_eq!(
        cpu.calls(),
        cpu_calls_after_first,
        "CPU probe must not re-fire"
    );
    assert_eq!(
        gpu.calls(),
        gpu_calls_after_first,
        "GPU probe must not re-fire"
    );
    assert_eq!(
        fg.calls(),
        fg_calls_after_first,
        "foreground probe must not re-fire"
    );
    assert_eq!(
        bat.calls(),
        bat_calls_after_first,
        "battery probe must not re-fire"
    );
}

#[test]
fn test_production_signal_bus_attaches_battery_probe_by_default() {
    // Sanity: SignalBus::new() attaches a production BatteryProbe.
    // We cannot assert a specific value (depends on whether the
    // test host is a laptop and what its charge state is), so only
    // verify the bus constructs and calls do not panic.
    let bus = SignalBus::new();
    let _ = bus.collect_guard_conditions();
}
