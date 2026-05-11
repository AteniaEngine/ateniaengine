//! APX v20 M3-e.8 — integration tests for foreground-application
//! detection wired into `SignalBus`.
//!
//! Observability-only semantics: the foreground indicator shows up
//! in `GuardConditions::foreground_is_atenia` and in
//! `[AMG Guard]` log lines, but no reaction site currently gates
//! decisions on it. These tests only validate that the signal
//! flows end-to-end and that the cache invariants established by
//! M3-e.4 and extended through M3-e.7 are preserved with a third
//! probe attached.
//!
//! Covered:
//!
//! 1. Field populated on `GuardConditions` when a probe is attached
//!    and returns `Some(true)` or `Some(false)`.
//! 2. Field absent when the bus is built without a probe, or when
//!    the attached probe returns `Ok(None)` (platform stub).
//! 3. Probe failure (`Err`) nulls out only the foreground field —
//!    other signals remain populated.
//! 4. Cache monotonicity with three probes attached: rapid-fire
//!    calls invoke each probe exactly once per cache miss, zero
//!    times during cache hits.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// A foreground probe that always fails. Used to validate the
/// failure-independence contract: a failed foreground probe must
/// not contaminate the other signals.
struct FailingForegroundProbe {
    calls: AtomicU64,
}
impl FailingForegroundProbe {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
        }
    }
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}
impl ForegroundProbeApi for FailingForegroundProbe {
    fn snapshot(&self) -> Result<ForegroundSnapshot, ForegroundProbeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Err(ForegroundProbeError::QueryFailed(
            "injected failure".to_string(),
        ))
    }
}

// --- tests -------------------------------------------------------

#[test]
fn test_foreground_field_populated_as_atenia() {
    let cpu = Arc::new(FixedCpuProbe::new(0.10, 0.05));
    let gpu = Arc::new(FixedGpuUtilProbe::new(0.20, 0.10));
    let fg = Arc::new(FixedForegroundProbe::new(Some(true)));
    let bus = SignalBus::with_probes(
        Some(cpu.clone()),
        Some(gpu.clone()),
        Some(fg.clone()),
        None,
        None,
        None,
    );

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_foreground_field_populated_as_atenia] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.foreground_is_atenia, Some(true));
    // Spot-check the others to confirm the three-probe wire does
    // not break the pre-existing fields.
    assert_eq!(c.cpu_pressure_total, Some(0.10));
    assert_eq!(c.gpu_util_total, Some(0.20));
}

#[test]
fn test_foreground_field_populated_as_other() {
    let cpu = Arc::new(FixedCpuProbe::new(0.10, 0.05));
    let fg = Arc::new(FixedForegroundProbe::new(Some(false)));
    let bus = SignalBus::with_probes(Some(cpu.clone()), None, Some(fg.clone()), None, None, None);

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_foreground_field_populated_as_other] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.foreground_is_atenia, Some(false));
}

#[test]
fn test_foreground_field_none_when_probe_says_none() {
    // Probe ran, returned Ok(None) — platform stub behavior on
    // non-Windows hosts, or Windows in screen-lock state. The
    // field must be None, and crucially the probe MUST still be
    // called (to exercise the stub path).
    let cpu = Arc::new(FixedCpuProbe::new(0.10, 0.05));
    let fg = Arc::new(FixedForegroundProbe::new(None));
    let bus = SignalBus::with_probes(Some(cpu.clone()), None, Some(fg.clone()), None, None, None);

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_foreground_field_none_when_probe_says_none] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.foreground_is_atenia, None);
    assert_eq!(fg.calls(), 1, "probe must have been called exactly once");
}

#[test]
fn test_foreground_field_none_when_no_probe_attached() {
    // Explicit "no foreground probe" construction. The other
    // signals must still flow correctly.
    let cpu = Arc::new(FixedCpuProbe::new(0.40, 0.30));
    let bus = SignalBus::with_probes(Some(cpu.clone()), None, None, None, None, None);

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_foreground_field_none_when_no_probe_attached] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.foreground_is_atenia, None);
    assert_eq!(c.cpu_pressure_total, Some(0.40));
}

#[test]
fn test_foreground_failure_does_not_contaminate_other_signals() {
    // Failure-independence: if the foreground probe errors, only
    // the foreground field is nulled; CPU and GPU stay populated.
    let cpu = Arc::new(FixedCpuProbe::new(0.55, 0.25));
    let gpu = Arc::new(FixedGpuUtilProbe::new(0.33, 0.11));
    let fg = Arc::new(FailingForegroundProbe::new());
    let bus = SignalBus::with_probes(
        Some(cpu.clone()),
        Some(gpu.clone()),
        Some(fg.clone()),
        None,
        None,
        None,
    );

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_foreground_failure_does_not_contaminate_other_signals] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.cpu_pressure_total, Some(0.55));
    assert_eq!(c.gpu_util_total, Some(0.33));
    assert_eq!(c.foreground_is_atenia, None);
    assert_eq!(
        fg.calls(),
        1,
        "foreground probe must have been called exactly once despite erroring"
    );
}

#[test]
fn test_cache_monotonicity_with_all_three_probes() {
    // M3-e.6 risk 3 carried through e.7 and e.8: rapid-fire calls
    // hit the cache and do NOT re-probe. With a third probe now in
    // the mix, validate that every probe is still called exactly
    // once per cache miss and zero times during cache hits.
    let cpu = Arc::new(FixedCpuProbe::new(0.20, 0.10));
    let gpu = Arc::new(FixedGpuUtilProbe::new(0.33, 0.11));
    let fg = Arc::new(FixedForegroundProbe::new(Some(true)));
    let bus = SignalBus::with_probes(
        Some(cpu.clone()),
        Some(gpu.clone()),
        Some(fg.clone()),
        None,
        None,
        None,
    );

    // First call: populates the cache (if memory probe works).
    let first = bus.collect_guard_conditions();
    if first.is_none() {
        println!(
            "[TEST:test_cache_monotonicity_with_all_three_probes] \
             memory probe unavailable -> graceful skip"
        );
        return;
    }

    let probes_after_first = bus.probe_calls_count();
    let cpu_calls_after_first = cpu.calls();
    let gpu_calls_after_first = gpu.calls();
    let fg_calls_after_first = fg.calls();
    assert_eq!(probes_after_first, 1);
    assert_eq!(cpu_calls_after_first, 1);
    assert_eq!(gpu_calls_after_first, 1);
    assert_eq!(fg_calls_after_first, 1);

    // 20 rapid-fire calls.
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
        "foreground probe must not re-fire during cached burst"
    );
}

#[test]
fn test_production_signal_bus_attaches_foreground_probe_by_default() {
    // Sanity: SignalBus::new() attaches a production ForegroundProbe.
    // We cannot assert a specific value (depends on what is in
    // foreground during the test run), so only check that the bus
    // constructs and a probe call does not panic.
    let bus = SignalBus::new();
    let _ = bus.collect_guard_conditions();
}
