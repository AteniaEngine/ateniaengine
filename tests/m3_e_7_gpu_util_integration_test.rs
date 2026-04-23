//! APX v20 M3-e.7 — integration tests for the GPU compute utilization
//! probe wired into `SignalBus`.
//!
//! M3-e.7 is **observability-only**: the probe feeds `GuardConditions`
//! with `gpu_util_total` / `gpu_util_self` and the `[AMG Guard]` logs
//! reflect the values, but no reaction decision gates on them.
//! Accordingly, these tests do NOT assert anything about migration
//! behavior; they validate that the signal flows end-to-end through
//! the SignalBus without perturbing the existing reaction loop.
//!
//! Covered:
//!
//! 1. GPU fields populated on `GuardConditions` when a probe is
//!    attached (both fields present, with the values the probe
//!    returned).
//! 2. GPU fields absent (`None`) when the bus is built without a
//!    probe — fail-open default preserved.
//! 3. Cache monotonicity: rapid-fire `collect_guard_conditions`
//!    calls hit the single cache slot; both probes (CPU + GPU) are
//!    invoked exactly once per cache miss. Regression guard against
//!    a refactor that would double-probe or re-probe on cache hits.
//! 4. Probe failure on one probe does not contaminate the other:
//!    if the GPU probe errors, memory and CPU fields are still
//!    populated.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use atenia_engine::amm::cpu_probe::{CpuProbeApi, CpuProbeError, CpuSnapshot};
use atenia_engine::amm::gpu_util_probe::{
    GpuUtilProbeApi, GpuUtilProbeError, GpuUtilSnapshot,
};
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

/// A GPU probe that always fails. Used to validate that the GPU
/// probe's failure does not prevent the other signals from being
/// populated (independence contract from the SignalBus docs).
struct FailingGpuUtilProbe {
    calls: AtomicU64,
}

impl FailingGpuUtilProbe {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
        }
    }
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

impl GpuUtilProbeApi for FailingGpuUtilProbe {
    fn snapshot(&self) -> Result<GpuUtilSnapshot, GpuUtilProbeError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Err(GpuUtilProbeError::PmonFailed("injected failure".to_string()))
    }
}

// --- tests -------------------------------------------------------

#[test]
fn test_gpu_util_fields_populated_when_probe_attached() {
    let cpu = Arc::new(FixedCpuProbe::new(0.30, 0.20));
    let gpu = Arc::new(FixedGpuUtilProbe::new(0.77, 0.22));
    let bus = SignalBus::with_probes(Some(cpu.clone()), Some(gpu.clone()), None, None, None, None);

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_gpu_util_fields_populated_when_probe_attached] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.cpu_pressure_total, Some(0.30));
    assert_eq!(c.cpu_pressure_self, Some(0.20));
    assert_eq!(c.gpu_util_total, Some(0.77));
    assert_eq!(c.gpu_util_self, Some(0.22));
}

#[test]
fn test_gpu_util_fields_none_when_no_probe() {
    // Explicit "no GPU probe" construction. Memory and CPU paths
    // must keep working; GPU fields must be None.
    let cpu = Arc::new(FixedCpuProbe::new(0.10, 0.05));
    let bus = SignalBus::with_probes(Some(cpu.clone()), None, None, None, None, None);

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_gpu_util_fields_none_when_no_probe] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    assert_eq!(c.cpu_pressure_total, Some(0.10));
    assert_eq!(c.gpu_util_total, None);
    assert_eq!(c.gpu_util_self, None);
}

#[test]
fn test_gpu_probe_failure_does_not_contaminate_other_signals() {
    // Independence contract: a failing GPU probe nulls out only
    // the GPU fields, never the memory or CPU ones.
    let cpu = Arc::new(FixedCpuProbe::new(0.40, 0.25));
    let gpu = Arc::new(FailingGpuUtilProbe::new());
    let bus = SignalBus::with_probes(Some(cpu.clone()), Some(gpu.clone()), None, None, None, None);

    let Some(c) = bus.collect_guard_conditions() else {
        println!(
            "[TEST:test_gpu_probe_failure_does_not_contaminate_other_signals] \
             memory probe unavailable -> graceful skip"
        );
        return;
    };

    // CPU still populated despite GPU failure.
    assert_eq!(c.cpu_pressure_total, Some(0.40));
    assert_eq!(c.cpu_pressure_self, Some(0.25));
    // GPU fields null because the probe errored.
    assert_eq!(c.gpu_util_total, None);
    assert_eq!(c.gpu_util_self, None);
    // The GPU probe was called exactly once (the memory probe's
    // cache miss triggered one full cycle).
    assert_eq!(gpu.calls(), 1);
}

#[test]
fn test_cache_monotonicity_with_both_probes() {
    // Risk 3 from the M3-e.6 research carried forward: rapid-fire
    // calls must share a single cache slot, and BOTH probes must
    // be called at most once during the burst.
    let cpu = Arc::new(FixedCpuProbe::new(0.20, 0.10));
    let gpu = Arc::new(FixedGpuUtilProbe::new(0.33, 0.11));
    let bus = SignalBus::with_probes(Some(cpu.clone()), Some(gpu.clone()), None, None, None, None);

    // First call: populates the cache (if memory probe works).
    let first = bus.collect_guard_conditions();
    if first.is_none() {
        println!(
            "[TEST:test_cache_monotonicity_with_both_probes] \
             memory probe unavailable -> graceful skip"
        );
        return;
    }

    let probes_after_first = bus.probe_calls_count();
    let cpu_calls_after_first = cpu.calls();
    let gpu_calls_after_first = gpu.calls();
    assert_eq!(probes_after_first, 1);
    assert_eq!(cpu_calls_after_first, 1);
    assert_eq!(gpu_calls_after_first, 1);

    // 20 rapid-fire calls: every one must hit the cache. Neither
    // probe nor the bus-wide counter should move.
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
        "CPU probe must not be called during cache hits"
    );
    assert_eq!(
        gpu.calls(),
        gpu_calls_after_first,
        "GPU probe must not be called during cache hits"
    );
}

#[test]
fn test_production_signal_bus_builds_with_gpu_probe_by_default() {
    // Sanity: SignalBus::new() attaches a GPU probe unconditionally
    // (the probe is stateless and free to construct). We cannot
    // assert that the probe returns Some snapshot without requiring
    // nvidia-smi, so the test only verifies the bus builds and the
    // first call returns either Some (with or without GPU fields,
    // depending on host) or None (if memory probe fails).
    let bus = SignalBus::new();
    let result = bus.collect_guard_conditions();
    // Either outcome is acceptable; the important property is
    // no panic and no compile-time requirement for nvidia-smi.
    let _ = result;
}
