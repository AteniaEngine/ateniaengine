//! Real-signal producer for v16 Guard conditions.
//!
//! The SignalBus pulls live telemetry from memory probes on demand and
//! constructs a `GuardConditions` struct that downstream Guards (v16)
//! can evaluate. It closes the first real sensor → decision path in the
//! engine.
//!
//! Scope in APX v19:
//! - All four `GuardConditions` fields (`memory_pressure`, `pre_oom_signal`,
//!   `recent_failures`, `latency_spike`) are sourced from real telemetry:
//!   memory probes for the first two, `FailureCounter` and
//!   `LatencyMonitor` for the last two.
//! - Four of the five `PolicySignalKind` variants are produced. The
//!   remaining one (`FragmentationWarning`) is intentionally not
//!   emitted; see the `collect_policy_evidence` doc for the rationale.
//!
//! Note: this module calls the VRAM/RAM probes directly (one snapshot
//! per tier). The bus holds no state today; if future milestones require
//! persistent state (failure counters, latency baselines, etc.), it will
//! live either on this type or on dedicated producer types, not on the
//! forecaster.
//!
//! One consequence of bypassing the forecaster: the forecaster's
//! one-shot "probe unavailable" warning is not triggered when the bus
//! fails to collect. Callers that need explicit failure logging must do
//! so themselves when `collect_guard_conditions` returns `None`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::amm::cpu_probe::{CpuProbe, CpuProbeApi};
use crate::amm::failure_counter::FailureCounter;
use crate::amm::latency_monitor::LatencyMonitor;
use crate::amm::ram_probe;
use crate::amm::vram_probe;
use crate::v15::policy::evidence::signals::{PolicySignal, PolicySignalKind};
use crate::v15::policy::evidence::snapshot::PolicyEvidenceSnapshot;
use crate::v16::guards::guard_conditions::GuardConditions;

/// Caching TTL for memory pressure probes. Balances subprocess
/// overhead (~40ms per probe via `nvidia-smi`) against detection
/// latency. Values under 100ms make caching ineffective on
/// rapid-fire calls; values over 100ms delay reaction to genuine
/// pressure spikes. 100ms is the empirical starting point; adjust
/// based on observed workload behavior.
pub const SIGNAL_BUS_CACHE_TTL: Duration = Duration::from_millis(100);

/// Produces `GuardConditions` and `PolicyEvidenceSnapshot` from live
/// memory telemetry and a per-instance failure counter.
///
/// Memory-pressure probes are cached for
/// [`SIGNAL_BUS_CACHE_TTL`] to amortize subprocess cost over
/// rapid-fire calls (e.g. one `check_guard_before_node` per graph
/// node). Other signals (`recent_failures`, `latency_spike`) are
/// sourced fresh on every call — they are cheap memory reads of
/// internal state and do not benefit from caching; caching them
/// would delay reaction to injected/recorded events.
///
/// Unlike earlier milestones, the bus carries state: a
/// [`FailureCounter`] and [`LatencyMonitor`] wrapped in `Arc`,
/// accessible via their respective accessors so callers can record
/// events from anywhere in the engine. The cache state lives as
/// interior mutability (`Mutex`) so the bus's public methods stay
/// `&self` and `Arc<SignalBus>` can be shared across threads.
pub struct SignalBus {
    failure_counter: Arc<FailureCounter>,
    latency_monitor: Arc<LatencyMonitor>,
    /// M3-e.6: CPU probe behind a trait object so tests can inject
    /// deterministic fakes via [`SignalBus::with_cpu_probe`]. `None`
    /// when the production probe could not be constructed (rare —
    /// only happens if the platform cannot resolve the current PID);
    /// in that case the CPU fields on `GuardConditions` are
    /// unconditionally `None` and the reaction path falls back to
    /// memory-only decisions.
    cpu_probe: Option<Arc<dyn CpuProbeApi>>,
    /// Cached `(ProbeValues, captured_at)`. Populated whenever the
    /// probe path runs (a cache miss); consulted first on every call
    /// and refreshed only when the entry is absent or older than
    /// [`SIGNAL_BUS_CACHE_TTL`]. Probe failures on memory are not
    /// cached (fail-open for the aggregate signal). CPU probe
    /// failures are recorded as `None` inside the cached entry,
    /// because a failed CPU probe should not discard an otherwise
    /// valid memory probe.
    probe_cache: Mutex<Option<(ProbeValues, Instant)>>,
    /// Number of real probe invocations served so far. Incremented
    /// exclusively when a cache miss triggers the memory probes;
    /// cache hits leave this count unchanged. CPU probe calls do not
    /// carry a separate counter — they run within the same cache
    /// miss as the memory probe and their cost is amortized by the
    /// same TTL. Exposed for tests and operational telemetry via
    /// [`SignalBus::probe_calls_count`].
    probe_calls_count: AtomicU64,
}

/// Bundle of probe readings captured on a single cache-miss cycle.
/// Keeps memory and CPU aligned in time so a downstream consumer
/// reads a self-consistent snapshot.
#[derive(Debug, Clone, Copy)]
struct ProbeValues {
    memory_pressure: f32,
    /// CPU readings, or `None` if the probe was unavailable or
    /// returned an error on this cycle. Memory is still usable in
    /// that case — hence the independent `Option`.
    cpu_total: Option<f32>,
    cpu_self: Option<f32>,
}

impl SignalBus {
    /// Build a bus with the production CPU probe attached.
    ///
    /// **Warmup cost**: instantiating [`CpuProbe::new`] pays a
    /// one-time cost of ~200 ms to establish the sysinfo baseline
    /// (see `cpu_probe` module docs). Call this during engine bring-
    /// up, not on the hot path. If the CPU probe cannot be
    /// constructed (rare; platform cannot resolve the current PID),
    /// the bus is still built — memory-pressure signals work as
    /// before, and CPU fields on `GuardConditions` are `None`.
    pub fn new() -> Self {
        let cpu_probe: Option<Arc<dyn CpuProbeApi>> = match CpuProbe::new() {
            Ok(probe) => Some(Arc::new(probe)),
            Err(_) => None,
        };
        Self::with_cpu_probe_opt(cpu_probe)
    }

    /// Test / advanced constructor: build a bus with a caller-
    /// provided CPU probe. Used by integration tests to inject a
    /// deterministic fake that implements [`CpuProbeApi`] and returns
    /// canned snapshots, so the reaction path can be exercised
    /// without depending on real host CPU state.
    pub fn with_cpu_probe(cpu_probe: Arc<dyn CpuProbeApi>) -> Self {
        Self::with_cpu_probe_opt(Some(cpu_probe))
    }

    fn with_cpu_probe_opt(cpu_probe: Option<Arc<dyn CpuProbeApi>>) -> Self {
        Self {
            failure_counter: Arc::new(FailureCounter::new()),
            latency_monitor: Arc::new(LatencyMonitor::new()),
            cpu_probe,
            probe_cache: Mutex::new(None),
            probe_calls_count: AtomicU64::new(0),
        }
    }

    /// Number of actual probe invocations (subprocess calls to
    /// `nvidia-smi` plus the RAM probe) served by this bus since
    /// creation. Cache hits are not counted. Useful for testing cache
    /// behavior and for telemetry in production.
    pub fn probe_calls_count(&self) -> u64 {
        self.probe_calls_count.load(Ordering::Relaxed)
    }

    /// Returns a shared handle to the internal failure counter. Callers
    /// clone it (cheap `Arc::clone`) to record failures from multiple
    /// sites while the bus keeps reading the same counter.
    pub fn failure_counter(&self) -> Arc<FailureCounter> {
        Arc::clone(&self.failure_counter)
    }

    /// Returns a shared handle to the internal latency monitor. Callers
    /// clone it to record latency measurements from multiple sites.
    pub fn latency_monitor(&self) -> Arc<LatencyMonitor> {
        Arc::clone(&self.latency_monitor)
    }

    /// Collects current runtime conditions from live probes, the
    /// failure counter, and the latency monitor.
    ///
    /// All four fields are sourced from real telemetry:
    /// - `memory_pressure`: `max(vram_pressure, ram_pressure)` in [0.0, 1.0],
    ///   where tier pressure is `1 - available/total`.
    /// - `pre_oom_signal`: true iff `memory_pressure > 0.9`.
    /// - `recent_failures`: count of failures recorded within the
    ///   failure counter's window (60s by default).
    /// - `latency_spike`: true iff the latency monitor has seen a
    ///   spike within its recency window (30s by default).
    ///
    /// Returns `None` if any memory probe fails (fail-fast).
    ///
    /// Constructs pre_oom_signal as true when any memory tier exceeds 90%
    /// utilization. Note: 0.9 is a hardcoded placeholder for v19. In
    /// production, this threshold should be parameterizable by the caller
    /// — 0.9 is likely too late to react before OOM under fast memory
    /// growth patterns. Parameterization will be introduced in a later
    /// APX version.
    pub fn collect_guard_conditions(&self) -> Option<GuardConditions> {
        let probes = self.collect_probes()?;
        let pre_oom_signal = probes.memory_pressure > 0.9;

        let mut conditions = GuardConditions::new(
            probes.memory_pressure,
            self.failure_counter.recent_count(),
            self.latency_monitor.has_recent_spike(),
            pre_oom_signal,
        );
        // Populate CPU fields only when both readings are available.
        // A partial reading (e.g. `total` present but `self` missing)
        // is suppressed entirely — the skip logic in
        // `check_guard_before_node` needs both or neither, and a
        // half-populated struct would invite subtle bugs at the
        // decision site.
        if let (Some(total), Some(self_)) = (probes.cpu_total, probes.cpu_self) {
            conditions = conditions.with_cpu_pressure(total, self_);
        }
        Some(conditions)
    }

    /// Collects current policy evidence from live memory telemetry
    /// and the failure counter.
    ///
    /// Currently produces 4 of the 5 PolicySignalKind variants defined in v15:
    /// - `HighMemoryPressure` (always, score = memory_pressure)
    /// - `PreOomSignal` (only when memory_pressure > 0.9)
    /// - `RecentRecovery` (score = 1.0 when the last failure is at least
    ///   10 seconds but less than 60 seconds old — i.e. within a
    ///   "just recovered" window)
    /// - `StableLatency` (score = 1.0 when the latency monitor reports
    ///   no recent spike and enough samples for a reliable baseline)
    ///
    /// Signals not produced by this method:
    /// - `FragmentationWarning`: producing this reliably requires
    ///   observation into the memory allocator's internal state,
    ///   which is not reliably exposed by public OS or vendor APIs.
    ///   External proxies (reserved memory, per-process attribution)
    ///   would be semantically misleading because they measure driver
    ///   overhead or accounting gaps rather than actual fragmentation.
    ///   Deferred pending a dedicated allocator instrumentation layer,
    ///   likely in APX v20+ when the engine exposes its own GPU memory
    ///   allocator.
    ///
    /// Returns `None` only if a memory probe fails. If probes succeed but
    /// no signals qualify (e.g., memory_pressure below all thresholds
    /// that gate conditional signals), returns `Some(snapshot)` with an
    /// empty or partial signal list. The distinction matters: `None`
    /// means "we don't know"; `Some(empty)` means "we checked and there
    /// is nothing to report".
    pub fn collect_policy_evidence(&self) -> Option<PolicyEvidenceSnapshot> {
        let memory_pressure = self.collect_probes()?.memory_pressure;

        let mut signals: Vec<PolicySignal> = Vec::new();

        signals.push(PolicySignal {
            kind: PolicySignalKind::HighMemoryPressure,
            score: memory_pressure,
        });

        if memory_pressure > 0.9 {
            signals.push(PolicySignal {
                kind: PolicySignalKind::PreOomSignal,
                score: memory_pressure,
            });
        }

        if let Some(elapsed) = self.failure_counter.time_since_last_failure() {
            if elapsed >= Duration::from_secs(10) && elapsed < Duration::from_secs(60) {
                signals.push(PolicySignal {
                    kind: PolicySignalKind::RecentRecovery,
                    score: 1.0,
                });
            }
        }

        if self.latency_monitor.has_stable_latency() {
            signals.push(PolicySignal {
                kind: PolicySignalKind::StableLatency,
                score: 1.0,
            });
        }

        Some(PolicyEvidenceSnapshot::new(signals))
    }

    /// Internal: reads memory tiers and the CPU probe, returning a
    /// `ProbeValues` bundle aligned in time. Single source of truth
    /// for both [`collect_guard_conditions`] and
    /// [`collect_policy_evidence`].
    ///
    /// Cache behavior (unchanged from M3-e.4):
    /// - Cache hit within [`SIGNAL_BUS_CACHE_TTL`]: returns the stored
    ///   `ProbeValues` in O(1), leaves `probe_calls_count`
    ///   **unchanged**. This is what gives the monotonicity property
    ///   the caching tests rely on.
    /// - Cache miss (entry absent or stale): runs the real probes,
    ///   increments `probe_calls_count` by exactly 1, and on success
    ///   stores the fresh bundle.
    ///
    /// Failure semantics:
    /// - Memory probe failure: the whole call returns `None` (fail-
    ///   open for `collect_guard_conditions`). No cache write.
    /// - CPU probe failure: the call still returns `Some(...)` with
    ///   `cpu_total`/`cpu_self` as `None`. The skip logic downstream
    ///   treats absent CPU signals as "unknown" and does not veto
    ///   migration.
    fn collect_probes(&self) -> Option<ProbeValues> {
        // Cache lookup. A stale or absent entry falls through to the
        // probe path below.
        if let Ok(guard) = self.probe_cache.lock() {
            if let Some((values, captured_at)) = guard.as_ref() {
                if captured_at.elapsed() < SIGNAL_BUS_CACHE_TTL {
                    return Some(*values);
                }
            }
        }

        // Cache miss: run the real probes. Increment the counter
        // exactly once per miss, regardless of whether the probes
        // succeed — the counter tracks "probe attempts", which is
        // the quantity callers want for cost accounting.
        self.probe_calls_count.fetch_add(1, Ordering::Relaxed);

        let vram_snap = vram_probe::read_nvidia_vram_snapshot().ok()?;
        let ram_snap = ram_probe::read_system_ram_snapshot().ok()?;

        if vram_snap.total_bytes == 0 || ram_snap.total_bytes == 0 {
            return None;
        }

        let vram_pressure =
            1.0 - (vram_snap.free_bytes as f32 / vram_snap.total_bytes as f32);
        let ram_pressure =
            1.0 - (ram_snap.available_bytes as f32 / ram_snap.total_bytes as f32);
        let memory_pressure = vram_pressure.max(ram_pressure);

        // CPU probe runs independently. Its success or failure does
        // not affect the memory path — a failed CPU read only nulls
        // out the CPU fields.
        let (cpu_total, cpu_self) = match self.cpu_probe.as_ref() {
            Some(probe) => match probe.snapshot() {
                Ok(snap) => (Some(snap.total_fraction), Some(snap.self_fraction)),
                Err(_) => (None, None),
            },
            None => (None, None),
        };

        let values = ProbeValues {
            memory_pressure,
            cpu_total,
            cpu_self,
        };

        // Store the fresh value. Only reachable when both memory
        // probes succeeded and totals were non-zero, so the cache
        // never records a degraded memory result. If the lock is
        // poisoned we simply skip the update — the value returned
        // to the caller is still correct; the next call will re-
        // probe.
        if let Ok(mut guard) = self.probe_cache.lock() {
            *guard = Some((values, Instant::now()));
        }

        Some(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bus() -> SignalBus {
        SignalBus::new()
    }

    #[test]
    fn test_collect_returns_some_on_nvidia_host() {
        let bus = make_bus();
        match bus.collect_guard_conditions() {
            Some(_) => {}
            None => eprintln!(
                "SKIPPED: memory probe unavailable \
                 (test_collect_returns_some_on_nvidia_host)"
            ),
        }
    }

    #[test]
    fn test_memory_pressure_in_valid_range() {
        let bus = make_bus();
        let Some(c) = bus.collect_guard_conditions() else {
            eprintln!("SKIPPED: probe unavailable (test_memory_pressure_in_valid_range)");
            return;
        };
        assert!(
            c.memory_pressure >= 0.0,
            "pressure must be non-negative, got {}",
            c.memory_pressure
        );
        assert!(
            c.memory_pressure <= 1.0,
            "pressure must be <= 1.0, got {}",
            c.memory_pressure
        );
    }

    #[test]
    fn test_placeholder_fields_are_defaults() {
        let bus = make_bus();
        let Some(c) = bus.collect_guard_conditions() else {
            eprintln!("SKIPPED: probe unavailable (test_placeholder_fields_are_defaults)");
            return;
        };
        assert_eq!(c.recent_failures, 0);
        assert_eq!(c.latency_spike, false);
    }

    #[test]
    fn test_pre_oom_signal_consistency() {
        let bus = make_bus();
        let Some(c) = bus.collect_guard_conditions() else {
            eprintln!("SKIPPED: probe unavailable (test_pre_oom_signal_consistency)");
            return;
        };
        assert_eq!(c.pre_oom_signal, c.memory_pressure > 0.9);
    }

    #[test]
    fn test_policy_evidence_returns_some_on_nvidia_host() {
        let bus = make_bus();
        match bus.collect_policy_evidence() {
            Some(_) => {}
            None => eprintln!(
                "SKIPPED: probe unavailable \
                 (test_policy_evidence_returns_some_on_nvidia_host)"
            ),
        }
    }

    #[test]
    fn test_policy_evidence_always_has_high_memory_pressure() {
        let bus = make_bus();
        let Some(ev) = bus.collect_policy_evidence() else {
            eprintln!(
                "SKIPPED: probe unavailable \
                 (test_policy_evidence_always_has_high_memory_pressure)"
            );
            return;
        };
        let has_hmp = ev
            .all_signals()
            .iter()
            .any(|s| s.kind == PolicySignalKind::HighMemoryPressure);
        assert!(
            has_hmp,
            "HighMemoryPressure signal must always be present when probes succeed"
        );
    }

    #[test]
    fn test_policy_evidence_pre_oom_only_above_threshold() {
        let bus = make_bus();
        let Some(ev) = bus.collect_policy_evidence() else {
            eprintln!(
                "SKIPPED: probe unavailable \
                 (test_policy_evidence_pre_oom_only_above_threshold)"
            );
            return;
        };
        // If PreOomSignal is present, its score must be > 0.9 (by construction).
        if let Some(pre_oom) = ev
            .all_signals()
            .iter()
            .find(|s| s.kind == PolicySignalKind::PreOomSignal)
        {
            assert!(
                pre_oom.score > 0.9,
                "PreOomSignal present but score {} is not > 0.9",
                pre_oom.score
            );
        }
    }

    #[test]
    fn test_policy_evidence_all_scores_normalized() {
        let bus = make_bus();
        let Some(ev) = bus.collect_policy_evidence() else {
            eprintln!(
                "SKIPPED: probe unavailable \
                 (test_policy_evidence_all_scores_normalized)"
            );
            return;
        };
        assert!(
            ev.is_normalized(),
            "all signal scores must be in [0.0, 1.0]"
        );
    }

    #[test]
    fn test_recent_failures_starts_at_zero() {
        let bus = make_bus();
        let Some(c) = bus.collect_guard_conditions() else {
            eprintln!("SKIPPED: probe unavailable (test_recent_failures_starts_at_zero)");
            return;
        };
        assert_eq!(c.recent_failures, 0);
    }

    #[test]
    fn test_recent_failures_reflects_recorded() {
        let bus = make_bus();
        let counter = bus.failure_counter();
        for _ in 0..3 {
            counter.record_failure();
        }
        let Some(c) = bus.collect_guard_conditions() else {
            eprintln!("SKIPPED: probe unavailable (test_recent_failures_reflects_recorded)");
            return;
        };
        assert_eq!(c.recent_failures, 3);
    }

    #[test]
    fn test_recent_recovery_absent_initially() {
        let bus = make_bus();
        let Some(ev) = bus.collect_policy_evidence() else {
            eprintln!("SKIPPED: probe unavailable (test_recent_recovery_absent_initially)");
            return;
        };
        let has_recovery = ev
            .all_signals()
            .iter()
            .any(|s| s.kind == PolicySignalKind::RecentRecovery);
        assert!(
            !has_recovery,
            "RecentRecovery must not be present when no failure has been recorded"
        );
    }

    #[test]
    fn test_recent_recovery_absent_right_after_failure() {
        let bus = make_bus();
        bus.failure_counter().record_failure();
        // elapsed < 10s by construction (we just recorded)
        let Some(ev) = bus.collect_policy_evidence() else {
            eprintln!(
                "SKIPPED: probe unavailable \
                 (test_recent_recovery_absent_right_after_failure)"
            );
            return;
        };
        let has_recovery = ev
            .all_signals()
            .iter()
            .any(|s| s.kind == PolicySignalKind::RecentRecovery);
        assert!(
            !has_recovery,
            "RecentRecovery must not fire immediately after a failure \
             (requires elapsed >= 10s)"
        );
    }

    #[test]
    fn test_latency_spike_false_initially() {
        let bus = make_bus();
        let Some(c) = bus.collect_guard_conditions() else {
            eprintln!("SKIPPED: probe unavailable (test_latency_spike_false_initially)");
            return;
        };
        assert_eq!(c.latency_spike, false);
    }

    #[test]
    fn test_latency_spike_true_after_spike() {
        let bus = make_bus();
        let monitor = bus.latency_monitor();
        for _ in 0..15 {
            monitor.record_latency(Duration::from_millis(50));
        }
        monitor.record_latency(Duration::from_millis(500));

        let Some(c) = bus.collect_guard_conditions() else {
            eprintln!("SKIPPED: probe unavailable (test_latency_spike_true_after_spike)");
            return;
        };
        assert_eq!(c.latency_spike, true);
    }

    #[test]
    fn test_stable_latency_absent_initially() {
        let bus = make_bus();
        let Some(ev) = bus.collect_policy_evidence() else {
            eprintln!("SKIPPED: probe unavailable (test_stable_latency_absent_initially)");
            return;
        };
        let has_stable = ev
            .all_signals()
            .iter()
            .any(|s| s.kind == PolicySignalKind::StableLatency);
        assert!(
            !has_stable,
            "StableLatency must not be present without enough samples"
        );
    }

    #[test]
    fn test_stable_latency_present_after_enough_stable_samples() {
        let bus = make_bus();
        let monitor = bus.latency_monitor();
        for _ in 0..15 {
            monitor.record_latency(Duration::from_millis(50));
        }

        let Some(ev) = bus.collect_policy_evidence() else {
            eprintln!(
                "SKIPPED: probe unavailable \
                 (test_stable_latency_present_after_enough_stable_samples)"
            );
            return;
        };
        let has_stable = ev
            .all_signals()
            .iter()
            .any(|s| s.kind == PolicySignalKind::StableLatency);
        assert!(
            has_stable,
            "StableLatency must be present with enough stable samples"
        );
    }
}
