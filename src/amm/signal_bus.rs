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

use std::sync::Arc;
use std::time::Duration;

use crate::amm::failure_counter::FailureCounter;
use crate::amm::latency_monitor::LatencyMonitor;
use crate::amm::ram_probe;
use crate::amm::vram_probe;
use crate::v15::policy::evidence::signals::{PolicySignal, PolicySignalKind};
use crate::v15::policy::evidence::snapshot::PolicyEvidenceSnapshot;
use crate::v16::guards::guard_conditions::GuardConditions;

/// Produces `GuardConditions` and `PolicyEvidenceSnapshot` from live
/// memory telemetry and a per-instance failure counter.
///
/// Pull-only for memory telemetry: each call to `collect_*` issues
/// fresh probe reads with no caching.
///
/// Unlike earlier milestones, the bus now carries state: a
/// [`FailureCounter`] wrapped in `Arc`, accessible via
/// [`failure_counter`](Self::failure_counter) so callers can record
/// failure events from anywhere in the engine.
pub struct SignalBus {
    failure_counter: Arc<FailureCounter>,
    latency_monitor: Arc<LatencyMonitor>,
}

impl SignalBus {
    pub fn new() -> Self {
        Self {
            failure_counter: Arc::new(FailureCounter::new()),
            latency_monitor: Arc::new(LatencyMonitor::new()),
        }
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
        let memory_pressure = self.collect_memory_pressure()?;
        let pre_oom_signal = memory_pressure > 0.9;

        Some(GuardConditions::new(
            memory_pressure,
            self.failure_counter.recent_count(),
            self.latency_monitor.has_recent_spike(),
            pre_oom_signal,
        ))
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
        let memory_pressure = self.collect_memory_pressure()?;

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

    /// Internal: reads both tier snapshots and returns the aggregate
    /// memory pressure (max across tiers) in [0.0, 1.0]. Single source
    /// of truth for both [`collect_guard_conditions`] and
    /// [`collect_policy_evidence`].
    ///
    /// Returns `None` on probe failure or zero totals (which would
    /// produce NaN).
    fn collect_memory_pressure(&self) -> Option<f32> {
        let vram_snap = vram_probe::read_nvidia_vram_snapshot().ok()?;
        let ram_snap = ram_probe::read_system_ram_snapshot().ok()?;

        if vram_snap.total_bytes == 0 || ram_snap.total_bytes == 0 {
            return None;
        }

        let vram_pressure =
            1.0 - (vram_snap.free_bytes as f32 / vram_snap.total_bytes as f32);
        let ram_pressure =
            1.0 - (ram_snap.available_bytes as f32 / ram_snap.total_bytes as f32);
        Some(vram_pressure.max(ram_pressure))
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
