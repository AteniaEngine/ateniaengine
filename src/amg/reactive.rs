//! Reactive execution layer for AMG graphs (APX v20 M2).
//!
//! Groups the three pieces a graph needs to gate execution on live
//! telemetry: a signal bus that produces `GuardConditions`, a
//! contract that declares what is legally allowed, and a guard
//! manager that combines multiple guards into a single verdict.
//!
//! A `Graph` carries an `Option<ReactiveExecutionContext>`. When set,
//! schedulers consult `Graph::check_guard_before_node` before each
//! node and abort cleanly if a guard triggers. When unset (the
//! default), execution behavior is identical to pre-M2.
//!
//! This module also defines the abort reason enum surfaced by the
//! checked execution path.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::amm::signal_bus::SignalBus;
use crate::v16::contract::execution_contract::ExecutionContract;
use crate::v16::guards::guard_conditions::GuardConditions;
use crate::v16::guards::guard_manager::GuardManager;

/// Runtime reactive execution layer attached to a `Graph`.
///
/// Cheap to attach; `signal_bus` is an `Arc` and the other two are
/// small owned structs. Future milestones (policy evaluation in M3+,
/// strategy selection in M4+) will add fields here rather than on
/// `Graph` itself.
pub struct ReactiveExecutionContext {
    pub signal_bus: Arc<SignalBus>,
    pub contract: ExecutionContract,
    pub guard_manager: GuardManager,
    /// M3-e.5: per-context counter of processed `GuardAction::Degrade`
    /// verdicts. Incremented each time the guard-handling site observes
    /// a `Degrade` action and initiates a migration, whether the
    /// migration itself succeeds or fails. Readable via
    /// [`degrade_events_count`](Self::degrade_events_count).
    pub(crate) degrade_events_count: AtomicU64,
    /// M3-e.6: per-context counter of `Degrade` verdicts that were
    /// vetoed by the CPU-availability precondition. Incremented when
    /// the guard said "migrate" but the CPU signals say another
    /// process is saturating CPU and Atenia's contribution is small,
    /// so migration would make the external pressure worse. Not a
    /// subset of `degrade_events_count` — vetoed verdicts never
    /// reached the migration site. Readable via
    /// [`degrade_vetoed_by_cpu_count`](Self::degrade_vetoed_by_cpu_count).
    pub(crate) degrade_vetoed_by_cpu_count: AtomicU64,
}

/// M3-e.6: system-wide CPU utilization threshold above which the
/// reaction path considers "the CPU is under pressure". Paired with
/// [`CPU_SELF_CONTRIBUTION_MIN`] to decide whether that pressure is
/// caused by Atenia or by something else.
pub const CPU_PRESSURE_TOTAL_THRESHOLD: f32 = 0.80;

/// M3-e.6: minimum share `self / total` that attributes CPU pressure
/// to Atenia. Below this, Atenia is considered **not responsible** —
/// some other process is loading the CPU and migrating more work to
/// the CPU would worsen the external pressure, so the Degrade arm
/// skips the migration.
pub const CPU_SELF_CONTRIBUTION_MIN: f32 = 0.50;

/// M3-e.7: format a compact fragment describing the GPU compute
/// utilization for inclusion in `[AMG Guard]` log lines. Returns
/// `" gpu_util_total=0.XX, gpu_util_self=0.XX,"` when both fields
/// are populated, or `" gpu_util=n/a,"` when either is absent.
///
/// The leading space and trailing comma let the caller inline the
/// fragment into an existing log format string without rebuilding
/// the whole message. Observability-only: the fragment never
/// changes behavior, it only adds diagnostic context to the log.
pub fn format_gpu_util_fragment(conditions: &GuardConditions) -> String {
    match (conditions.gpu_util_total, conditions.gpu_util_self) {
        (Some(total), Some(self_)) => format!(
            " gpu_util_total={:.2}, gpu_util_self={:.2},",
            total, self_
        ),
        _ => " gpu_util=n/a,".to_string(),
    }
}

/// M3-e.8: format a compact fragment describing the foreground-app
/// indicator for inclusion in `[AMG Guard]` log lines. Returns one
/// of three strings:
/// - `" foreground=atenia,"`  — the OS foreground is this process.
/// - `" foreground=other,"`   — the OS foreground is a different process.
/// - `" foreground=n/a,"`     — the probe could not determine (unsupported
///   platform, screen locked, etc.).
///
/// Same leading-space / trailing-comma conventions as the other
/// fragments. Observability-only; no behavior changes.
pub fn format_foreground_fragment(conditions: &GuardConditions) -> String {
    match conditions.foreground_is_atenia {
        Some(true) => " foreground=atenia,".to_string(),
        Some(false) => " foreground=other,".to_string(),
        None => " foreground=n/a,".to_string(),
    }
}

/// M3-e.9: format a compact fragment describing battery state for
/// inclusion in `[AMG Guard]` log lines. The two `GuardConditions`
/// fields (`on_battery` and `battery_level`) can vary independently,
/// so the fragment is built from both:
/// - `" battery=plugged_0.85,"` — plugged in at 85% charge.
/// - `" battery=plugged,"`      — plugged in, level unknown.
/// - `" battery=on_0.15,"`      — on battery at 15% charge.
/// - `" battery=on,"`           — on battery, level unknown.
/// - `" battery=0.50,"`         — level known but AC state unknown (rare driver state).
/// - `" battery=n/a,"`          — neither field present (desktop or stub platform).
///
/// Same leading-space / trailing-comma convention as the other
/// fragments. Observability-only.
pub fn format_battery_fragment(conditions: &GuardConditions) -> String {
    match (conditions.on_battery, conditions.battery_level) {
        (Some(true), Some(level)) => format!(" battery=on_{:.2},", level),
        (Some(true), None) => " battery=on,".to_string(),
        (Some(false), Some(level)) => format!(" battery=plugged_{:.2},", level),
        (Some(false), None) => " battery=plugged,".to_string(),
        (None, Some(level)) => format!(" battery={:.2},", level),
        (None, None) => " battery=n/a,".to_string(),
    }
}

/// Decide whether a `Degrade` verdict should be vetoed because the
/// CPU is saturated by *external* processes rather than by Atenia.
///
/// Returns `true` (veto, skip migration) only when **all** of the
/// following hold:
/// - Both CPU fields are populated on `conditions`. If either is
///   `None` (probe absent or probe failure), the function returns
///   `false` — fail-open: an unknown CPU state must not block a
///   reaction the memory guard has already requested.
/// - `cpu_pressure_total > `[`CPU_PRESSURE_TOTAL_THRESHOLD`]: the
///   system is genuinely under CPU pressure.
/// - `self / total < `[`CPU_SELF_CONTRIBUTION_MIN`]: Atenia's share
///   of that pressure is small, so Atenia is not the cause.
///
/// See the M3-e.6 handoff scenarios table for the decision matrix
/// this implements.
pub fn cpu_saturated_externally(conditions: &GuardConditions) -> bool {
    let (total, self_) = match (conditions.cpu_pressure_total, conditions.cpu_pressure_self) {
        (Some(t), Some(s)) => (t, s),
        _ => return false,
    };
    if total <= CPU_PRESSURE_TOTAL_THRESHOLD {
        return false;
    }
    // `total` > 0.80 here, so it cannot be zero; division is safe.
    let share = self_ / total;
    share < CPU_SELF_CONTRIBUTION_MIN
}

impl ReactiveExecutionContext {
    pub fn new(
        signal_bus: Arc<SignalBus>,
        contract: ExecutionContract,
        guard_manager: GuardManager,
    ) -> Self {
        Self {
            signal_bus,
            contract,
            guard_manager,
            degrade_events_count: AtomicU64::new(0),
            degrade_vetoed_by_cpu_count: AtomicU64::new(0),
        }
    }

    /// Number of times `GuardAction::Degrade` was processed by this
    /// context, whether the resulting migration succeeded or failed.
    /// Useful for monitoring how often reaction is triggered in a
    /// given execution run.
    pub fn degrade_events_count(&self) -> u64 {
        self.degrade_events_count.load(Ordering::Relaxed)
    }

    /// M3-e.6: number of times a `Degrade` verdict was vetoed by the
    /// CPU-availability precondition because the CPU was saturated by
    /// external processes. These verdicts never reached the migration
    /// site; they are disjoint from `degrade_events_count`.
    pub fn degrade_vetoed_by_cpu_count(&self) -> u64 {
        self.degrade_vetoed_by_cpu_count.load(Ordering::Relaxed)
    }

    /// Record that a Degrade verdict was processed. Called from the
    /// graph's guard-handling site; not part of the public API because
    /// external code has no business incrementing the counter.
    pub(crate) fn record_degrade_event(&self) {
        self.degrade_events_count.fetch_add(1, Ordering::Relaxed);
    }

    /// M3-e.6: record that a Degrade verdict was vetoed by the CPU
    /// precondition. Called from the guard-handling site before the
    /// `Degrade` arm's migration body runs.
    pub(crate) fn record_degrade_veto_by_cpu(&self) {
        self.degrade_vetoed_by_cpu_count
            .fetch_add(1, Ordering::Relaxed);
    }
}

/// Report produced by a successful `Graph::migrate_all_cuda_to_cpu`
/// call. Returned to `check_guard_before_node` so the guard-handling
/// site can log what the Degrade action actually did. Bytes freed are
/// an estimate based on `numel * size_of::<f32>()`; the actual VRAM
/// release depends on the refcount of the underlying `Arc<InnerGpuPtr>`
/// at drop time, which is not observable from the migration site.
#[derive(Debug, Clone)]
pub struct DegradeReport {
    pub tensors_migrated: usize,
    pub bytes_freed_estimate: usize,
}

impl fmt::Display for DegradeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mib = self.bytes_freed_estimate as f64 / (1024.0 * 1024.0);
        write!(
            f,
            "Degrade: migrated {} tensors, freed ~{:.2} MiB (estimate)",
            self.tensors_migrated, mib
        )
    }
}

/// Reasons why a checked execution may abort before a given node runs.
///
/// Produced only when the graph has a `reactive_context` set. When the
/// context is absent, checked execution never returns these errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionAbortReason {
    /// The combined guard verdict for the current runtime state was
    /// `Abort`. The full `GuardConditions` snapshot is included for
    /// diagnostics.
    GuardAborted {
        at_node: usize,
        conditions: GuardConditions,
    },
    /// The `GuardManager` rejected the combined guard verdict as
    /// illegal given the current `ExecutionContract` (e.g. continuing
    /// under a pre-OOM signal while stability is required). The
    /// message comes from `GuardError::IllegalAction`.
    GuardEvaluationFailed {
        at_node: usize,
        message: String,
    },
}
