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
        }
    }

    /// Number of times `GuardAction::Degrade` was processed by this
    /// context, whether the resulting migration succeeded or failed.
    /// Useful for monitoring how often reaction is triggered in a
    /// given execution run.
    pub fn degrade_events_count(&self) -> u64 {
        self.degrade_events_count.load(Ordering::Relaxed)
    }

    /// Record that a Degrade verdict was processed. Called from the
    /// graph's guard-handling site; not part of the public API because
    /// external code has no business incrementing the counter.
    pub(crate) fn record_degrade_event(&self) {
        self.degrade_events_count.fetch_add(1, Ordering::Relaxed);
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
