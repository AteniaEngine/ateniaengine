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
        }
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
