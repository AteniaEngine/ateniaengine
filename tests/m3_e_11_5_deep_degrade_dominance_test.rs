//! APX v20 M3-e.11.5 — dominance tests for `GuardAction::DeepDegrade`
//! inside `GuardManager::evaluate`.
//!
//! The verdict combination rule landed in M3-e.11.5:
//!
//!   Abort > DeepDegrade > Degrade > Continue
//!
//! Where each level dominates the ones to its right. These tests
//! only touch the manager — no graph, no reactive context, no
//! migrations. Promotion logic is covered separately in
//! `m3_e_11_5_promotion_test.rs`.
//!
//! Each test builds a `GuardManager` with several
//! `FixedActionGuard` instances (a trivial guard that always
//! emits a caller-specified action) and asserts on the combined
//! verdict.
//!
//! Legal contract is kept permissive so the post-aggregation
//! legality check in `evaluate` never changes the result —
//! M3-e.11.5 is about the aggregation ordering, not about
//! contract invariants.

use atenia_engine::v15::policy::types::DecisionBias;
use atenia_engine::v16::contract::constraints::{Constraints, RuntimeState};
use atenia_engine::v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use atenia_engine::v16::guards::execution_guard::ExecutionGuard;
use atenia_engine::v16::guards::guard_action::GuardAction;
use atenia_engine::v16::guards::guard_conditions::GuardConditions;
use atenia_engine::v16::guards::guard_manager::GuardManager;

/// Trivial guard that always returns a single, pre-configured
/// action. Used purely to exercise `GuardManager::evaluate`'s
/// aggregation rules.
struct FixedActionGuard {
    label: &'static str,
    action: GuardAction,
}

impl ExecutionGuard for FixedActionGuard {
    fn name(&self) -> &'static str {
        self.label
    }
    fn evaluate(
        &self,
        _contract: &ExecutionContract,
        _conditions: &GuardConditions,
    ) -> GuardAction {
        self.action.clone()
    }
}

fn permissive_contract() -> ExecutionContract {
    ExecutionContract {
        bias: DecisionBias {
            risk_weight: 0.3,
            latency_weight: 0.4,
            stability_weight: 0.5,
            memory_pressure_weight: 0.5,
            offload_cost_weight: 0.4,
        },
        runtime_snapshot: RuntimeState {
            memory_headroom: 0.8,
            is_stable: true,
            recent_recovery: false,
            offload_supported: true,
        },
        allowed_backends: vec![ExecutionBackend::Local],
        forbidden_backends: vec![],
        max_aggressiveness: 0.5,
        require_fallback: false,
        require_stability: false,
        constraints: Constraints { items: vec![] },
    }
}

fn clean_conditions() -> GuardConditions {
    GuardConditions::new(0.1, 0, false, false)
}

fn manager_with(actions: Vec<GuardAction>) -> GuardManager {
    let guards: Vec<Box<dyn ExecutionGuard>> = actions
        .into_iter()
        .enumerate()
        .map(|(i, a)| {
            // Labels are static strings baked at compile time; use
            // Box::leak to produce a single &'static str per label
            // without needing a fancier lifetime scheme — tests
            // only, cost is negligible.
            let label: &'static str = Box::leak(format!("fixed_guard_{}", i).into_boxed_str());
            Box::new(FixedActionGuard { label, action: a }) as Box<dyn ExecutionGuard>
        })
        .collect();
    GuardManager::new(guards)
}

#[test]
fn test_deep_degrade_dominates_degrade() {
    let mgr = manager_with(vec![GuardAction::Degrade, GuardAction::DeepDegrade]);
    let verdict = mgr
        .evaluate(&permissive_contract(), &clean_conditions())
        .expect("contract is permissive, no legal-check failure");
    assert_eq!(verdict, GuardAction::DeepDegrade);
}

#[test]
fn test_deep_degrade_dominates_continue() {
    let mgr = manager_with(vec![GuardAction::Continue, GuardAction::DeepDegrade]);
    let verdict = mgr
        .evaluate(&permissive_contract(), &clean_conditions())
        .expect("ok");
    assert_eq!(verdict, GuardAction::DeepDegrade);
}

#[test]
fn test_abort_dominates_deep_degrade() {
    // Abort must win even when DeepDegrade is in the mix.
    let mgr = manager_with(vec![GuardAction::DeepDegrade, GuardAction::Abort]);
    let verdict = mgr
        .evaluate(&permissive_contract(), &clean_conditions())
        .expect("ok");
    assert_eq!(verdict, GuardAction::Abort);

    // Order should not matter.
    let mgr2 = manager_with(vec![GuardAction::Abort, GuardAction::DeepDegrade]);
    let verdict2 = mgr2
        .evaluate(&permissive_contract(), &clean_conditions())
        .expect("ok");
    assert_eq!(verdict2, GuardAction::Abort);
}

#[test]
fn test_degrade_still_dominates_continue() {
    // Regression: the pre-existing rule (Degrade > Continue) must
    // still hold after the aggregation refactor.
    let mgr = manager_with(vec![GuardAction::Continue, GuardAction::Degrade]);
    let verdict = mgr
        .evaluate(&permissive_contract(), &clean_conditions())
        .expect("ok");
    assert_eq!(verdict, GuardAction::Degrade);
}

#[test]
fn test_all_continue_yields_continue() {
    let mgr = manager_with(vec![
        GuardAction::Continue,
        GuardAction::Continue,
        GuardAction::Continue,
    ]);
    let verdict = mgr
        .evaluate(&permissive_contract(), &clean_conditions())
        .expect("ok");
    assert_eq!(verdict, GuardAction::Continue);
}

#[test]
fn test_multiple_guards_full_stack_ordering() {
    // Every severity present simultaneously — Abort wins.
    let mgr = manager_with(vec![
        GuardAction::Continue,
        GuardAction::Degrade,
        GuardAction::DeepDegrade,
        GuardAction::Abort,
    ]);
    let verdict = mgr
        .evaluate(&permissive_contract(), &clean_conditions())
        .expect("ok");
    assert_eq!(verdict, GuardAction::Abort);

    // Drop Abort — DeepDegrade wins.
    let mgr_no_abort = manager_with(vec![
        GuardAction::Continue,
        GuardAction::Degrade,
        GuardAction::DeepDegrade,
    ]);
    let verdict_no_abort = mgr_no_abort
        .evaluate(&permissive_contract(), &clean_conditions())
        .expect("ok");
    assert_eq!(verdict_no_abort, GuardAction::DeepDegrade);

    // Drop DeepDegrade — Degrade wins.
    let mgr_no_deep = manager_with(vec![GuardAction::Continue, GuardAction::Degrade]);
    let verdict_no_deep = mgr_no_deep
        .evaluate(&permissive_contract(), &clean_conditions())
        .expect("ok");
    assert_eq!(verdict_no_deep, GuardAction::Degrade);
}

#[test]
fn test_deep_degrade_single_guard_surfaces_as_verdict() {
    // A lone DeepDegrade guard surfaces directly as the verdict —
    // dominance rule does not require another guard in the mix.
    let mgr = manager_with(vec![GuardAction::DeepDegrade]);
    let verdict = mgr
        .evaluate(&permissive_contract(), &clean_conditions())
        .expect("ok");
    assert_eq!(verdict, GuardAction::DeepDegrade);
}
