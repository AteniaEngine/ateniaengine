#![allow(dead_code)]

#[path = "../src/v16/mod.rs"]
mod v16;

#[path = "../src/v15/mod.rs"]
mod v15;

use v15::policy::types::DecisionBias;
use v16::contract::constraints::{Constraint, ConstraintKind, Constraints, RuntimeState};
use v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use v16::executor::execution_context::{ExecutionContext, RuntimeFacade};
use v16::executor::executor_state::ExecutorStatus;
use v16::executor::safe_executor::SafeExecutor;
use v16::planner::execution_plan::ExecutionPlan;
use v16::planner::execution_planner::ExecutionPlanner;

#[derive(Debug, Default, Clone, PartialEq)]
struct MockRuntime {
    calls: Vec<&'static str>,
    fail_on_call: Option<&'static str>,
}

impl RuntimeFacade for MockRuntime {
    fn ensure_memory_headroom(&mut self) -> Result<(), String> {
        self.calls.push("ensure_memory_headroom");
        if self.fail_on_call == Some("ensure_memory_headroom") {
            Err("headroom check failed".to_string())
        } else {
            Ok(())
        }
    }

    fn select_backend_candidate(&mut self) -> Result<(), String> {
        self.calls.push("select_backend_candidate");
        if self.fail_on_call == Some("select_backend_candidate") {
            Err("backend selection failed".to_string())
        } else {
            Ok(())
        }
    }

    fn prepare_fallback(&mut self) -> Result<(), String> {
        self.calls.push("prepare_fallback");
        if self.fail_on_call == Some("prepare_fallback") {
            Err("fallback preparation failed".to_string())
        } else {
            Ok(())
        }
    }

    fn mark_tensors_movable(&mut self) -> Result<(), String> {
        self.calls.push("mark_tensors_movable");
        if self.fail_on_call == Some("mark_tensors_movable") {
            Err("mark movable failed".to_string())
        } else {
            Ok(())
        }
    }
}

fn make_contract() -> ExecutionContract {
    let bias = DecisionBias {
        risk_weight: 0.3,
        latency_weight: 0.4,
        stability_weight: 0.8,
        memory_pressure_weight: 0.5,
        offload_cost_weight: 0.4,
    };

    let state = RuntimeState {
        memory_headroom: 0.8,
        is_stable: true,
        recent_recovery: false,
        offload_supported: true,
    };

    let constraints = Constraints {
        items: vec![
            Constraint::hard(ConstraintKind::MemoryHeadroom { min: 0.2 }),
            Constraint::hard(ConstraintKind::RequireStability),
        ],
    };

    ExecutionContract {
        bias,
        runtime_snapshot: state,
        allowed_backends: vec![ExecutionBackend::Local, ExecutionBackend::Offload],
        forbidden_backends: vec![],
        max_aggressiveness: 0.7,
        require_fallback: true,
        require_stability: true,
        constraints,
    }
}

fn make_plan() -> ExecutionPlan {
    let contract = make_contract();
    ExecutionPlanner::build_plan(&contract).expect("plan should be produced")
}

#[test]
fn executes_steps_in_order() {
    let plan = make_plan();
    let runtime = MockRuntime::default();
    let ctx = ExecutionContext::new(runtime);
    let mut executor = SafeExecutor::new(plan.clone(), ctx);

    while let ExecutorStatus::Running = executor.status() {
        executor.step().expect("step should succeed");
    }

    assert_eq!(executor.status(), ExecutorStatus::Completed);

    let calls = executor.context.runtime.calls;
    // At least one call per plan step, in the same order.
    assert!(!calls.is_empty());
    assert_eq!(calls[0], "ensure_memory_headroom");
    assert!(calls.contains(&"select_backend_candidate"));
}

#[test]
fn stops_execution_on_failed_step() {
    let plan = make_plan();
    let runtime = MockRuntime {
        fail_on_call: Some("select_backend_candidate"),
        ..MockRuntime::default()
    };
    let ctx = ExecutionContext::new(runtime);
    let mut executor = SafeExecutor::new(plan, ctx);

    // First step should succeed.
    executor.step().expect("first step should succeed");
    // Second step should fail.
    let _ = executor.step().unwrap_err();

    assert_eq!(executor.status(), ExecutorStatus::Failed);

    // No further steps should be executed.
    let calls = &executor.context.runtime.calls;
    assert!(calls.contains(&"ensure_memory_headroom"));
    assert!(calls.contains(&"select_backend_candidate"));
    assert!(!calls.contains(&"prepare_fallback"));
}

#[test]
fn abort_stops_execution_safely() {
    let plan = make_plan();
    let runtime = MockRuntime::default();
    let ctx = ExecutionContext::new(runtime);
    let mut executor = SafeExecutor::new(plan, ctx);

    // Run a single step, then abort.
    executor.step().expect("first step should succeed");
    let _ = executor.abort("test abort");

    assert_eq!(executor.status(), ExecutorStatus::Aborted);

    // Further steps must not execute.
    let _ = executor.step();
    let calls = &executor.context.runtime.calls;
    assert!(calls.contains(&"ensure_memory_headroom"));
    assert!(!calls.contains(&"select_backend_candidate"));
}

#[test]
fn no_execution_outside_plan() {
    let plan = make_plan();
    let runtime = MockRuntime::default();
    let ctx = ExecutionContext::new(runtime);
    let mut executor = SafeExecutor::new(plan, ctx);

    // Run until completion.
    while let ExecutorStatus::Running = executor.status() {
        executor.step().expect("step should succeed");
    }

    assert_eq!(executor.status(), ExecutorStatus::Completed);

    let calls = &executor.context.runtime.calls;
    // Number of calls must not exceed number of plan steps.
    assert!(calls.len() <= executor.plan.steps.len());
}

#[test]
fn executor_state_transitions_are_valid() {
    let plan = make_plan();
    let runtime = MockRuntime::default();
    let ctx = ExecutionContext::new(runtime);
    let mut executor = SafeExecutor::new(plan, ctx);

    assert_eq!(executor.status(), ExecutorStatus::Running);

    // Execute one step.
    executor.step().expect("first step should succeed");
    assert_eq!(executor.status(), ExecutorStatus::Running);

    // Abort and ensure status changes and remains stable.
    let _ = executor.abort("manual abort");
    assert_eq!(executor.status(), ExecutorStatus::Aborted);

    // Further calls to step must not change the status.
    let _ = executor.step();
    assert_eq!(executor.status(), ExecutorStatus::Aborted);
}
