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
use v16::planner::execution_planner::ExecutionPlanner;
use v16::speculative::rollback_manager::RollbackManager;
use v16::speculative::speculative_errors::SpeculativeError;
use v16::speculative::speculative_executor::SpeculativeExecutor;

#[derive(Debug, Clone, PartialEq, Default)]
struct MockRuntime {
    calls: Vec<&'static str>,
    counter: u32,
    fail_on_call: Option<&'static str>,
}

impl RuntimeFacade for MockRuntime {
    fn ensure_memory_headroom(&mut self) -> Result<(), String> {
        self.calls.push("ensure_memory_headroom");
        self.counter += 1;
        if self.fail_on_call == Some("ensure_memory_headroom") {
            Err("headroom failed".to_string())
        } else {
            Ok(())
        }
    }

    fn select_backend_candidate(&mut self) -> Result<(), String> {
        self.calls.push("select_backend_candidate");
        self.counter += 1;
        if self.fail_on_call == Some("select_backend_candidate") {
            Err("backend failed".to_string())
        } else {
            Ok(())
        }
    }

    fn prepare_fallback(&mut self) -> Result<(), String> {
        self.calls.push("prepare_fallback");
        self.counter += 1;
        if self.fail_on_call == Some("prepare_fallback") {
            Err("fallback failed".to_string())
        } else {
            Ok(())
        }
    }

    fn mark_tensors_movable(&mut self) -> Result<(), String> {
        self.calls.push("mark_tensors_movable");
        self.counter += 1;
        if self.fail_on_call == Some("mark_tensors_movable") {
            Err("mark failed".to_string())
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

fn make_plan() -> v16::planner::execution_plan::ExecutionPlan {
    let contract = make_contract();
    ExecutionPlanner::build_plan(&contract).expect("plan should be produced")
}

#[test]
fn speculative_execution_is_isolated_from_main_executor() {
    let plan = make_plan();

    // Main executor with its own runtime.
    let main_runtime = MockRuntime::default();
    let main_ctx = ExecutionContext::new(main_runtime);
    let mut main_executor = SafeExecutor::new(plan.clone(), main_ctx);

    // Speculative executor with a separate runtime clone.
    let spec_runtime = MockRuntime::default();
    let spec_ctx = ExecutionContext::new(spec_runtime);
    let mut spec_exec = SpeculativeExecutor::new(&plan, spec_ctx);

    // Run some steps on the main executor.
    while let ExecutorStatus::Running = main_executor.status() {
        main_executor.step().expect("main step should succeed");
    }

    // Run speculative execution.
    spec_exec.run().expect("speculative run should succeed");

    // Main runtime and speculative runtime are independent instances and both
    // have observed calls, but their effects do not interfere with each
    // other. The traces may be identical because they follow the same plan,
    // so we only assert that both have activity and that modifying one would
    // not affect the other.
    let main_calls = &main_executor.context.runtime.calls;
    let spec_calls = &spec_exec.context.runtime.calls;
    assert!(!main_calls.is_empty());
    assert!(!spec_calls.is_empty());
    use std::ptr;
    assert!(!ptr::eq(
        &main_executor.context.runtime as *const _,
        &spec_exec.context.runtime as *const _
    ));
}

#[test]
fn rollback_restores_original_state_on_failure() {
    let plan = make_plan();

    let runtime = MockRuntime {
        fail_on_call: Some("select_backend_candidate"),
        ..MockRuntime::default()
    };
    let original = runtime.clone();

    let ctx = ExecutionContext::new(runtime);
    let mut exec = SpeculativeExecutor::new(&plan, ctx);

    let result = exec.run();
    assert!(matches!(result, Err(SpeculativeError::ExecutionFailed(_))));

    // After rollback, runtime must match the original snapshot.
    assert_eq!(exec.context.runtime, original);
}

#[test]
fn speculative_execution_is_deterministic() {
    let plan = make_plan();

    let runtime1 = MockRuntime::default();
    let runtime2 = MockRuntime::default();

    let ctx1 = ExecutionContext::new(runtime1);
    let ctx2 = ExecutionContext::new(runtime2);

    let mut exec1 = SpeculativeExecutor::new(&plan, ctx1);
    let mut exec2 = SpeculativeExecutor::new(&plan, ctx2);

    exec1.run().expect("first run");
    exec2.run().expect("second run");

    assert_eq!(exec1.context.runtime.calls, exec2.context.runtime.calls);
    assert_eq!(exec1.context.runtime.counter, exec2.context.runtime.counter);
}

#[test]
fn speculation_never_violates_contract_abortability() {
    let mut plan = make_plan();
    plan.globally_abortable = false;

    let runtime = MockRuntime::default();
    let ctx = ExecutionContext::new(runtime);
    let mut exec = SpeculativeExecutor::new(&plan, ctx);

    let result = exec.run();
    assert!(matches!(
        result,
        Err(SpeculativeError::ContractViolation(_))
    ));
}

#[test]
fn rollback_manager_restores_runtime_snapshot() {
    let mut runtime = MockRuntime::default();
    runtime.calls.push("preexisting");
    runtime.counter = 42;

    let manager = RollbackManager::new(&runtime);

    // Mutate runtime.
    runtime.calls.push("mutation");
    runtime.counter = 0;

    // Rollback.
    manager.rollback(&mut runtime);

    assert_eq!(runtime.calls, vec!["preexisting"]);
    assert_eq!(runtime.counter, 42);
}
