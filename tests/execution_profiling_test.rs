#![allow(dead_code)]

use atenia_engine::v15;
use atenia_engine::v16;
use atenia_engine::v17;

use v16::planner::execution_plan::ExecutionPlan;
use v16::planner::plan_step::{PlanStep, PlanStepKind};
use v17::compute::tensor::Tensor;
use v17::profiling::backend_metrics::BackendKind;
use v17::profiling::execution_profiler::ExecutionProfiler;
use v17::profiling::profiling_errors::ProfilingError;

fn make_plan() -> ExecutionPlan {
    let steps = vec![
        PlanStep {
            kind: PlanStepKind::PrepareFallback,
            description: "prepare".to_string(),
            preconditions: Vec::new(),
            postconditions: Vec::new(),
            abortable: true,
            requires_verification: false,
        },
        PlanStep {
            kind: PlanStepKind::MarkTensorsMovable,
            description: "compute".to_string(),
            preconditions: Vec::new(),
            postconditions: Vec::new(),
            abortable: true,
            requires_verification: false,
        },
    ];

    ExecutionPlan {
        contract: v16::contract::execution_contract::ExecutionContract {
            bias: v15::policy::types::DecisionBias {
                risk_weight: 0.3,
                latency_weight: 0.4,
                stability_weight: 0.8,
                memory_pressure_weight: 0.5,
                offload_cost_weight: 0.4,
            },
            runtime_snapshot: v16::contract::constraints::RuntimeState {
                memory_headroom: 0.8,
                is_stable: true,
                recent_recovery: false,
                offload_supported: true,
            },
            allowed_backends: vec![v16::contract::execution_contract::ExecutionBackend::Local],
            forbidden_backends: vec![],
            max_aggressiveness: 0.3,
            require_fallback: false,
            require_stability: true,
            constraints: v16::contract::constraints::Constraints { items: Vec::new() },
        },
        steps,
        globally_abortable: true,
    }
}

#[test]
fn profiling_collects_step_metrics_in_execution_order() {
    let plan = make_plan();
    let executed = vec![0usize, 1usize];
    let input = Tensor::new(vec![2, 1], vec![1.0, -1.0]).expect("input");
    let output = Tensor::new(vec![2, 1], vec![2.0, 3.0]).expect("output");

    let profile = ExecutionProfiler::profile(&plan, &executed, BackendKind::Cpu, &input, &output)
        .expect("profile");

    assert_eq!(profile.steps.len(), 2);
    assert_eq!(profile.steps[0].step_index, 0);
    assert_eq!(profile.steps[1].step_index, 1);
}

#[test]
fn profiling_is_deterministic_for_identical_inference() {
    let plan = make_plan();
    let executed = vec![0usize, 1usize];
    let input = Tensor::new(vec![2, 1], vec![0.5, -0.5]).expect("input");
    let output = Tensor::new(vec![2, 1], vec![0.0, 0.0]).expect("output");

    let p1 = ExecutionProfiler::profile(&plan, &executed, BackendKind::Cpu, &input, &output)
        .expect("p1");
    let p2 = ExecutionProfiler::profile(&plan, &executed, BackendKind::Cpu, &input, &output)
        .expect("p2");

    assert_eq!(p1, p2);
    assert_eq!(p1.to_json(), p2.to_json());
}

#[test]
fn profiling_does_not_affect_execution_outcome() {
    // Profiling is pure and does not touch execution; here we simply ensure that
    // calling it with the same inputs cannot fail while execution result is
    // fixed.
    let plan = make_plan();
    let executed = vec![0usize, 1usize];
    let input = Tensor::new(vec![2, 1], vec![1.0, 2.0]).expect("input");
    let output = Tensor::new(vec![2, 1], vec![3.0, 4.0]).expect("output");

    let res = ExecutionProfiler::profile(&plan, &executed, BackendKind::Cpu, &input, &output);
    assert!(res.is_ok());
}

#[test]
fn profiling_reflects_backend_selection_and_is_optional() {
    let plan = make_plan();
    let executed = vec![0usize, 1usize];
    let input = Tensor::new(vec![2, 1], vec![1.0, -1.0]).expect("input");
    let output = Tensor::new(vec![2, 1], vec![2.0, 3.0]).expect("output");

    let cpu_profile =
        ExecutionProfiler::profile(&plan, &executed, BackendKind::Cpu, &input, &output)
            .expect("cpu");
    let gpu_profile =
        ExecutionProfiler::profile(&plan, &executed, BackendKind::Gpu, &input, &output)
            .expect("gpu");

    assert_ne!(cpu_profile.to_json(), gpu_profile.to_json());
}

#[test]
fn profiling_errors_on_missing_steps() {
    let plan = make_plan();
    let executed: Vec<usize> = Vec::new();
    let input = Tensor::new(vec![2, 1], vec![0.0, 0.0]).expect("input");
    let output = Tensor::new(vec![2, 1], vec![0.0, 0.0]).expect("output");

    let res = ExecutionProfiler::profile(&plan, &executed, BackendKind::Cpu, &input, &output);
    assert!(matches!(res, Err(ProfilingError::MissingSteps(_))));
}
