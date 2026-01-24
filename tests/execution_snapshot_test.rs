#![allow(dead_code)]

use atenia_engine::v15;
use atenia_engine::v16;
use atenia_engine::v17;

use v15::policy::types::DecisionBias;
use v16::contract::constraints::RuntimeState;
use v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use v16::contract::constraints::Constraints;
use v16::planner::execution_plan::ExecutionPlan;
use v16::planner::plan_step::{PlanStep, PlanStepKind};
use v16::feedback::execution_outcome::{ExecutionOutcome, ExecutionOutcomeKind};
use v17::compute::tensor::Tensor;
use v17::inference::inference_result::InferenceResult;
use v17::profiling::backend_metrics::{BackendKind, BackendMetrics, ExecutionProfile};
use v17::snapshot::snapshot_builder::SnapshotBuilder;
use v17::snapshot::snapshot_errors::SnapshotError;

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
        contract: ExecutionContract {
            bias: DecisionBias {
                risk_weight: 0.3,
                latency_weight: 0.4,
                stability_weight: 0.8,
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
            max_aggressiveness: 0.3,
            require_fallback: false,
            require_stability: true,
            constraints: Constraints { items: Vec::new() },
        },
        steps,
        globally_abortable: true,
    }
}

fn make_profile() -> ExecutionProfile {
    let backend_metrics = BackendMetrics {
        backend: BackendKind::Cpu,
        matmul_count: 1,
        add_count: 0,
        relu_count: 1,
        elements_processed: 4,
    };
    ExecutionProfile {
        steps: Vec::new(),
        backends: vec![backend_metrics],
    }
}

fn make_inference_result() -> InferenceResult {
    let output = Tensor::new(vec![2, 1], vec![2.0, 3.0]).expect("output");
    let outcome = ExecutionOutcome {
        kind: ExecutionOutcomeKind::Completed,
        executed_steps: vec![0, 1],
        final_error: None,
    };

    InferenceResult {
        output,
        outcome,
        executed_steps: vec![0, 1],
        explanation_text: "ok".to_string(),
        explanation_json: "{}".to_string(),
        replay_events: Vec::new(),
        replay_outcome: ExecutionOutcome {
            kind: ExecutionOutcomeKind::Completed,
            executed_steps: vec![0, 1],
            final_error: None,
        },
        profile: Some(make_profile()),
    }
}

#[test]
fn snapshot_is_built_from_valid_inference_result() {
    let plan = make_plan();
    let contract = &plan.contract;
    let result = make_inference_result();

    let snapshot = SnapshotBuilder::build(&result, contract, &plan).expect("snapshot");
    let json = snapshot.to_json();
    assert!(json.contains("\"model_id\""));
}

#[test]
fn snapshot_hash_is_deterministic() {
    let plan = make_plan();
    let contract = &plan.contract;
    let result = make_inference_result();

    let s1 = SnapshotBuilder::build(&result, contract, &plan).expect("s1");
    let s2 = SnapshotBuilder::build(&result, contract, &plan).expect("s2");

    assert_eq!(s1.snapshot_hash, s2.snapshot_hash);
    assert_eq!(s1.to_json(), s2.to_json());
}

#[test]
fn identical_executions_yield_identical_snapshots() {
    let plan = make_plan();
    let contract = &plan.contract;
    let r1 = make_inference_result();
    let r2 = make_inference_result();

    let s1 = SnapshotBuilder::build(&r1, contract, &plan).expect("s1");
    let s2 = SnapshotBuilder::build(&r2, contract, &plan).expect("s2");

    assert_eq!(s1, s2);
}

#[test]
fn snapshot_rejects_incomplete_execution_data() {
    let plan = make_plan();
    let contract = &plan.contract;
    let mut result = make_inference_result();
    result.executed_steps.clear();

    let res = SnapshotBuilder::build(&result, contract, &plan);
    assert!(matches!(res, Err(SnapshotError::IncompleteExecution(_))));
}

#[test]
fn snapshot_rejects_missing_profile_or_explanation() {
    let plan = make_plan();
    let contract = &plan.contract;
    let mut result = make_inference_result();
    result.profile = None;

    let res = SnapshotBuilder::build(&result, contract, &plan);
    assert!(matches!(res, Err(SnapshotError::MissingProfile(_))));

    let mut result = make_inference_result();
    result.explanation_text = "".to_string();
    let res = SnapshotBuilder::build(&result, contract, &plan);
    assert!(matches!(res, Err(SnapshotError::MissingExplanation(_))));
}
