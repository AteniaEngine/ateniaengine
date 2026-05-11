#![allow(dead_code)]

use atenia_engine::v15;
use atenia_engine::v16;
use atenia_engine::v17;

use v15::policy::types::DecisionBias;
use v16::contract::constraints::{Constraint, ConstraintKind, Constraints, RuntimeState};
use v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use v16::guards::guard_action::GuardAction;
use v16::planner::execution_plan::ExecutionPlan;
use v16::planner::plan_step::{PlanStep, PlanStepKind};
use v17::adapter::adapter_context::AdapterContext;
use v17::adapter::adapter_errors::AdapterError;
use v17::adapter::execution_adapter::ExecutionAdapter;
use v17::compute::cpu_backend::CpuBackend;
use v17::compute::tensor::Tensor;
use v17::loader::loader_policy::LoaderPolicy;
use v17::loader::model_loader::ModelLoader;
use v17::model::model_artifact::ModelArtifact;
use v17::model::model_format::ModelFormat;
use v17::model::model_metadata::ModelMetadata;

use std::fs;
use std::path::PathBuf;

fn temp_model_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("atenia_{name}_execution_adapter_test.bin"));
    p
}

fn make_contract(stable: bool) -> ExecutionContract {
    let bias = DecisionBias {
        risk_weight: 0.3,
        latency_weight: 0.4,
        stability_weight: 0.8,
        memory_pressure_weight: 0.5,
        offload_cost_weight: 0.4,
    };

    let state = RuntimeState {
        memory_headroom: 0.8,
        is_stable: stable,
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
        allowed_backends: vec![ExecutionBackend::Local],
        forbidden_backends: vec![],
        max_aggressiveness: 0.3,
        require_fallback: false,
        require_stability: true,
        constraints,
    }
}

fn make_metadata() -> ModelMetadata {
    ModelMetadata::new(
        "tiny-linear".to_string(),
        "1.0.0".to_string(),
        "llm".to_string(),
        Some("Atenia".to_string()),
        "cafebabe".to_string(),
        16,
    )
}

fn make_linear_model(path: &PathBuf) -> ModelArtifact {
    let weights: [f32; 4] = [1.0, 0.0, 0.0, -1.0];
    let mut bytes = Vec::new();
    for w in &weights {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    fs::write(path, &bytes).expect("write weights");

    ModelArtifact::new(
        "model-linear".to_string(),
        make_metadata(),
        ModelFormat::Raw,
        path.to_string_lossy().to_string(),
        bytes.len() as u64,
    )
    .expect("artifact")
}

fn make_plan_with_compute_step() -> ExecutionPlan {
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
        contract: make_contract(true),
        steps,
        globally_abortable: true,
    }
}

fn make_plan_with_unsupported_step() -> ExecutionPlan {
    let steps = vec![PlanStep {
        kind: PlanStepKind::EnsureMemoryHeadroom,
        description: "unsupported".to_string(),
        preconditions: Vec::new(),
        postconditions: Vec::new(),
        abortable: true,
        requires_verification: false,
    }];

    ExecutionPlan {
        contract: make_contract(true),
        steps,
        globally_abortable: true,
    }
}

#[test]
fn execution_plan_steps_are_executed_in_order() {
    let path = temp_model_path("order");
    let artifact = make_linear_model(&path);
    let policy = LoaderPolicy::LoadAll;
    let handle = ModelLoader::load(&artifact, &policy, 1024 * 1024).expect("load");

    let plan = make_plan_with_compute_step();

    let backend = CpuBackend::new();
    let mut ctx = AdapterContext::new(handle, plan.contract.clone(), GuardAction::Continue);
    let adapter = ExecutionAdapter::new(backend);

    let input = Tensor::new(vec![2, 1], vec![1.0, -1.0]).expect("tensor");

    let _ = adapter
        .execute_plan(&plan, &mut ctx, &input)
        .expect("execute");

    assert_eq!(ctx.executed_steps, vec![0, 1]);
}

#[test]
fn unsupported_step_yields_explicit_error() {
    let path = temp_model_path("unsupported");
    let artifact = make_linear_model(&path);
    let policy = LoaderPolicy::LoadAll;
    let handle = ModelLoader::load(&artifact, &policy, 1024 * 1024).expect("load");

    let plan = make_plan_with_unsupported_step();

    let backend = CpuBackend::new();
    let mut ctx = AdapterContext::new(handle, plan.contract.clone(), GuardAction::Continue);
    let adapter = ExecutionAdapter::new(backend);

    let input = Tensor::new(vec![2, 1], vec![0.0, 0.0]).expect("tensor");

    let result = adapter.execute_plan(&plan, &mut ctx, &input);
    assert!(matches!(result, Err(AdapterError::BackendFailure(_))));
}

#[test]
fn abort_stops_execution_safely() {
    let path = temp_model_path("abort");
    let artifact = make_linear_model(&path);
    let policy = LoaderPolicy::LoadAll;
    let handle = ModelLoader::load(&artifact, &policy, 1024 * 1024).expect("load");

    let plan = make_plan_with_compute_step();

    let backend = CpuBackend::new();
    let mut ctx = AdapterContext::new(handle, plan.contract.clone(), GuardAction::Abort);
    let adapter = ExecutionAdapter::new(backend);

    let input = Tensor::new(vec![2, 1], vec![1.0, -1.0]).expect("tensor");

    let result = adapter.execute_plan(&plan, &mut ctx, &input);
    assert!(matches!(result, Err(AdapterError::AbortedByGuard(_))));
    assert!(ctx.last_output.is_none());
}

#[test]
fn adapter_respects_execution_contract_and_is_deterministic() {
    let path = temp_model_path("det");
    let artifact = make_linear_model(&path);
    let policy = LoaderPolicy::LoadAll;
    let handle = ModelLoader::load(&artifact, &policy, 1024 * 1024).expect("load");

    let plan = make_plan_with_compute_step();

    let backend = CpuBackend::new();
    let mut ctx1 =
        AdapterContext::new(handle.clone(), plan.contract.clone(), GuardAction::Continue);
    let mut ctx2 = AdapterContext::new(handle, plan.contract.clone(), GuardAction::Continue);
    let adapter = ExecutionAdapter::new(backend);

    let input = Tensor::new(vec![2, 1], vec![2.0, -3.0]).expect("tensor");

    let y1 = adapter.execute_plan(&plan, &mut ctx1, &input).expect("y1");
    let y2 = adapter.execute_plan(&plan, &mut ctx2, &input).expect("y2");

    assert_eq!(y1, y2);
}
