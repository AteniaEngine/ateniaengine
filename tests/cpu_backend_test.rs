#![allow(dead_code)]

use atenia_engine::v15;
use atenia_engine::v16;
use atenia_engine::v17;

use std::fs;
use std::path::PathBuf;

use v15::policy::types::DecisionBias;

use v16::contract::constraints::{Constraint, ConstraintKind, Constraints, RuntimeState};
use v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use v16::guards::guard_action::GuardAction;
use v17::compute::compute_errors::ComputeError;
use v17::compute::cpu_backend::CpuBackend;
use v17::compute::tensor::Tensor;
use v17::loader::loader_policy::LoaderPolicy;
use v17::loader::model_loader::ModelLoader;
use v17::model::model_artifact::ModelArtifact;
use v17::model::model_format::ModelFormat;
use v17::model::model_metadata::ModelMetadata;

fn temp_model_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("atenia_{name}_cpu_backend_test.bin"));
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
        "deadbeef".to_string(),
        16,
    )
}

fn make_linear_model(path: &PathBuf) -> ModelArtifact {
    // 2x2 weight matrix stored as f32 little-endian: [[1.0, 0.0], [0.0, -1.0]]
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

#[test]
fn simple_inference_produces_correct_output() {
    let path = temp_model_path("simple_inference");
    let artifact = make_linear_model(&path);
    let policy = LoaderPolicy::LoadAll;

    let handle = ModelLoader::load(&artifact, &policy, 1024 * 1024).expect("load model");

    let backend = CpuBackend::new();
    let contract = make_contract(true);

    let input = Tensor::new(vec![2, 1], vec![2.0, -3.0]).expect("tensor");

    let out = backend
        .run_inference(&handle, &input, &contract, GuardAction::Continue)
        .expect("inference");

    // y = ReLU(W x) with W = [[1, 0], [0, -1]] and x = [2, -3]^
    // W x = [2, 3]; ReLU -> [2, 3]
    assert_eq!(out.shape, vec![2, 1]);
    assert_eq!(out.data.clone(), vec![2.0, 3.0]);
}

#[test]
fn backend_respects_execution_contract() {
    let path = temp_model_path("contract");
    let artifact = make_linear_model(&path);
    let policy = LoaderPolicy::LoadAll;
    let handle = ModelLoader::load(&artifact, &policy, 1024 * 1024).expect("load");

    let backend = CpuBackend::new();
    let contract = make_contract(false); // unstable runtime
    let input = Tensor::new(vec![2, 1], vec![1.0, 1.0]).expect("tensor");

    let result = backend.run_inference(&handle, &input, &contract, GuardAction::Continue);
    assert!(matches!(result, Err(ComputeError::ContractViolation(_))));
}

#[test]
fn invalid_shapes_yield_explicit_error() {
    use v17::compute::ops::matmul;

    let a = Tensor::new(vec![2, 3], vec![0.0; 6]).expect("a");
    let b = Tensor::new(vec![4, 2], vec![0.0; 8]).expect("b");

    let result = matmul(&a, &b);
    assert!(matches!(result, Err(ComputeError::ShapeMismatch(_))));
}

#[test]
fn inference_is_deterministic_and_abortable() {
    let path = temp_model_path("deterministic");
    let artifact = make_linear_model(&path);
    let policy = LoaderPolicy::LoadAll;
    let handle = ModelLoader::load(&artifact, &policy, 1024 * 1024).expect("load");

    let backend = CpuBackend::new();
    let contract = make_contract(true);
    let input = Tensor::new(vec![2, 1], vec![0.5, -0.5]).expect("tensor");

    let y1 = backend
        .run_inference(&handle, &input, &contract, GuardAction::Continue)
        .expect("y1");
    let y2 = backend
        .run_inference(&handle, &input, &contract, GuardAction::Continue)
        .expect("y2");

    assert_eq!(y1, y2);

    // Abort via guard.
    let aborted = backend.run_inference(&handle, &input, &contract, GuardAction::Abort);
    assert!(matches!(aborted, Err(ComputeError::AbortedByGuard(_))));
}
