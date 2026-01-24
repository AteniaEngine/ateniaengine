#![allow(dead_code)]

use atenia_engine::v15;
use atenia_engine::v16;
use atenia_engine::v17;

use v15::policy::types::DecisionBias;
use v16::contract::constraints::{Constraint, ConstraintKind, Constraints, RuntimeState};
use v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use v16::guards::guard_action::GuardAction;
use v17::compute::backend_trait::ComputeBackend;
use v17::compute::compute_errors::ComputeError;
use v17::compute::cpu_backend::CpuBackend;
use v17::compute::gpu_backend::GpuBackend;
use v17::compute::tensor::Tensor;
use v17::loader::model_loader::LoadedModelHandle;

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

fn make_dummy_model() -> LoadedModelHandle {
    // Use a 2x2 identity-like matrix for simplicity.
    let weights: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    let mut bytes = Vec::new();
    for w in &weights {
        bytes.extend_from_slice(&w.to_le_bytes());
    }

    use v17::loader::memory_map::{MemoryMap, MemorySegment};

    let memory_map = MemoryMap {
        artifact_id: "dummy".to_string(),
        total_size_bytes: bytes.len() as u64,
        loaded_bytes: bytes.len() as u64,
        segments: vec![MemorySegment {
            offset: 0,
            length: bytes.len() as u64,
        }],
    };

    LoadedModelHandle {
        artifact_id: "dummy".to_string(),
        format: v17::model::model_format::ModelFormat::Raw,
        memory_map,
        bytes,
    }
}

#[test]
fn gpu_backend_implements_compute_backend_and_matches_cpu_output() {
    let cpu = CpuBackend::new();
    let gpu = GpuBackend::new();

    let a = Tensor::new(vec![2, 1], vec![1.0, 2.0]).expect("a");
    let b = Tensor::new(vec![2, 1], vec![3.0, 4.0]).expect("b");

    let cpu_add = cpu.add(&a, &b).expect("cpu add");
    let gpu_add = gpu.add(&a, &b).expect("gpu add");
    assert_eq!(cpu_add, gpu_add);

    let relu_in = Tensor::new(vec![2, 1], vec![-1.0, 2.0]).expect("relu in");
    let cpu_relu = cpu.relu(&relu_in).expect("cpu relu");
    let gpu_relu = gpu.relu(&relu_in).expect("gpu relu");
    assert_eq!(cpu_relu, gpu_relu);
}

#[test]
fn gpu_and_cpu_produce_equivalent_matmul_output() {
    let cpu = CpuBackend::new();
    let gpu = GpuBackend::new();

    let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("a");
    let b = Tensor::new(vec![2, 1], vec![1.0, -1.0]).expect("b");

    let cpu_y = cpu.matmul(&a, &b).expect("cpu matmul");
    let gpu_y = gpu.matmul(&a, &b).expect("gpu matmul");

    assert_eq!(cpu_y, gpu_y);
}

#[test]
fn gpu_backend_respects_execution_contract_via_run_inference() {
    let gpu = GpuBackend::new();
    let model = make_dummy_model();
    let input = Tensor::new(vec![2, 1], vec![1.0, -1.0]).expect("input");

    let good_contract = make_contract(true);
    let bad_contract = make_contract(false);

    // Under a stable contract, inference succeeds.
    let y_ok = gpu
        .run_inference(&model, &input, &good_contract, GuardAction::Continue)
        .expect("gpu inference");

    // Under an unstable contract, we get a contract violation.
    let err = gpu.run_inference(&model, &input, &bad_contract, GuardAction::Continue);
    assert!(matches!(err, Err(ComputeError::ContractViolation(_))));

    // Determinism: repeated runs under the same inputs produce identical outputs.
    let y2 = gpu
        .run_inference(&model, &input, &good_contract, GuardAction::Continue)
        .expect("gpu inference 2");
    assert_eq!(y_ok, y2);
}

#[test]
fn gpu_fallback_to_cpu_is_safe() {
    let cpu = CpuBackend::new();
    let gpu = GpuBackend::new();
    let model = make_dummy_model();
    let input = Tensor::new(vec![2, 1], vec![0.5, -0.5]).expect("input");
    let contract = make_contract(true);

    let y_gpu = gpu
        .run_inference(&model, &input, &contract, GuardAction::Continue)
        .expect("gpu");
    let y_cpu = cpu
        .run_inference(&model, &input, &contract, GuardAction::Continue)
        .expect("cpu");

    // In this version GPU and CPU share the same implementation, so fallback to
    // CPU is equivalent and fully deterministic.
    assert_eq!(y_gpu, y_cpu);
}
