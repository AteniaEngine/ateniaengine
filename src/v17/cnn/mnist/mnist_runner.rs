use crate::v16::contract::constraints::{Constraint, ConstraintKind, Constraints, RuntimeState};
use crate::v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use crate::v16::planner::execution_planner::ExecutionPlanner;
use crate::v16::planner::execution_plan::ExecutionPlan;
use crate::v17::compute::tensor::Tensor;
use crate::v17::cnn::activation::relu;
use crate::v17::cnn::bias::add_bias;
use crate::v17::cnn::conv2d::{conv2d_cpu, Conv2DParams, AbortFlag};
use crate::v17::cnn::maxpool2d::{maxpool2d_cpu, MaxPool2DParams};
use crate::v17::cnn::cnn_adapter::{CNNExecutionAdapter, CNNExecutionPlan, CNNGraph, CNNLayer, CNNLayerKind, CNNAdapterError};
use crate::v17::cnn::mnist::mnist_input::mnist_synthetic_input;
use crate::v17::cnn::mnist::mnist_model::MnistCNNModel;
use crate::v17::snapshot::execution_snapshot::ExecutionSnapshot;
use crate::v17::snapshot::snapshot_hash::hash_str;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MnistRunnerError {
    AdapterError(CNNAdapterError),
    Aborted,
    InvalidState(String),
}

#[derive(Debug, Clone)]
pub struct MnistInferenceResult {
    pub logits: Tensor,          // [1, 10]
    pub predicted_digit: usize,
    pub cnn_plan: CNNExecutionPlan,
    pub logical_plan: ExecutionPlan,
    pub snapshot: ExecutionSnapshot,
    pub explanation_text: String,
}

fn make_default_contract() -> ExecutionContract {
    use crate::v15::policy::types::DecisionBias;

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
        allowed_backends: vec![ExecutionBackend::Local],
        forbidden_backends: vec![],
        max_aggressiveness: 0.3,
        require_fallback: false,
        require_stability: true,
        constraints,
    }
}

fn build_logical_plan() -> ExecutionPlan {
    let contract = make_default_contract();
    ExecutionPlanner::build_plan(&contract).expect("logical plan should be produced")
}

fn dense_forward(model: &MnistCNNModel, input: &Tensor) -> Tensor {
    // input is [1, flat_dim].
    let flat_dim = input.shape[1];
    let out_classes = 10usize;
    assert_eq!(model.dense_weights.shape, vec![out_classes, flat_dim]);

    let mut logits = vec![0.0_f32; out_classes];

    for cls in 0..out_classes {
        let mut acc = model.dense_bias.data[cls];
        for i in 0..flat_dim {
            let w = model.dense_weights.data[cls * flat_dim + i];
            let x = input.data[i];
            acc += w * x;
        }
        logits[cls] = acc;
    }

    Tensor::new(vec![1, out_classes], logits).expect("logits shape mismatch")
}

fn flatten_nchw_to_nc(input: &Tensor) -> Tensor {
    assert_eq!(input.shape.len(), 4);
    let n = input.shape[0];
    let c = input.shape[1];
    let h = input.shape[2];
    let w = input.shape[3];
    let flat_dim = c * h * w;
    assert_eq!(n, 1);

    Tensor::new(vec![1, flat_dim], input.data.clone()).expect("flatten shape mismatch")
}

fn build_cnn_graph() -> CNNGraph {
    CNNGraph {
        layers: vec![
            CNNLayer { name: "conv2d".to_string(), kind: CNNLayerKind::Conv2D },
            CNNLayer { name: "bias".to_string(), kind: CNNLayerKind::Bias },
            CNNLayer { name: "relu".to_string(), kind: CNNLayerKind::ReLU },
            CNNLayer { name: "maxpool".to_string(), kind: CNNLayerKind::MaxPool2D },
        ],
    }
}

fn build_cnn_execution_plan(abort_flag: &AbortFlag) -> Result<CNNExecutionPlan, MnistRunnerError> {
    let graph = build_cnn_graph();
    CNNExecutionAdapter::build_plan(&graph, abort_flag).map_err(MnistRunnerError::AdapterError)
}

fn run_cnn_pipeline(
    model: &MnistCNNModel,
    abort_flag: &AbortFlag,
) -> Result<Tensor, MnistRunnerError> {
    if abort_flag.is_aborted() {
        return Err(MnistRunnerError::Aborted);
    }
    let mut x = mnist_synthetic_input();

    // Step 1: Conv2D
    let conv_params = Conv2DParams { stride: (1, 1), padding: (1, 1) };
    x = conv2d_cpu(&x, &model.conv_weights, Some(&model.conv_bias), &conv_params, abort_flag)
        .map_err(|_| MnistRunnerError::InvalidState("conv2d failed".to_string()))?;

    // Step 2: Bias (no-op in this synthetic model as bias is already applied in conv)
    // Included for structural completeness; use zero bias tensor.
    let zero_bias = Tensor::new(vec![1], vec![0.0]).expect("zero bias shape mismatch");
    x = add_bias(&x, &zero_bias, abort_flag)
        .map_err(|_| MnistRunnerError::InvalidState("bias failed".to_string()))?;

    // Step 3: ReLU
    x = relu(&x, abort_flag)
        .map_err(|_| MnistRunnerError::InvalidState("relu failed".to_string()))?;

    // Step 4: MaxPool2D 2x2 stride 2
    let pool_params = MaxPool2DParams { kernel: (2, 2), stride: (2, 2), padding: (0, 0) };
    x = maxpool2d_cpu(&x, &pool_params, abort_flag)
        .map_err(|_| MnistRunnerError::InvalidState("maxpool failed".to_string()))?;

    Ok(x)
}

fn build_snapshot(
    logits: &Tensor,
    cnn_plan: &CNNExecutionPlan,
    logical_plan: &ExecutionPlan,
) -> (ExecutionSnapshot, String) {
    // Simple textual explanation based on CNN plan steps.
    let mut explanation = String::from("CNN execution steps: ");
    for (idx, step) in cnn_plan.steps.iter().enumerate() {
        if idx > 0 {
            explanation.push_str(" -> ");
        }
        explanation.push_str(&step.description);
    }

    let model_id = "mnist_synthetic_cnn".to_string();
    let contract_fingerprint = hash_str("mnist_contract_v1");
    let plan_fingerprint = hash_str(&format!("logical_plan_steps:{}", logical_plan.steps.len()));
    let backend_usage = "cpu".to_string();
    let profile_hash = hash_str("mnist_profile_placeholder");
    let output_signature = hash_str(&format!("{:?}:{:?}", logits.shape, logits.data));
    let explanation_signature = hash_str(&explanation);

    let snapshot_concat = format!(
        "{}|{}|{}|{}|{}|{}|{}",
        model_id,
        contract_fingerprint,
        plan_fingerprint,
        backend_usage,
        profile_hash,
        output_signature,
        explanation_signature,
    );
    let snapshot_hash = hash_str(&snapshot_concat);

    let snapshot = ExecutionSnapshot {
        model_id,
        contract_fingerprint,
        plan_fingerprint,
        backend_usage,
        profile_hash,
        output_signature,
        explanation_signature,
        snapshot_hash,
    };

    (snapshot, explanation)
}

pub fn run_mnist_inference(
    abort_flag: &AbortFlag,
) -> Result<MnistInferenceResult, MnistRunnerError> {
    if abort_flag.is_aborted() {
        return Err(MnistRunnerError::Aborted);
    }

    let model = MnistCNNModel::synthetic();
    let cnn_plan = build_cnn_execution_plan(abort_flag)?;
    let logical_plan = build_logical_plan();

    // Run the CNN pipeline to obtain feature maps, then flatten to feed the
    // dense layer.
    let features = run_cnn_pipeline(&model, abort_flag)?;
    let flat = flatten_nchw_to_nc(&features);
    let logits = dense_forward(&model, &flat);

    // Argmax over classes.
    let mut best_idx = 0usize;
    let mut best_val = logits.data[0];
    for (i, &v) in logits.data.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }

    let (snapshot, explanation_text) = build_snapshot(&logits, &cnn_plan, &logical_plan);

    Ok(MnistInferenceResult {
        logits,
        predicted_digit: best_idx,
        cnn_plan,
        logical_plan,
        snapshot,
        explanation_text,
    })
}
