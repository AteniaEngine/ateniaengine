#![allow(dead_code)]

use crate::v15::policy::preferences::user_preferences::UserPreferences;
use crate::v15::policy::types::DecisionBias;
use crate::v16::contract::constraints::{Constraint, ConstraintKind, Constraints, RuntimeState};
use crate::v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use crate::v16::feedback::event_emitter::EventEmitter;
use crate::v16::feedback::feedback_errors::FeedbackError;
use crate::v16::executor::executor_state::ExecutorStatus;
use crate::v16::planner::execution_planner::ExecutionPlanner;
use crate::v17::adapter::adapter_context::AdapterContext;
use crate::v17::adapter::adapter_errors::AdapterError;
use crate::v17::adapter::execution_adapter::ExecutionAdapter;
use crate::v17::compute::cpu_backend::CpuBackend;
use crate::v17::compute::tensor::Tensor;
use crate::v17::inference::inference_context::InferenceContext;
use crate::v17::inference::inference_errors::InferenceError;
use crate::v17::inference::inference_result::InferenceResult;
use crate::v17::loader::loader_policy::LoaderPolicy;
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::model_loader::ModelLoader;
use crate::v17::model::model_artifact::ModelArtifact;

use crate::v16::explain::explanation_builder::ExplanationBuilder;
use crate::v17::profiling::backend_metrics::{BackendKind, ExecutionProfile};
use crate::v17::profiling::execution_profiler::ExecutionProfiler;

/// Build a simple default execution contract for inference.
fn make_default_contract() -> ExecutionContract {
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

fn build_inference_context(
    artifact: &ModelArtifact,
    _preferences: Option<UserPreferences>,
) -> Result<InferenceContext, InferenceError> {
    if artifact.id.trim().is_empty() {
        return Err(InferenceError::InvalidArtifact(
            "artifact id must not be empty".to_string(),
        ));
    }

    let policy = LoaderPolicy::LoadAll;
    let loaded_model = ModelLoader::load(artifact, &policy, 1_048_576)
        .map_err(|e| match e {
            LoaderError::FileNotFound(p) => InferenceError::LoadFailed(format!("file not found: {p}")),
            LoaderError::SizeMismatch { expected, actual } => {
                InferenceError::LoadFailed(format!("size mismatch: expected {}, actual {}", expected, actual))
            }
            LoaderError::InsufficientMemory { required, available } => InferenceError::LoadFailed(format!(
                "insufficient memory: required {}, available {}",
                required, available
            )),
            LoaderError::PolicyDenied(msg) | LoaderError::IoError(msg) => {
                InferenceError::LoadFailed(format!("io/policy error: {msg}"))
            }
            // M4-a additions: InvalidFormat and UnsupportedDType are
            // produced by the safetensors reader, which is not yet
            // wired into this inference entrypoint (M4-b/c will add
            // that). Route them through the same LoadFailed variant
            // so the match stays exhaustive.
            LoaderError::InvalidFormat(msg) => {
                InferenceError::LoadFailed(format!("invalid model format: {msg}"))
            }
            LoaderError::UnsupportedDType(msg) => {
                InferenceError::LoadFailed(format!("unsupported dtype: {msg}"))
            }
            LoaderError::ShapeMismatch { tensor_name, expected, actual } => {
                InferenceError::LoadFailed(format!(
                    "shape mismatch for '{}': expected {:?}, got {:?}",
                    tensor_name, expected, actual
                ))
            }
        })?;

    let contract = make_default_contract();
    let plan = ExecutionPlanner::build_plan(&contract)
        .map_err(|e| InferenceError::PlanningError(format!("planner: {:?}", e)))?;

    let backend = CpuBackend::new();
    let adapter_ctx = AdapterContext::new(loaded_model.clone(), contract.clone(), GuardAction::Continue);

    Ok(InferenceContext {
        artifact: artifact.clone(),
        loaded_model,
        contract,
        plan,
        backend,
        adapter_ctx,
    })
}

use crate::v16::guards::guard_action::GuardAction;

fn build_feedback_and_explanation(
    ctx: &InferenceContext,
    executed_steps: &[usize],
) -> Result<(Vec<crate::v16::feedback::execution_event::ExecutionEvent>, crate::v16::feedback::execution_outcome::ExecutionOutcome, String, String), InferenceError> {
    let (events, outcome) = EventEmitter::emit_for_snapshot(
        &ctx.plan,
        executed_steps,
        &ExecutorStatus::Completed,
        None,
    )
    .map_err(|e: FeedbackError| InferenceError::FeedbackError(format!("feedback: {:?}", e)))?;

    let explanation = ExplanationBuilder::build(
        &ctx.contract.bias,
        &ctx.contract,
        "end-to-end inference".to_string(),
        events.clone(),
        outcome.clone(),
        Vec::new(),
        None,
    )
    .map_err(|e| InferenceError::FeedbackError(format!("explain: {:?}", e)))?;

    use crate::v16::explain::explanation_formatter::{format_explanation_json, format_explanation_text};
    let text = format_explanation_text(&explanation);
    let json = format_explanation_json(&explanation);

    Ok((events, outcome, text, json))
}

pub fn infer(
    artifact: &ModelArtifact,
    input: Tensor,
    preferences: Option<UserPreferences>,
) -> Result<InferenceResult, InferenceError> {
    let mut ctx = build_inference_context(artifact, preferences)?;

    let adapter = ExecutionAdapter::new(ctx.backend.clone());

    let output = adapter
        .execute_plan(&ctx.plan, &mut ctx.adapter_ctx, &input)
        .map_err(|e: AdapterError| InferenceError::AdapterError(format!("adapter: {:?}", e)))?;

    let executed_steps = ctx.adapter_ctx.executed_steps.clone();
    let (events, outcome, text, json) = build_feedback_and_explanation(&ctx, &executed_steps)?;

    // Build profiling information based solely on logical data; this is
    // deterministic and has no effect on execution behavior.
    let profile = ExecutionProfiler::profile(
        &ctx.plan,
        &ctx.adapter_ctx.executed_steps,
        BackendKind::Cpu,
        &input,
        &output,
    )
    .unwrap_or_else(|_| ExecutionProfile {
        steps: Vec::new(),
        backends: Vec::new(),
    });

    Ok(InferenceResult {
        output,
        outcome: outcome.clone(),
        executed_steps,
        explanation_text: text,
        explanation_json: json,
        replay_events: events,
        replay_outcome: outcome,
        profile: Some(profile),
    })
}
