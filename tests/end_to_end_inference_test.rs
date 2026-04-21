#![allow(dead_code)]

use atenia_engine::v16;
use atenia_engine::v17;

use std::fs;
use std::path::PathBuf;

use v17::compute::tensor::Tensor;
use v17::inference::infer::infer;
use v17::inference::inference_errors::InferenceError;
use v17::model::model_artifact::ModelArtifact;
use v17::model::model_format::ModelFormat;
use v17::model::model_metadata::ModelMetadata;

fn temp_model_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("atenia_{name}_end_to_end_test.bin"));
    p
}

fn make_metadata() -> ModelMetadata {
    ModelMetadata::new(
        "tiny-linear".to_string(),
        "1.0.0".to_string(),
        "llm".to_string(),
        Some("Atenia".to_string()),
        "feedface".to_string(),
        16,
    )
}

fn make_linear_artifact(path: &PathBuf) -> ModelArtifact {
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
fn end_to_end_inference_produces_correct_output() {
    let path = temp_model_path("ok");
    let artifact = make_linear_artifact(&path);

    let input = Tensor::new(vec![2, 1], vec![2.0, -3.0]).expect("tensor");

    let result = infer(&artifact, input, None).expect("inference");

    assert_eq!(result.output.shape, vec![2, 1]);
    assert_eq!(result.output.data.clone(), vec![2.0, 3.0]);
    assert!(!result.executed_steps.is_empty());
    assert!(!result.explanation_text.is_empty());
    assert!(!result.explanation_json.is_empty());
    assert!(!result.replay_events.is_empty());
    assert!(result.profile.is_some());
}

#[test]
fn inference_respects_execution_contract() {
    let path = temp_model_path("contract");
    let artifact = make_linear_artifact(&path);

    let input = Tensor::new(vec![2, 1], vec![1.0, 1.0]).expect("tensor");

    let result = infer(&artifact, input, None).expect("inference");

    // Under the default contract, inference must succeed and outcome should be
    // completed.
    use v16::feedback::execution_outcome::ExecutionOutcomeKind;
    assert!(matches!(
        result.outcome.kind,
        ExecutionOutcomeKind::Completed | ExecutionOutcomeKind::PartiallyCompleted
    ));
}

#[test]
fn abort_during_inference_yields_explicit_error() {
    let path = temp_model_path("missing");
    // Intentionally do not write any file here, so loading fails.
    let artifact = ModelArtifact::new(
        "model-missing".to_string(),
        make_metadata(),
        ModelFormat::Raw,
        path.to_string_lossy().to_string(),
        16,
    )
    .expect("artifact");

    let input = Tensor::new(vec![2, 1], vec![0.0, 0.0]).expect("tensor");
    let result = infer(&artifact, input, None);

    assert!(matches!(result, Err(InferenceError::LoadFailed(_))));
}

#[test]
fn end_to_end_inference_is_deterministic() {
    let path = temp_model_path("det");
    let artifact = make_linear_artifact(&path);

    let input = Tensor::new(vec![2, 1], vec![0.5, -0.5]).expect("tensor");

    let r1 = infer(&artifact, input.clone(), None).expect("r1");
    let r2 = infer(&artifact, input, None).expect("r2");

    assert_eq!(r1.output, r2.output);
    assert_eq!(r1.executed_steps, r2.executed_steps);
    assert_eq!(r1.outcome, r2.outcome);
    assert_eq!(r1.profile.is_some(), r2.profile.is_some());
}
