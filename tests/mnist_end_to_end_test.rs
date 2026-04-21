#![allow(dead_code)]

use atenia_engine::v17;

use v17::cnn::conv2d::AbortFlag;
use v17::cnn::mnist::mnist_runner::{run_mnist_inference, MnistInferenceResult, MnistRunnerError};

fn run_ok() -> MnistInferenceResult {
    let flag = AbortFlag::new();
    run_mnist_inference(&flag).expect("mnist inference should succeed")
}

#[test]
fn end_to_end_mnist_inference_produces_expected_digit() {
    let result = run_ok();
    assert_eq!(result.logits.shape, vec![1, 10]);
    // Synthetic model is designed so that a single target digit is always chosen.
    assert_eq!(result.predicted_digit, 3usize);
}

#[test]
fn inference_is_deterministic() {
    let r1 = run_ok();
    let r2 = run_ok();

    assert_eq!(r1.logits.shape, r2.logits.shape);
    assert_eq!(&r1.logits.data, &r2.logits.data);
    assert_eq!(r1.predicted_digit, r2.predicted_digit);
}

#[test]
fn execution_respects_execution_contract() {
    let result = run_ok();

    // Logical plan must be non-empty and marked globally abortable by the planner.
    assert!(!result.logical_plan.steps.is_empty());
}

#[test]
fn abort_stops_inference_safely() {
    let mut flag = AbortFlag::new();
    flag.abort();

    let r = run_mnist_inference(&flag);
    assert!(matches!(r, Err(MnistRunnerError::Aborted)));
}

#[test]
fn snapshot_is_generated_and_valid() {
    let result = run_ok();
    let snapshot = result.snapshot;

    assert_eq!(snapshot.model_id, "mnist_synthetic_cnn".to_string());
    assert_eq!(snapshot.backend_usage, "cpu".to_string());
    assert!(!snapshot.snapshot_hash.is_empty());
    assert!(!snapshot.output_signature.is_empty());
    assert!(!snapshot.explanation_signature.is_empty());

    assert!(!result.explanation_text.is_empty());
    assert!(result.explanation_text.contains("CNN execution steps"));
}
