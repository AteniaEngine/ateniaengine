#![allow(dead_code)]

use atenia_engine::v17;

use v17::model::model_artifact::ModelArtifact;
use v17::model::model_errors::ModelError;
use v17::model::model_format::ModelFormat;
use v17::model::model_metadata::ModelMetadata;

fn make_metadata() -> ModelMetadata {
    ModelMetadata::new(
        "atenia-small".to_string(),
        "1.0.0".to_string(),
        "llm".to_string(),
        Some("Atenia".to_string()),
        "abc123".to_string(),
        123_456_789,
    )
}

#[test]
fn model_artifact_constructs_from_valid_metadata() {
    let metadata = make_metadata();
    let artifact = ModelArtifact::new(
        "model-1".to_string(),
        metadata.clone(),
        ModelFormat::SafeTensors,
        "/models/atenia-small.safetensors".to_string(),
        123_456_789,
    )
    .expect("artifact should be valid");

    assert_eq!(artifact.id, "model-1");
    assert_eq!(artifact.metadata, metadata);
    assert_eq!(artifact.format, ModelFormat::SafeTensors);
    assert_eq!(artifact.location, "/models/atenia-small.safetensors");
    assert_eq!(artifact.total_size_bytes, 123_456_789);
}

#[test]
fn invalid_metadata_yields_explicit_error() {
    let mut metadata = make_metadata();
    metadata.name = "".to_string();

    let result = ModelArtifact::new(
        "model-1".to_string(),
        metadata,
        ModelFormat::Onnx,
        "/models/atenia.onnx".to_string(),
        42,
    );

    assert!(matches!(result, Err(ModelError::InvalidMetadata(_))));
}

#[test]
fn invalid_location_or_size_yield_explicit_errors() {
    let metadata = make_metadata();

    let empty_location = ModelArtifact::new(
        "model-1".to_string(),
        metadata.clone(),
        ModelFormat::Onnx,
        "  ".to_string(),
        42,
    );
    assert!(matches!(empty_location, Err(ModelError::InvalidPath(_))));

    let zero_size = ModelArtifact::new(
        "model-1".to_string(),
        metadata,
        ModelFormat::Onnx,
        "/models/atenia.onnx".to_string(),
        0,
    );
    assert!(matches!(zero_size, Err(ModelError::InvalidSize(_))));
}

#[test]
fn construction_is_deterministic_and_artifact_is_immutable() {
    let metadata = make_metadata();

    let a1 = ModelArtifact::new(
        "model-1".to_string(),
        metadata.clone(),
        ModelFormat::Gguf,
        "/models/atenia.gguf".to_string(),
        10,
    )
    .expect("a1");

    let a2 = ModelArtifact::new(
        "model-1".to_string(),
        metadata,
        ModelFormat::Gguf,
        "/models/atenia.gguf".to_string(),
        10,
    )
    .expect("a2");

    assert_eq!(a1, a2);
}
