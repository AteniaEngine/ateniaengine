#![allow(dead_code)]

use atenia_engine::v17;

use std::fs;
use std::path::PathBuf;

use v17::loader::loader_errors::LoaderError;
use v17::loader::loader_policy::LoaderPolicy;
use v17::loader::model_loader::ModelLoader;
use v17::model::model_artifact::ModelArtifact;
use v17::model::model_format::ModelFormat;
use v17::model::model_metadata::ModelMetadata;

fn temp_model_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("atenia_{name}_model_loader_test.bin"));
    p
}

fn make_metadata() -> ModelMetadata {
    ModelMetadata::new(
        "atenia-small".to_string(),
        "1.0.0".to_string(),
        "llm".to_string(),
        Some("Atenia".to_string()),
        "abc123".to_string(),
        16,
    )
}

fn make_artifact(path: &PathBuf, size: u64) -> ModelArtifact {
    ModelArtifact::new(
        "model-1".to_string(),
        make_metadata(),
        ModelFormat::SafeTensors,
        path.to_string_lossy().to_string(),
        size,
    )
    .expect("valid artifact")
}

#[test]
fn valid_model_is_loaded_into_ram() {
    let path = temp_model_path("valid");
    let data = vec![1u8; 16];
    fs::write(&path, &data).expect("write test file");

    let artifact = make_artifact(&path, data.len() as u64);
    let policy = LoaderPolicy::LoadAll;

    let handle = ModelLoader::load(&artifact, &policy, 1024 * 1024)
        .expect("model should load");

    assert_eq!(handle.artifact_id, artifact.id);
    assert_eq!(handle.bytes, data);
    assert_eq!(handle.memory_map.total_size_bytes, artifact.total_size_bytes);
    assert!(handle.memory_map.fully_loaded());
}

#[test]
fn insufficient_memory_yields_explicit_error() {
    let path = temp_model_path("oom");
    let data = vec![0u8; 32];
    fs::write(&path, &data).expect("write test file");

    let artifact = make_artifact(&path, data.len() as u64);
    let policy = LoaderPolicy::FailIfInsufficientRam;

    let result = ModelLoader::load(&artifact, &policy, 16);
    assert!(matches!(
        result,
        Err(LoaderError::InsufficientMemory { required, available }) if required == 32 && available == 16
    ));
}

#[test]
fn loaded_model_matches_artifact_size_and_is_deterministic() {
    let path = temp_model_path("deterministic");
    let data = vec![2u8; 24];
    fs::write(&path, &data).expect("write test file");

    let artifact = make_artifact(&path, data.len() as u64);
    let policy = LoaderPolicy::LoadAll;

    let h1 = ModelLoader::load(&artifact, &policy, 1024 * 1024).expect("h1");
    let h2 = ModelLoader::load(&artifact, &policy, 1024 * 1024).expect("h2");

    assert_eq!(h1.bytes.len() as u64, artifact.total_size_bytes);
    assert_eq!(h1.memory_map, h2.memory_map);
    assert_eq!(h1.bytes, h2.bytes);
}

#[test]
fn no_compute_is_triggered_during_load() {
    let path = temp_model_path("nocompute");
    // Use a non-trivial pattern to ensure bytes are not altered.
    let data: Vec<u8> = (0u8..16u8).collect();
    fs::write(&path, &data).expect("write test file");

    let artifact = make_artifact(&path, data.len() as u64);
    let policy = LoaderPolicy::LoadAll;

    let handle = ModelLoader::load(&artifact, &policy, 1024 * 1024)
        .expect("model should load");

    // The loader must not transform bytes; it only moves them into RAM.
    assert_eq!(handle.bytes, data);
}
