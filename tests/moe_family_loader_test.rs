//! **MOE-FULL-9** — integration test: family-aware loader fail-loud + family
//! recognition on real checkpoint tensors.
//!
//! The productive loader still **refuses** a MoE checkpoint (fail-loud guard
//! unchanged), but now emits a precise, family-aware message:
//!
//! ```text
//! MoE detected
//! Family: Mixtral (...)
//! Productive support not enabled (...)
//! ```
//!
//! This test drives the **real** `WeightMapper::load_into` path and asserts the
//! error is `LoaderError::MoeUnsupported` carrying that message; it also checks
//! family classification against the committed Mixtral fixture, and that a
//! dense checkpoint still loads.

use std::collections::HashMap;
use std::path::PathBuf;

use atenia_engine::amg::graph::Graph;
use atenia_engine::moe::{classify_family, moe_failloud_report, MoeFamily};
use atenia_engine::v17::loader::loader_errors::LoaderError;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use atenia_engine::v17::loader::weight_mapper::WeightMapper;
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

/// Build a `SafetensorsReader` from a set of tensor names (values irrelevant).
fn reader_with_names(names: &[&str]) -> SafetensorsReader {
    let payload = [0u8; 4];
    let bufs: Vec<[u8; 4]> = names.iter().map(|_| payload).collect();
    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (i, name) in names.iter().enumerate() {
        let v = TensorView::new(StDtype::F32, vec![1], &bufs[i]).unwrap();
        views.insert((*name).to_string(), v);
    }
    let buffer = safetensors::serialize(&views, &None).unwrap();
    SafetensorsReader::from_bytes(buffer).unwrap()
}

#[test]
fn loader_fails_loud_with_family_message_on_mixtral() {
    let reader = reader_with_names(&[
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.block_sparse_moe.experts.0.w2.weight",
        "model.layers.0.block_sparse_moe.experts.0.w3.weight",
        "model.layers.0.block_sparse_moe.experts.7.w1.weight",
    ]);
    // Empty mapping: the MoE guard fires before the mapping loop.
    let mapper = WeightMapper::from_param_names_and_ids(&[], &[]).unwrap();
    let mut graph = Graph::build(Vec::new());
    let err = mapper.load_into(&mut graph, &reader).unwrap_err();
    match err {
        LoaderError::MoeUnsupported(msg) => {
            assert!(msg.contains("MoE detected"), "msg: {msg}");
            assert!(msg.contains("Family: Mixtral"), "msg: {msg}");
            assert!(msg.contains("Productive support not enabled"), "msg: {msg}");
            assert!(msg.contains("experts=8"), "msg: {msg}");
        }
        other => panic!("expected MoeUnsupported, got {other:?}"),
    }
}

#[test]
fn loader_names_qwen_moe_family() {
    let reader = reader_with_names(&[
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
    ]);
    let mapper = WeightMapper::from_param_names_and_ids(&[], &[]).unwrap();
    let mut graph = Graph::build(Vec::new());
    let err = mapper.load_into(&mut graph, &reader).unwrap_err();
    match err {
        LoaderError::MoeUnsupported(msg) => {
            assert!(msg.contains("Family: Qwen-MoE"), "msg: {msg}");
        }
        other => panic!("expected MoeUnsupported, got {other:?}"),
    }
}

#[test]
fn dense_checkpoint_still_loads_past_the_guard() {
    // A dense checkpoint must NOT trigger the MoE guard. With an empty mapping
    // every tensor is "skipped", but crucially `load_into` returns Ok (the
    // guard did not fire), proving dense models are unaffected.
    let reader = reader_with_names(&[
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
    ]);
    let mapper = WeightMapper::from_param_names_and_ids(&[], &[]).unwrap();
    let mut graph = Graph::build(Vec::new());
    let report = mapper.load_into(&mut graph, &reader).expect("dense must not fail loud");
    assert_eq!(report.loaded, 0); // empty mapping → nothing loaded, but no error
    assert!(!report.skipped.is_empty());
}

#[test]
fn real_mixtral_fixture_classifies_as_mixtral() {
    // The committed real tiny Mixtral (MOE-FULL-6) is recognised as Mixtral and
    // produces the family-aware loader report.
    let reader = SafetensorsReader::open(&fixture_dir().join("full_mixtral.safetensors")).unwrap();
    let names: Vec<String> = reader.iter().map(|e| e.name.to_string()).collect();
    assert_eq!(
        classify_family(names.iter().map(|s| s.as_str())),
        Some(MoeFamily::Mixtral)
    );
    let msg = moe_failloud_report(names.iter().map(|s| s.as_str()));
    assert!(msg.contains("Family: Mixtral"));
    assert!(msg.contains("Productive support not enabled"));
}
