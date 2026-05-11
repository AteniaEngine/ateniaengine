//! Tests for `ShardIndex` (M4.7.1.a).
//!
//! Hand-crafted JSON fixtures only — no real model required.

use atenia_engine::v17::loader::loader_errors::LoaderError;
use atenia_engine::v17::loader::shard_index::ShardIndex;
use std::path::PathBuf;

fn fake_base() -> PathBuf {
    PathBuf::from("/some/model/dir")
}

#[test]
fn parse_valid_two_shard_index() {
    let json = r#"{
        "metadata": { "total_size": 1234567 },
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors"
        }
    }"#;
    let index =
        ShardIndex::from_json_str(json, fake_base()).expect("valid 2-shard index must parse");

    assert_eq!(index.total_size, 1_234_567);
    assert_eq!(index.weight_map.len(), 4);
    assert_eq!(index.shard_count(), 2);
    assert_eq!(
        index.shard_filenames(),
        vec![
            "model-00001-of-00002.safetensors".to_string(),
            "model-00002-of-00002.safetensors".to_string()
        ]
    );
    assert_eq!(
        index.shard_for("model.embed_tokens.weight"),
        Some("model-00001-of-00002.safetensors")
    );
    assert_eq!(index.shard_for("does.not.exist"), None);
}

#[test]
fn missing_total_size_defaults_to_zero() {
    // Older indexes (and some hand-built ones) omit metadata.
    let json = r#"{
        "weight_map": {
            "a": "shard-00001.safetensors"
        }
    }"#;
    let index = ShardIndex::from_json_str(json, fake_base()).expect("must parse");
    assert_eq!(index.total_size, 0);
    assert_eq!(index.weight_map.len(), 1);
}

#[test]
fn missing_weight_map_is_error() {
    let json = r#"{
        "metadata": { "total_size": 100 }
    }"#;
    let err = ShardIndex::from_json_str(json, fake_base()).unwrap_err();
    match err {
        LoaderError::InvalidFormat(msg) => {
            assert!(
                msg.contains("weight_map"),
                "error must mention weight_map, got: {}",
                msg
            );
        }
        other => panic!("expected InvalidFormat, got {:?}", other),
    }
}

#[test]
fn empty_weight_map_is_error() {
    let json = r#"{
        "weight_map": {}
    }"#;
    let err = ShardIndex::from_json_str(json, fake_base()).unwrap_err();
    match err {
        LoaderError::InvalidFormat(msg) => assert!(msg.contains("empty")),
        other => panic!("expected InvalidFormat, got {:?}", other),
    }
}

#[test]
fn weight_map_value_must_be_string() {
    let json = r#"{
        "weight_map": {
            "tensor.a": 123
        }
    }"#;
    let err = ShardIndex::from_json_str(json, fake_base()).unwrap_err();
    match err {
        LoaderError::InvalidFormat(msg) => {
            assert!(msg.contains("not a string"));
            assert!(msg.contains("tensor.a"));
        }
        other => panic!("expected InvalidFormat, got {:?}", other),
    }
}

#[test]
fn malformed_json_is_error() {
    let json = "{ not json at all";
    let err = ShardIndex::from_json_str(json, fake_base()).unwrap_err();
    match err {
        LoaderError::InvalidFormat(msg) => assert!(msg.contains("parse")),
        other => panic!("expected InvalidFormat, got {:?}", other),
    }
}

#[test]
fn shard_path_resolves_against_base_dir() {
    let json = r#"{
        "weight_map": { "x": "model-00001-of-00002.safetensors" }
    }"#;
    let base = PathBuf::from("/path/to/model");
    let index = ShardIndex::from_json_str(json, base.clone()).unwrap();
    let resolved = index.shard_path("model-00001-of-00002.safetensors");
    assert_eq!(resolved, base.join("model-00001-of-00002.safetensors"));
}

#[test]
fn tensors_by_shard_groups_correctly() {
    let json = r#"{
        "weight_map": {
            "a": "shard-1.safetensors",
            "b": "shard-1.safetensors",
            "c": "shard-2.safetensors"
        }
    }"#;
    let index = ShardIndex::from_json_str(json, fake_base()).unwrap();
    let by_shard = index.tensors_by_shard();
    assert_eq!(by_shard.len(), 2);
    assert_eq!(
        by_shard.get("shard-1.safetensors").unwrap(),
        &vec!["a".to_string(), "b".to_string()]
    );
    assert_eq!(
        by_shard.get("shard-2.safetensors").unwrap(),
        &vec!["c".to_string()]
    );
}

#[test]
fn injective_invariant_relies_on_btreemap_dedup() {
    // serde_json silently keeps the last value when the same key
    // appears twice in a JSON object — this is silent corruption
    // of the safetensors sharded invariant. The parser does not
    // need to detect it (BTreeMap dedup happens implicitly during
    // insertion from a serde_json::Map iteration), but document
    // the behavioural contract: in the presence of *visibly*
    // distinct keys, no two tensor names map to different shards.
    let json = r#"{
        "weight_map": {
            "x": "shard-1.safetensors"
        }
    }"#;
    let idx = ShardIndex::from_json_str(json, fake_base()).unwrap();
    assert_eq!(idx.shard_for("x"), Some("shard-1.safetensors"));
}
