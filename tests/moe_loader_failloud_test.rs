//! **MOE-2** — integration test for MoE fail-loud detection on a real
//! `SafetensorsReader`.
//!
//! Builds tiny synthetic safetensors buffers (no real model) and feeds
//! their tensor names — as the loader's fail-loud guard does — through
//! `atenia_engine::moe::detect_moe`. Confirms a MoE checkpoint is detected
//! (so the loader returns `LoaderError::MoeUnsupported`) and a dense
//! checkpoint is not.

use std::collections::HashMap;

use atenia_engine::moe::{detect_moe, unsupported_message};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

/// Serialize a set of named F32 scalars into a safetensors buffer and open
/// it as a `SafetensorsReader`. Values are irrelevant — only names matter.
fn reader_with_names(names: &[&str]) -> SafetensorsReader {
    let payload = [0u8; 4]; // one F32 = 4 bytes
    // TensorView borrows the data; keep one buffer alive per tensor.
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
fn mixtral_moe_checkpoint_is_detected_via_reader() {
    let reader = reader_with_names(&[
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.block_sparse_moe.experts.0.w2.weight",
        "model.layers.0.block_sparse_moe.experts.0.w3.weight",
        "model.layers.0.block_sparse_moe.experts.7.w1.weight",
    ]);
    let det = detect_moe(reader.iter().map(|e| e.name));
    assert!(det.is_moe, "Mixtral-style checkpoint must be detected as MoE");
    assert_eq!(det.implied_expert_count(), Some(8));
    let msg = unsupported_message(&det);
    assert!(msg.contains("MoE checkpoint detected"));
    assert!(msg.contains("not implemented yet"));
}

#[test]
fn qwen_moe_with_shared_expert_is_detected_via_reader() {
    let reader = reader_with_names(&[
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
    ]);
    let det = detect_moe(reader.iter().map(|e| e.name));
    assert!(det.is_moe);
    assert!(det.shared_expert_tensor_count >= 1);
}

#[test]
fn dense_checkpoint_is_not_detected_as_moe_via_reader() {
    // Dense Llama / DeepSeek-distill style — no experts.
    let reader = reader_with_names(&[
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
    ]);
    let det = detect_moe(reader.iter().map(|e| e.name));
    assert!(!det.is_moe, "dense checkpoint must NOT be detected as MoE");
    assert_eq!(det.expert_tensor_count, 0);
}
