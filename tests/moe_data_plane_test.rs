//! **MOE-9** — integration tests for the real MoE data plane on a real
//! `SafetensorsReader`. Builds synthetic safetensors buffers with real
//! Mixtral / Qwen-MoE tensor names, feeds the reader's `(name, shape)`
//! listing into `MoeWeightMap`, and confirms detection, mapping, registry
//! lookup, and that fail-loud + dense loading are unaffected.
//!
//! No real models downloaded; no model data executed.

use std::collections::HashMap;

use atenia_engine::moe::{detect_moe, MoeWeightMap};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

/// Serialize named (zeroed) tensors with the given shapes into a
/// safetensors buffer and open it as a reader. Only names + shapes matter.
fn reader_with(tensors: &[(&str, Vec<usize>)]) -> SafetensorsReader {
    // Keep each tensor's backing bytes alive for the borrow.
    let datas: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, shape)| {
            let numel: usize = shape.iter().product();
            vec![0u8; numel * 4] // F32
        })
        .collect();
    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (i, (name, shape)) in tensors.iter().enumerate() {
        let v = TensorView::new(StDtype::F32, shape.clone(), &datas[i]).unwrap();
        views.insert((*name).to_string(), v);
    }
    let buffer = safetensors::serialize(&views, &None).unwrap();
    SafetensorsReader::from_bytes(buffer).unwrap()
}

fn weight_map(reader: &SafetensorsReader) -> MoeWeightMap {
    MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())))
}

#[test]
fn graph_can_access_registered_expert() {
    // Mixtral-style, 2 experts. Use small shapes so the buffer is tiny.
    let reader = reader_with(&[
        ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
        ("model.layers.0.block_sparse_moe.gate.weight", vec![2, 4]),
        ("model.layers.0.block_sparse_moe.experts.0.w1.weight", vec![8, 4]),
        ("model.layers.0.block_sparse_moe.experts.0.w3.weight", vec![8, 4]),
        ("model.layers.0.block_sparse_moe.experts.0.w2.weight", vec![4, 8]),
        ("model.layers.0.block_sparse_moe.experts.1.w1.weight", vec![8, 4]),
        ("model.layers.0.block_sparse_moe.experts.1.w3.weight", vec![8, 4]),
        ("model.layers.0.block_sparse_moe.experts.1.w2.weight", vec![4, 8]),
    ]);
    let map = weight_map(&reader);
    assert_eq!(map.num_experts(0), 2);
    // The "graph layer" can look up a specific expert's projections.
    let e1 = map.expert(0, 1).expect("expert (0,1) must be registered");
    assert!(e1.is_complete());
    assert_eq!(e1.gate.as_ref().unwrap().shape, vec![8, 4]);
    assert_eq!(e1.down.as_ref().unwrap().shape, vec![4, 8]);
    assert!(map.router_weight(0).is_some());
}

#[test]
fn qwen_moe_data_plane_via_reader() {
    let reader = reader_with(&[
        ("model.layers.0.mlp.gate.weight", vec![4, 4]),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", vec![6, 4]),
        ("model.layers.0.mlp.experts.0.up_proj.weight", vec![6, 4]),
        ("model.layers.0.mlp.experts.0.down_proj.weight", vec![4, 6]),
        ("model.layers.0.mlp.shared_expert.gate_proj.weight", vec![8, 4]),
    ]);
    let map = weight_map(&reader);
    assert_eq!(map.num_experts(0), 1);
    assert!(map.expert(0, 0).unwrap().is_complete());
    assert_eq!(map.layer(0).unwrap().shared.len(), 1);
}

#[test]
fn fail_loud_still_active() {
    // The MOE-2 detection (which drives the loader fail-loud) still fires
    // on a real MoE reader.
    let reader = reader_with(&[
        ("model.layers.0.block_sparse_moe.gate.weight", vec![2, 4]),
        ("model.layers.0.block_sparse_moe.experts.0.w1.weight", vec![8, 4]),
    ]);
    let det = detect_moe(reader.iter().map(|e| e.name));
    assert!(det.is_moe, "MoE checkpoint must still be detected (fail-loud preserved)");
}

#[test]
fn dense_models_still_load() {
    // A dense reader: no MoE detection, empty MoE map.
    let reader = reader_with(&[
        ("model.embed_tokens.weight", vec![16, 4]),
        ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
        ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
        ("lm_head.weight", vec![16, 4]),
    ]);
    assert!(!detect_moe(reader.iter().map(|e| e.name)).is_moe);
    let map = weight_map(&reader);
    assert!(map.is_empty(), "dense checkpoint must produce no MoE map");
}
