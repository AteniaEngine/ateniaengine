//! **MOE-11** — integration tests for real MoE layer assembly.
//!
//! Builds small synthetic safetensors buffers with *real* Mixtral /
//! Qwen-MoE tensor names (including a shared expert), opens them with the
//! production `SafetensorsReader`, assembles a complete `RealMoeLayer`
//! (router + routed experts + optional shared expert) from real tensor
//! bytes, and runs the full-layer forward. The assembled layer is validated
//! against the MOE-4 sparse reference path. No models are downloaded; no full
//! model is loaded; the MOE-2 loader fail-loud guard is untouched.

use std::collections::HashMap;

use atenia_engine::moe::{detect_moe, MoeLayerConfig, MoeWeightMap, RealMoeLayer};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

fn seeded(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = (state >> 11) as u32;
        out.push((u as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    out
}

/// Serialize named F32 tensors (real values) into a safetensors buffer and
/// open it with the production reader.
fn reader_with(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> SafetensorsReader {
    let datas: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, vals)| {
            let mut bytes = Vec::with_capacity(vals.len() * 4);
            for v in vals {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes
        })
        .collect();
    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (i, (name, shape, _)) in tensors.iter().enumerate() {
        let v = TensorView::new(StDtype::F32, shape.clone(), &datas[i]).unwrap();
        views.insert(name.clone(), v);
    }
    let buffer = safetensors::serialize(&views, &None).unwrap();
    SafetensorsReader::from_bytes(buffer).unwrap()
}

fn resolver(reader: &SafetensorsReader) -> impl Fn(&str) -> Option<Vec<f32>> + '_ {
    move |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok())
}

fn weight_map(reader: &SafetensorsReader) -> MoeWeightMap {
    MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())))
}

/// A Mixtral-style single layer (no shared expert).
fn mixtral_layer(n: usize, d_model: usize, d_ff: usize) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    t.push((
        "model.layers.0.block_sparse_moe.gate.weight".to_string(),
        vec![n, d_model],
        seeded(2, n * d_model),
    ));
    for e in 0..n {
        let base = 100 + e as u64;
        t.push((
            format!("model.layers.0.block_sparse_moe.experts.{e}.w1.weight"),
            vec![d_ff, d_model],
            seeded(base * 10 + 1, d_ff * d_model),
        ));
        t.push((
            format!("model.layers.0.block_sparse_moe.experts.{e}.w3.weight"),
            vec![d_ff, d_model],
            seeded(base * 10 + 2, d_ff * d_model),
        ));
        t.push((
            format!("model.layers.0.block_sparse_moe.experts.{e}.w2.weight"),
            vec![d_model, d_ff],
            seeded(base * 10 + 3, d_model * d_ff),
        ));
    }
    t
}

/// A Qwen-MoE-style single layer with a shared expert (own d_ff).
fn qwen_layer(
    n: usize,
    d_model: usize,
    d_ff: usize,
    shared_ff: usize,
) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    t.push((
        "model.layers.0.mlp.gate.weight".to_string(),
        vec![n, d_model],
        seeded(3, n * d_model),
    ));
    for e in 0..n {
        let base = 200 + e as u64;
        t.push((
            format!("model.layers.0.mlp.experts.{e}.gate_proj.weight"),
            vec![d_ff, d_model],
            seeded(base * 10 + 1, d_ff * d_model),
        ));
        t.push((
            format!("model.layers.0.mlp.experts.{e}.up_proj.weight"),
            vec![d_ff, d_model],
            seeded(base * 10 + 2, d_ff * d_model),
        ));
        t.push((
            format!("model.layers.0.mlp.experts.{e}.down_proj.weight"),
            vec![d_model, d_ff],
            seeded(base * 10 + 3, d_model * d_ff),
        ));
    }
    t.push((
        "model.layers.0.mlp.shared_expert.gate_proj.weight".to_string(),
        vec![shared_ff, d_model],
        seeded(9001, shared_ff * d_model),
    ));
    t.push((
        "model.layers.0.mlp.shared_expert.up_proj.weight".to_string(),
        vec![shared_ff, d_model],
        seeded(9002, shared_ff * d_model),
    ));
    t.push((
        "model.layers.0.mlp.shared_expert.down_proj.weight".to_string(),
        vec![d_model, shared_ff],
        seeded(9003, d_model * shared_ff),
    ));
    t
}

#[test]
fn real_moe_layer_assembly() {
    let reader = reader_with(&mixtral_layer(4, 8, 16));
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let cfg = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    assert_eq!(layer.num_experts(), 4);
    assert!(!layer.has_shared_expert());
    assert_eq!(layer.routed.d_model, 8);
    assert_eq!(layer.routed.d_ff, 16);
}

#[test]
fn real_moe_layer_forward() {
    let reader = reader_with(&mixtral_layer(4, 8, 16));
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let cfg = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    let x = seeded(77, 8);
    let out = layer.forward(&x).unwrap();
    assert_eq!(out.len(), 8);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn real_moe_layer_with_shared_expert() {
    let reader = reader_with(&qwen_layer(3, 4, 6, 10));
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let cfg = MoeLayerConfig::new(3, 2, true, 4, 6).unwrap();
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    assert!(layer.has_shared_expert());
    let se = layer.shared.as_ref().unwrap();
    assert_eq!(se.d_model, 4);
    assert_eq!(se.d_ff, 10);
    // Full forward = routed + shared.
    let x = seeded(88, 4);
    let full = layer.forward(&x).unwrap();
    let routed = layer.forward_routed(&x).unwrap();
    let shared = se.forward(&x).unwrap();
    for d in 0..4 {
        assert!((full[d] - (routed[d] + shared[d])).abs() < 1e-5);
    }
}

#[test]
fn router_and_expert_counts_match_config() {
    let reader = reader_with(&mixtral_layer(4, 8, 16));
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    // Correct config assembles; wrong expert count is rejected.
    let ok = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
    let layer = RealMoeLayer::assemble(&map, 0, ok, &resolve).unwrap();
    assert_eq!(layer.num_experts(), ok.num_experts);
    assert_eq!(layer.routed.w_router.len(), ok.num_experts * ok.d_model);

    let wrong = MoeLayerConfig::new(8, 2, false, 8, 16).unwrap();
    assert!(RealMoeLayer::assemble(&map, 0, wrong, &resolve).is_err());
}

#[test]
fn sparse_forward_matches_assembled_layer() {
    // With no shared expert, the assembled-layer forward must equal the
    // MOE-4 sparse reference path over the same experts, within 1e-5.
    let reader = reader_with(&mixtral_layer(4, 8, 16));
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let cfg = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    let x = seeded(99, 8);
    let assembled = layer.forward(&x).unwrap();
    let reference = layer.routed.forward_sparse(&x, 2).unwrap().output;
    for d in 0..8 {
        assert!((assembled[d] - reference[d]).abs() < 1e-5);
    }
}

#[test]
fn fail_loud_still_active() {
    // Assembling a layer in isolation does NOT lift the MOE-2 guard.
    let reader = reader_with(&mixtral_layer(2, 8, 16));
    let det = detect_moe(reader.iter().map(|e| e.name));
    assert!(det.is_moe, "MoE checkpoint must still be detected (fail-loud preserved)");
}

#[test]
fn dense_models_still_load() {
    let reader = reader_with(&[
        ("model.embed_tokens.weight".to_string(), vec![16, 4], seeded(1, 64)),
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![4, 4],
            seeded(2, 16),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            vec![8, 4],
            seeded(3, 32),
        ),
        (
            "model.layers.0.mlp.up_proj.weight".to_string(),
            vec![8, 4],
            seeded(4, 32),
        ),
        (
            "model.layers.0.mlp.down_proj.weight".to_string(),
            vec![4, 8],
            seeded(5, 32),
        ),
        ("lm_head.weight".to_string(), vec![16, 4], seeded(6, 64)),
    ]);
    assert!(!detect_moe(reader.iter().map(|e| e.name)).is_moe);
    let map = weight_map(&reader);
    assert!(map.is_empty(), "dense checkpoint must produce no MoE map");
}
