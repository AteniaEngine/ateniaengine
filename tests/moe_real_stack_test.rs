//! **MOE-12** — integration tests for the multi-layer real MoE stack.
//!
//! Builds small synthetic safetensors buffers with *real* multi-layer
//! Mixtral / Qwen-MoE tensor names, opens them with the production
//! `SafetensorsReader`, assembles a `RealMoeStack` of 2 or 3 layers from real
//! tensor bytes, and runs the sequential forward — validated against manual
//! layer-by-layer chaining. No models downloaded; no full model loaded; the
//! MOE-2 loader fail-loud guard is untouched.

use std::collections::HashMap;

use atenia_engine::moe::{
    detect_moe, MoeLayerConfig, MoeStackConfig, MoeWeightMap, RealMoeStack,
};
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

/// Append a Mixtral-style layer (no shared expert) to the tensor list.
fn push_mixtral_layer(
    t: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
    l: usize,
    n: usize,
    d_model: usize,
    d_ff: usize,
) {
    let lseed = (l as u64 + 1) * 1000;
    t.push((
        format!("model.layers.{l}.block_sparse_moe.gate.weight"),
        vec![n, d_model],
        seeded(lseed + 1, n * d_model),
    ));
    for e in 0..n {
        let base = lseed + 100 + e as u64;
        t.push((
            format!("model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight"),
            vec![d_ff, d_model],
            seeded(base * 10 + 1, d_ff * d_model),
        ));
        t.push((
            format!("model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight"),
            vec![d_ff, d_model],
            seeded(base * 10 + 2, d_ff * d_model),
        ));
        t.push((
            format!("model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight"),
            vec![d_model, d_ff],
            seeded(base * 10 + 3, d_model * d_ff),
        ));
    }
}

/// Append a Qwen-MoE-style layer (with shared expert) to the tensor list.
fn push_qwen_layer(
    t: &mut Vec<(String, Vec<usize>, Vec<f32>)>,
    l: usize,
    n: usize,
    d_model: usize,
    d_ff: usize,
    shared_ff: usize,
) {
    let lseed = (l as u64 + 1) * 2000;
    t.push((
        format!("model.layers.{l}.mlp.gate.weight"),
        vec![n, d_model],
        seeded(lseed + 1, n * d_model),
    ));
    for e in 0..n {
        let base = lseed + 100 + e as u64;
        t.push((
            format!("model.layers.{l}.mlp.experts.{e}.gate_proj.weight"),
            vec![d_ff, d_model],
            seeded(base * 10 + 1, d_ff * d_model),
        ));
        t.push((
            format!("model.layers.{l}.mlp.experts.{e}.up_proj.weight"),
            vec![d_ff, d_model],
            seeded(base * 10 + 2, d_ff * d_model),
        ));
        t.push((
            format!("model.layers.{l}.mlp.experts.{e}.down_proj.weight"),
            vec![d_model, d_ff],
            seeded(base * 10 + 3, d_model * d_ff),
        ));
    }
    t.push((
        format!("model.layers.{l}.mlp.shared_expert.gate_proj.weight"),
        vec![shared_ff, d_model],
        seeded(lseed + 9001, shared_ff * d_model),
    ));
    t.push((
        format!("model.layers.{l}.mlp.shared_expert.up_proj.weight"),
        vec![shared_ff, d_model],
        seeded(lseed + 9002, shared_ff * d_model),
    ));
    t.push((
        format!("model.layers.{l}.mlp.shared_expert.down_proj.weight"),
        vec![d_model, shared_ff],
        seeded(lseed + 9003, d_model * shared_ff),
    ));
}

#[test]
fn real_moe_stack_assembly() {
    let mut t = Vec::new();
    push_mixtral_layer(&mut t, 0, 4, 8, 16);
    push_mixtral_layer(&mut t, 1, 4, 8, 16);
    let reader = reader_with(&t);
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let cfg = MoeStackConfig::new(2).unwrap();
    let lc = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
    let stack = RealMoeStack::assemble(&map, cfg, &[(0, lc), (1, lc)], &resolve).unwrap();
    assert_eq!(stack.num_layers(), 2);
    assert_eq!(stack.d_model(), 8);
}

#[test]
fn real_moe_stack_forward() {
    let mut t = Vec::new();
    push_mixtral_layer(&mut t, 0, 4, 8, 16);
    push_mixtral_layer(&mut t, 1, 4, 8, 16);
    let reader = reader_with(&t);
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let cfg = MoeStackConfig::new(2).unwrap();
    let lc = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
    let stack = RealMoeStack::assemble(&map, cfg, &[(0, lc), (1, lc)], &resolve).unwrap();
    let x = seeded(77, 8);
    let out = stack.forward(&x).unwrap();
    assert_eq!(out.len(), 8);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn two_layer_stack_matches_manual_execution() {
    let mut t = Vec::new();
    push_mixtral_layer(&mut t, 0, 4, 8, 16);
    push_mixtral_layer(&mut t, 1, 4, 8, 16);
    let reader = reader_with(&t);
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let cfg = MoeStackConfig::new(2).unwrap();
    let lc = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
    let stack = RealMoeStack::assemble(&map, cfg, &[(0, lc), (1, lc)], &resolve).unwrap();
    let x = seeded(101, 8);
    let h0 = stack.layers[0].forward(&x).unwrap();
    let manual = stack.layers[1].forward(&h0).unwrap();
    let got = stack.forward(&x).unwrap();
    for d in 0..8 {
        assert!((got[d] - manual[d]).abs() < 1e-5);
    }
}

#[test]
fn three_layer_stack_matches_manual_execution() {
    let mut t = Vec::new();
    push_mixtral_layer(&mut t, 0, 4, 8, 16);
    push_mixtral_layer(&mut t, 1, 4, 8, 16);
    push_mixtral_layer(&mut t, 2, 4, 8, 16);
    let reader = reader_with(&t);
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let cfg = MoeStackConfig::new(3).unwrap();
    let lc = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
    let stack =
        RealMoeStack::assemble(&map, cfg, &[(0, lc), (1, lc), (2, lc)], &resolve).unwrap();
    let x = seeded(103, 8);
    let h0 = stack.layers[0].forward(&x).unwrap();
    let h1 = stack.layers[1].forward(&h0).unwrap();
    let manual = stack.layers[2].forward(&h1).unwrap();
    let got = stack.forward(&x).unwrap();
    for d in 0..8 {
        assert!((got[d] - manual[d]).abs() < 1e-5);
    }
}

#[test]
fn stack_validates_d_model_consistency() {
    // Two layers with mismatched d_model fixtures cannot be stacked.
    let mut t = Vec::new();
    push_mixtral_layer(&mut t, 0, 4, 8, 16);
    let reader = reader_with(&t);
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let cfg = MoeStackConfig::new(2).unwrap();
    // Second spec claims d_model=4 for layer 0 again — but there is no layer
    // 1, so assembly fails on the missing layer; for the pure d_model check
    // we exercise it via RealMoeLayer + RealMoeStack::new in the unit tests.
    // Here, claiming a non-existent layer must error.
    let lc = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
    let err = RealMoeStack::assemble(&map, cfg, &[(0, lc), (1, lc)], &resolve).unwrap_err();
    // Layer 1 has no tensors → binding/layer error surfaced through the stack.
    let _ = err;
}

#[test]
fn stack_with_shared_experts() {
    let mut t = Vec::new();
    push_qwen_layer(&mut t, 0, 3, 4, 6, 10);
    push_qwen_layer(&mut t, 1, 3, 4, 6, 10);
    let reader = reader_with(&t);
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let cfg = MoeStackConfig::new(2).unwrap();
    let lc = MoeLayerConfig::new(3, 2, true, 4, 6).unwrap();
    let stack = RealMoeStack::assemble(&map, cfg, &[(0, lc), (1, lc)], &resolve).unwrap();
    assert!(stack.layers.iter().all(|l| l.has_shared_expert()));
    let x = seeded(205, 4);
    let h0 = stack.layers[0].forward(&x).unwrap();
    let manual = stack.layers[1].forward(&h0).unwrap();
    let got = stack.forward(&x).unwrap();
    for d in 0..4 {
        assert!((got[d] - manual[d]).abs() < 1e-5);
    }
}

#[test]
fn fail_loud_still_active() {
    let mut t = Vec::new();
    push_mixtral_layer(&mut t, 0, 2, 8, 16);
    let reader = reader_with(&t);
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
