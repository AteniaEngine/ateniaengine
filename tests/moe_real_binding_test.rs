//! **MOE-10** — integration tests for real expert tensor binding.
//!
//! These build small synthetic safetensors buffers with *real* Mixtral /
//! Qwen-MoE tensor names, open them with the production `SafetensorsReader`,
//! resolve the real tensor bytes through the MOE-10 binding, construct real
//! `MoeDenseExpert`s, and run the MOE-4 sparse forward over them. No models
//! are downloaded; no full model is loaded; the MOE-2 loader fail-loud guard
//! is untouched (and re-asserted here).

use std::collections::HashMap;

use atenia_engine::moe::{
    build_real_layer, detect_moe, MoeWeightMap, RealExpertTensorBinding,
};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

/// Deterministic xorshift so tensor *values* are reproducible across runs.
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

/// Serialize named F32 tensors (with real values) into a safetensors buffer
/// and open it with the production reader.
fn reader_with(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> SafetensorsReader {
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
        views.insert((*name).to_string(), v);
    }
    let buffer = safetensors::serialize(&views, &None).unwrap();
    SafetensorsReader::from_bytes(buffer).unwrap()
}

/// Build a Mixtral-named reader for exactly 2 experts (8/16 dims).
fn mixtral_reader_2() -> SafetensorsReader {
    let d_model = 8usize;
    let d_ff = 16usize;
    reader_with(&[
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![d_model, d_model],
            seeded(1, d_model * d_model),
        ),
        (
            "model.layers.0.block_sparse_moe.gate.weight",
            vec![2, d_model],
            seeded(2, 2 * d_model),
        ),
        (
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            vec![d_ff, d_model],
            seeded(1001, d_ff * d_model),
        ),
        (
            "model.layers.0.block_sparse_moe.experts.0.w3.weight",
            vec![d_ff, d_model],
            seeded(1002, d_ff * d_model),
        ),
        (
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
            vec![d_model, d_ff],
            seeded(1003, d_model * d_ff),
        ),
        (
            "model.layers.0.block_sparse_moe.experts.1.w1.weight",
            vec![d_ff, d_model],
            seeded(1011, d_ff * d_model),
        ),
        (
            "model.layers.0.block_sparse_moe.experts.1.w3.weight",
            vec![d_ff, d_model],
            seeded(1012, d_ff * d_model),
        ),
        (
            "model.layers.0.block_sparse_moe.experts.1.w2.weight",
            vec![d_model, d_ff],
            seeded(1013, d_model * d_ff),
        ),
    ])
}

/// Wire a reader to a MOE-10 byte resolver (decode F32 → Vec<f32>).
fn resolver(reader: &SafetensorsReader) -> impl Fn(&str) -> Option<Vec<f32>> + '_ {
    move |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok())
}

fn weight_map(reader: &SafetensorsReader) -> MoeWeightMap {
    MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())))
}

#[test]
fn real_expert_tensor_resolution() {
    let reader = mixtral_reader_2();
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let tensors = map.expert(0, 0).expect("expert (0,0) present");
    let binding = RealExpertTensorBinding::resolve(0, 0, tensors, &resolve).unwrap();
    // Bytes resolved match the on-disk tensor exactly.
    let on_disk = reader
        .get("model.layers.0.block_sparse_moe.experts.0.w1.weight")
        .unwrap()
        .to_vec_f32()
        .unwrap();
    assert_eq!(binding.expert.w_gate, on_disk);
}

#[test]
fn real_expert_construction() {
    let reader = mixtral_reader_2();
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let tensors = map.expert(0, 1).unwrap();
    let binding = RealExpertTensorBinding::resolve(0, 1, tensors, &resolve).unwrap();
    assert_eq!(binding.shape.d_model, 8);
    assert_eq!(binding.shape.d_ff, 16);
    assert_eq!(binding.expert.w_gate.len(), 16 * 8);
    assert_eq!(binding.expert.w_up.len(), 16 * 8);
    assert_eq!(binding.expert.w_down.len(), 8 * 16);
}

#[test]
fn real_expert_forward() {
    let reader = mixtral_reader_2();
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let tensors = map.expert(0, 0).unwrap();
    let binding = RealExpertTensorBinding::resolve(0, 0, tensors, &resolve).unwrap();
    let x = seeded(77, binding.shape.d_model);
    let y = binding.expert.forward(&x).unwrap();
    assert_eq!(y.len(), binding.shape.d_model);
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn sparse_forward_with_real_experts() {
    let reader = mixtral_reader_2();
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let layer = build_real_layer(&map, 0, 2, &resolve).unwrap();
    assert_eq!(layer.num_experts(), 2);
    let x = seeded(88, layer.d_model);
    let sparse = layer.forward_sparse(&x, 2).unwrap();
    assert_eq!(sparse.selected_experts.len(), 2);
    // Real-weight sparse forward equals the dense-restricted oracle.
    let oracle = layer
        .forward_dense_restricted(&x, &sparse.selected_experts)
        .unwrap();
    for d in 0..layer.d_model {
        assert!((sparse.output[d] - oracle[d]).abs() < 1e-5);
    }
}

#[test]
fn expert_registry_resolves_real_tensors() {
    // Qwen-MoE-named layer: registry → binding resolves real gate/up/down.
    let d_model = 4usize;
    let d_ff = 6usize;
    let reader = reader_with(&[
        (
            "model.layers.0.mlp.gate.weight",
            vec![2, d_model],
            seeded(3, 2 * d_model),
        ),
        (
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            vec![d_ff, d_model],
            seeded(2001, d_ff * d_model),
        ),
        (
            "model.layers.0.mlp.experts.0.up_proj.weight",
            vec![d_ff, d_model],
            seeded(2002, d_ff * d_model),
        ),
        (
            "model.layers.0.mlp.experts.0.down_proj.weight",
            vec![d_model, d_ff],
            seeded(2003, d_model * d_ff),
        ),
        (
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            vec![d_ff, d_model],
            seeded(2011, d_ff * d_model),
        ),
        (
            "model.layers.0.mlp.experts.1.up_proj.weight",
            vec![d_ff, d_model],
            seeded(2012, d_ff * d_model),
        ),
        (
            "model.layers.0.mlp.experts.1.down_proj.weight",
            vec![d_model, d_ff],
            seeded(2013, d_model * d_ff),
        ),
    ]);
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    assert_eq!(map.num_experts(0), 2);
    let layer = build_real_layer(&map, 0, 2, &resolve).unwrap();
    assert_eq!(layer.d_model, d_model);
    assert_eq!(layer.d_ff, d_ff);
    let x = seeded(99, d_model);
    let out = layer.forward_sparse(&x, 2).unwrap();
    assert!(out.output.iter().all(|v| v.is_finite()));
}

#[test]
fn fail_loud_still_active() {
    // The MOE-2 detection that drives the loader fail-loud still fires:
    // binding real experts in isolation does NOT lift the guard.
    let reader = mixtral_reader_2();
    let det = detect_moe(reader.iter().map(|e| e.name));
    assert!(det.is_moe, "MoE checkpoint must still be detected (fail-loud preserved)");
}

#[test]
fn dense_models_still_load() {
    // A dense checkpoint: no MoE detection, empty map, no binding possible.
    let reader = reader_with(&[
        ("model.embed_tokens.weight", vec![16, 4], seeded(1, 64)),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![4, 4],
            seeded(2, 16),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![8, 4],
            seeded(3, 32),
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![8, 4],
            seeded(4, 32),
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![4, 8],
            seeded(5, 32),
        ),
        ("lm_head.weight", vec![16, 4], seeded(6, 64)),
    ]);
    assert!(!detect_moe(reader.iter().map(|e| e.name)).is_moe);
    let map = weight_map(&reader);
    assert!(map.is_empty(), "dense checkpoint must produce no MoE map");
}
