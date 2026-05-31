//! **MOE-FULL-4** — integration test: the experimental
//! `MoeRealLayerReference` graph op runs a whole certified `RealMoeLayer`
//! (router + routed experts + optional shared, MOE-11) through the real AMG
//! `Graph`, and matches `RealMoeLayer::forward_auto`.
//!
//! Scope: a single fused op, CPU-only. The MoE layer is assembled from the
//! committed real Mixtral layer-0 fixture (`fixtures/moe/mixtral_layer0.*`,
//! ~385 KB, packed experts, no shared) — a real checkpoint slice, not
//! synthetic. No model download, no scheduler change, no generation, no
//! production MoE path. The loader fail-loud guard is untouched.

use std::path::PathBuf;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::moe::{
    detect_moe, register_real_moe_layer, MoeExecutionConvention, MoeLayerConfig, MoeWeightMap,
    NumericalMetrics, RealMoeLayer,
};
use atenia_engine::tensor::Tensor;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn read_json(model: &str) -> serde_json::Value {
    let text = std::fs::read_to_string(fixture_dir().join(format!("{model}_layer0.json"))).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn f32_vec(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

/// Assemble the real Mixtral layer-0 from the committed fixture.
fn mixtral_real_layer() -> (RealMoeLayer, Vec<f32>, Vec<f32>) {
    let j = read_json("mixtral");
    let cfg = MoeLayerConfig::new(
        j["num_experts"].as_u64().unwrap() as usize,
        j["experts_per_token"].as_u64().unwrap() as usize,
        j["has_shared"].as_bool().unwrap(),
        j["d_model"].as_u64().unwrap() as usize,
        j["d_ff"].as_u64().unwrap() as usize,
    )
    .unwrap();
    let reader =
        SafetensorsReader::open(&fixture_dir().join("mixtral_layer0.safetensors")).unwrap();
    let map = MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())));
    let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    let input = f32_vec(&j["input"]);
    let hf_ref = f32_vec(&j["hf_ref"]);
    (layer, input, hf_ref)
}

/// Run input → MoeRealLayerReference → output through the AMG graph.
fn run_through_graph(layer_id: u32, input: &[f32], d_model: usize) -> Vec<f32> {
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let moe_id = gb.moe_real_layer_reference(x_id, layer_id);
    gb.output(moe_id);
    let mut graph = gb.build();
    let t = Tensor::new_cpu(vec![d_model], input.to_vec());
    let outputs = graph.execute(vec![t]).expect("graph exec");
    outputs[0].as_cpu_slice().to_vec()
}

#[test]
fn graph_real_moe_layer_matches_reference() {
    let (layer, input, hf_ref) = mixtral_real_layer();
    let d_model = layer.config.d_model;
    let reference = layer.forward_auto(&input).unwrap();
    let id = register_real_moe_layer(layer);

    let got = run_through_graph(id, &input, d_model);

    // Graph op must equal the certified RealMoeLayer::forward_auto.
    let m = NumericalMetrics::compute(&got, &reference).unwrap();
    assert!(m.max_abs_diff < 1e-5, "graph vs forward_auto: {m:?}");

    // And, transitively, it stays close to the MOE-16 HuggingFace reference
    // (Mixtral → Atenia convention; ~1e-10 in MIXTRAL-CERT-1).
    let mhf = NumericalMetrics::compute(&got, &hf_ref).unwrap();
    assert!(mhf.argmax_match && mhf.max_abs_diff < 0.5, "graph vs HF: {mhf:?}");
}

#[test]
fn graph_real_moe_layer_is_deterministic() {
    let (layer, input, _hf) = mixtral_real_layer();
    let d_model = layer.config.d_model;
    let id = register_real_moe_layer(layer);
    let a = run_through_graph(id, &input, d_model);
    let b = run_through_graph(id, &input, d_model);
    assert_eq!(a, b);
    assert!(a.iter().all(|v| v.is_finite()));
}

#[test]
#[should_panic(expected = "unknown layer_id")]
fn graph_real_moe_layer_rejects_unknown_layer() {
    // An unregistered layer id must fail loudly when executed.
    let input = vec![0.0_f32; 64];
    let _ = run_through_graph(u32::MAX, &input, 64);
}

#[test]
#[should_panic]
fn graph_real_moe_layer_rejects_bad_input_dim() {
    // Wrong input dimension (8 vs the layer's d_model=64) must fail.
    let (layer, _input, _hf) = mixtral_real_layer();
    let id = register_real_moe_layer(layer);
    let _ = run_through_graph(id, &[0.0_f32; 8], 8);
}

#[test]
fn graph_real_moe_layer_with_explicit_convention() {
    // The registry's `_with` path honours an explicit convention; for Mixtral
    // (no shared, renormalise) Atenia == forward_auto.
    let (layer, input, _hf) = mixtral_real_layer();
    let direct = layer.forward_auto(&input).unwrap();
    let id = register_real_moe_layer(layer);
    let via = atenia_engine::moe::execute_real_moe_layer_with(
        id,
        &input,
        MoeExecutionConvention::Atenia,
    )
    .unwrap();
    assert_eq!(via, direct);
}

#[test]
fn fail_loud_still_active() {
    // The fixture is real MoE tensors; detection still fires (the productive
    // loader guard is unchanged by this graph op).
    let reader =
        SafetensorsReader::open(&fixture_dir().join("mixtral_layer0.safetensors")).unwrap();
    assert!(detect_moe(reader.iter().map(|e| e.name)).is_moe);
}

#[test]
fn dense_models_still_load() {
    let dense = vec![
        ("model.embed_tokens.weight", vec![16usize, 4]),
        ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
    ];
    let names: Vec<&str> = dense.iter().map(|(n, _)| *n).collect();
    assert!(!detect_moe(names.into_iter()).is_moe);
    assert!(MoeWeightMap::from_tensors(dense.iter().map(|(n, s)| (*n, s.clone()))).is_empty());
}
