//! **MOE-18** — automatic convention selection tests.
//!
//! Confirms the convention is resolved from metadata (the `shared_expert_gate`
//! signal) and that `forward_auto` reproduces the MOE-17 explicit-convention
//! results exactly: Qwen-MoE → `HuggingFaceQwen`, Mixtral → `Atenia`.
//! Fixture-based; no models, no HF at test time.

use std::path::PathBuf;

use atenia_engine::moe::{
    detect_moe, MoeConventionResolver, MoeExecutionConvention, MoeLayerConfig, MoeWeightMap,
    NumericalMetrics, RealMoeLayer,
};
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

fn weight_map(reader: &SafetensorsReader) -> MoeWeightMap {
    MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())))
}

/// (layer, map, input, atenia_ref, hf_ref)
fn load(model: &str) -> (RealMoeLayer, MoeWeightMap, Vec<f32>, Vec<f32>, Vec<f32>) {
    let j = read_json(model);
    let num_experts = j["num_experts"].as_u64().unwrap() as usize;
    let experts_per_token = j["experts_per_token"].as_u64().unwrap() as usize;
    let d_model = j["d_model"].as_u64().unwrap() as usize;
    let d_ff = j["d_ff"].as_u64().unwrap() as usize;
    let has_shared = j["has_shared"].as_bool().unwrap();
    let input = f32_vec(&j["input"]);
    let atenia_ref = f32_vec(&j["atenia_ref"]);
    let hf_ref = f32_vec(&j["hf_ref"]);

    let st = fixture_dir().join(format!("{model}_layer0.safetensors"));
    let reader = SafetensorsReader::open(&st).unwrap();
    let map = weight_map(&reader);
    let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());
    let cfg = MoeLayerConfig::new(num_experts, experts_per_token, has_shared, d_model, d_ff).unwrap();
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    (layer, map, input, atenia_ref, hf_ref)
}

#[test]
fn resolver_detects_qwen_convention() {
    for model in ["qwen15_moe", "qwen2_moe"] {
        let (layer, map, _, _, _) = load(model);
        assert_eq!(
            MoeConventionResolver::from_weight_map(&map),
            MoeExecutionConvention::HuggingFaceQwen,
            "{model} (metadata)"
        );
        assert_eq!(layer.resolve_convention(), MoeExecutionConvention::HuggingFaceQwen, "{model} (layer)");
    }
}

#[test]
fn resolver_detects_mixtral_convention() {
    let (layer, map, _, _, _) = load("mixtral");
    assert_eq!(MoeConventionResolver::from_weight_map(&map), MoeExecutionConvention::Atenia);
    assert_eq!(layer.resolve_convention(), MoeExecutionConvention::Atenia);
}

#[test]
fn auto_forward_matches_explicit_qwen() {
    for model in ["qwen15_moe", "qwen2_moe"] {
        let (layer, _, x, _, _) = load(model);
        let auto = layer.forward_auto(&x).unwrap();
        let explicit = layer.forward_with(&x, MoeExecutionConvention::HuggingFaceQwen).unwrap();
        assert_eq!(auto, explicit, "{model}: auto must equal explicit HF");
    }
}

#[test]
fn auto_forward_matches_explicit_mixtral() {
    let (layer, _, x, _, _) = load("mixtral");
    let auto = layer.forward_auto(&x).unwrap();
    let explicit = layer.forward_with(&x, MoeExecutionConvention::Atenia).unwrap();
    assert_eq!(auto, explicit, "mixtral: auto must equal explicit Atenia");
}

#[test]
fn metrics_preserved() {
    // forward_auto must hit the MOE-17 parity numbers vs the HF reference:
    // Qwen ~1e-10 (HF convention), Mixtral ~1e-10 (Atenia convention).
    for model in ["qwen15_moe", "qwen2_moe", "mixtral"] {
        let (layer, _, x, _, hf_ref) = load(model);
        let out = layer.forward_auto(&x).unwrap();
        let m = NumericalMetrics::compute(&out, &hf_ref).unwrap();
        assert!(
            m.argmax_match && m.max_abs_diff < 1e-6,
            "{model}: forward_auto must match HF parity: {:?}",
            m
        );
    }
}

#[test]
fn fail_loud_still_active() {
    for model in ["qwen15_moe", "qwen2_moe", "mixtral"] {
        let st = fixture_dir().join(format!("{model}_layer0.safetensors"));
        let reader = SafetensorsReader::open(&st).unwrap();
        assert!(detect_moe(reader.iter().map(|e| e.name)).is_moe);
    }
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
    let map = MoeWeightMap::from_tensors(dense.iter().map(|(n, s)| (*n, s.clone())));
    assert!(map.is_empty());
    // A dense (empty) map resolves to the Atenia default.
    assert_eq!(MoeConventionResolver::from_weight_map(&map), MoeExecutionConvention::Atenia);
}
