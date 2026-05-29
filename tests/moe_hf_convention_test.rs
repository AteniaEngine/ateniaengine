//! **MOE-17** — HF convention parity tests.
//!
//! Verifies the optional `HuggingFaceQwen` execution convention (no top-k
//! renormalisation + sigmoid-gated shared expert) drives Atenia's Qwen-MoE
//! output to the HF transformers reference, while the `Atenia` default is
//! unchanged and Mixtral still matches. Fixture-based; no models, no HF at
//! test time.

use std::path::PathBuf;

use atenia_engine::moe::{
    detect_moe, MoeDenseExpert, MoeDenseLayer, MoeExecutionConvention, MoeLayerConfig,
    MoeWeightMap, NumericalMetrics, RealMoeLayer,
};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn read_json(model: &str) -> serde_json::Value {
    let path = fixture_dir().join(format!("{model}_layer0.json"));
    let text = std::fs::read_to_string(&path).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn f32_vec(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

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

/// Build the layer-0 RealMoeLayer from a fixture, returning (layer, input,
/// atenia_ref, hf_ref).
fn load_fixture_layer(model: &str) -> (RealMoeLayer, Vec<f32>, Vec<f32>, Vec<f32>) {
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
    let map = MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())));
    let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());
    let cfg = MoeLayerConfig::new(num_experts, experts_per_token, has_shared, d_model, d_ff).unwrap();
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    (layer, input, atenia_ref, hf_ref)
}

/// A synthetic single-layer RealMoeLayer with an optional shared expert +
/// gate, built from public constructors (no fixture).
fn synthetic_layer(with_shared: bool, gate_seed: u64) -> (RealMoeLayer, Vec<f32>) {
    let (d_model, d_ff, ne, k) = (4usize, 8usize, 4usize, 2usize);
    let experts: Vec<MoeDenseExpert> = (0..ne)
        .map(|e| {
            MoeDenseExpert::new(
                d_model,
                d_ff,
                seeded(100 + e as u64, d_ff * d_model),
                seeded(200 + e as u64, d_ff * d_model),
                seeded(300 + e as u64, d_model * d_ff),
            )
            .unwrap()
        })
        .collect();
    let routed = MoeDenseLayer::new(d_model, d_ff, seeded(7, ne * d_model), experts, k).unwrap();
    let shared = if with_shared {
        Some(MoeDenseExpert::new(d_model, d_ff, seeded(11, d_ff * d_model), seeded(12, d_ff * d_model), seeded(13, d_model * d_ff)).unwrap())
    } else {
        None
    };
    let shared_gate = if with_shared { Some(seeded(gate_seed, d_model)) } else { None };
    let cfg = MoeLayerConfig::new(ne, k, with_shared, d_model, d_ff).unwrap();
    let layer = RealMoeLayer { config: cfg, routed, shared, shared_gate };
    (layer, seeded(999, d_model))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    NumericalMetrics::compute(a, b).unwrap().max_abs_diff
}

#[test]
fn topk_prob_mode_changes_output() {
    // No shared expert: the only difference between conventions is top-k
    // renormalisation, which must change the output.
    let (layer, x) = synthetic_layer(false, 0);
    let atenia = layer.forward_with(&x, MoeExecutionConvention::Atenia).unwrap();
    let hf = layer.forward_with(&x, MoeExecutionConvention::HuggingFaceQwen).unwrap();
    assert!(max_abs_diff(&atenia, &hf) > 1e-3, "renormalisation must change the output");
}

#[test]
fn shared_gate_changes_output() {
    // Two different shared-gate weights must yield different HF-convention
    // outputs; and the Atenia convention must ignore the gate entirely.
    let (l1, x) = synthetic_layer(true, 21);
    let (l2, _) = synthetic_layer(true, 22); // different gate seed only
    let hf1 = l1.forward_with(&x, MoeExecutionConvention::HuggingFaceQwen).unwrap();
    let hf2 = l2.forward_with(&x, MoeExecutionConvention::HuggingFaceQwen).unwrap();
    assert!(max_abs_diff(&hf1, &hf2) > 1e-4, "changing the shared gate must change HF output");
    // Atenia convention ignores the gate: identical for both gates.
    let a1 = l1.forward_with(&x, MoeExecutionConvention::Atenia).unwrap();
    let a2 = l2.forward_with(&x, MoeExecutionConvention::Atenia).unwrap();
    assert!(max_abs_diff(&a1, &a2) < 1e-9, "Atenia convention must ignore the shared gate");
}

#[test]
fn atenia_mode_preserves_existing_results() {
    // The default forward (Atenia convention) still matches the MOE-16
    // primary reference for all three models — behaviour unchanged.
    for model in ["qwen15_moe", "qwen2_moe", "mixtral"] {
        let (layer, x, atenia_ref, _hf) = load_fixture_layer(model);
        let out = layer.forward(&x).unwrap();
        let m = NumericalMetrics::compute(&out, &atenia_ref).unwrap();
        assert!(m.passes(0.5) && m.max_abs_diff < 1e-4, "{model}: {:?}", m);
    }
}

#[test]
fn hf_qwen_mode_matches_reference() {
    // Under the HF convention, the Qwen-MoE output must match the HF
    // transformers reference (was ~3e-4 under the Atenia default in MOE-16).
    for model in ["qwen15_moe", "qwen2_moe"] {
        let (layer, x, _atenia, hf_ref) = load_fixture_layer(model);
        let out = layer.forward_with(&x, MoeExecutionConvention::HuggingFaceQwen).unwrap();
        let m = NumericalMetrics::compute(&out, &hf_ref).unwrap();
        println!("{model} HF-mode vs HF ref: {:?}", m);
        assert!(m.argmax_match && m.max_abs_diff < 1e-3, "{model} HF-mode must match HF ref: {:?}", m);
    }
}

#[test]
fn mixtral_still_matches_reference() {
    // Mixtral has no shared expert and renormalises, so the Atenia default
    // already equals HF — unchanged by MOE-17.
    let (layer, x, _atenia, hf_ref) = load_fixture_layer("mixtral");
    let out = layer.forward(&x).unwrap();
    let m = NumericalMetrics::compute(&out, &hf_ref).unwrap();
    assert!(m.argmax_match && m.max_abs_diff < 0.5, "mixtral must still match HF: {:?}", m);
}

#[test]
fn metrics_are_deterministic() {
    let (layer, x, _a, hf) = load_fixture_layer("qwen2_moe");
    let o1 = layer.forward_with(&x, MoeExecutionConvention::HuggingFaceQwen).unwrap();
    let o2 = layer.forward_with(&x, MoeExecutionConvention::HuggingFaceQwen).unwrap();
    assert_eq!(o1, o2);
    let m1 = NumericalMetrics::compute(&o1, &hf).unwrap();
    let m2 = NumericalMetrics::compute(&o2, &hf).unwrap();
    assert_eq!(m1, m2);
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
    assert!(MoeWeightMap::from_tensors(dense.iter().map(|(n, s)| (*n, s.clone()))).is_empty());
}
