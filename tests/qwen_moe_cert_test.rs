//! **QWEN-MOE-CERT-1** — Qwen-MoE family certification (experimental path).
//!
//! Certifies a representative set of Qwen-MoE checkpoints through the
//! experimental MoE path, fixture-based (no models, no HF at test time):
//!  - `qwen15_moe` — Qwen1.5-MoE-style (Qwen2Moe arch), classic per-expert,
//!    shared expert + sigmoid gate, `norm_topk_prob = false` → HuggingFaceQwen.
//!  - `qwen2_moe`  — Qwen2-MoE, packed experts, shared + gate,
//!    `norm_topk_prob = false` → HuggingFaceQwen.
//!  - `qwen3_moe`  — Qwen3-MoE, packed experts, NO shared expert,
//!    `norm_topk_prob = true`, router named `mlp.router.weight` → Atenia.
//!
//! For each: the convention is resolved automatically, `forward_auto` is run,
//! and the output is compared against the HuggingFace `transformers` f64
//! reference. Certification gate: `max_abs_diff < 0.5` and argmax match
//! (ADR-004); in practice all are ~1e-10.

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

struct Loaded {
    layer: RealMoeLayer,
    map: MoeWeightMap,
    input: Vec<f32>,
    hf_ref: Vec<f32>,
    is_moe: bool,
}

fn load(model: &str) -> Loaded {
    let j = read_json(model);
    let cfg = MoeLayerConfig::new(
        j["num_experts"].as_u64().unwrap() as usize,
        j["experts_per_token"].as_u64().unwrap() as usize,
        j["has_shared"].as_bool().unwrap(),
        j["d_model"].as_u64().unwrap() as usize,
        j["d_ff"].as_u64().unwrap() as usize,
    )
    .unwrap();
    let st = fixture_dir().join(format!("{model}_layer0.safetensors"));
    let reader = SafetensorsReader::open(&st).unwrap();
    let is_moe = detect_moe(reader.iter().map(|e| e.name)).is_moe;
    let map = MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())));
    let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    Loaded {
        layer,
        map,
        input: f32_vec(&j["input"]),
        hf_ref: f32_vec(&j["hf_ref"]),
        is_moe,
    }
}

/// Certify one Qwen-MoE model: auto-resolve convention, forward, compare to HF.
fn certify(model: &str, expected: MoeExecutionConvention) -> NumericalMetrics {
    let l = load(model);
    assert!(l.is_moe, "{model}: must be detected as MoE (fail-loud intact)");
    assert_eq!(
        MoeConventionResolver::from_weight_map(&l.map),
        expected,
        "{model}: convention from metadata"
    );
    assert_eq!(l.layer.resolve_convention(), expected, "{model}: convention from layer");
    let out = l.layer.forward_auto(&l.input).unwrap();
    let m = NumericalMetrics::compute(&out, &l.hf_ref).unwrap();
    println!("CERT {model} ({expected:?}) vs HF: {m:?}");
    assert!(
        m.argmax_match && m.max_abs_diff < 0.5,
        "{model}: must certify against HF (ADR-004 < 0.5): {m:?}"
    );
    // In practice parity is ~1e-10; assert the strong bound too.
    assert!(m.max_abs_diff < 1e-6, "{model}: expected ~1e-10 HF parity: {m:?}");
    m
}

#[test]
fn qwen15_moe_certifies() {
    certify("qwen15_moe", MoeExecutionConvention::HuggingFaceQwen);
}

#[test]
fn qwen2_moe_certifies() {
    certify("qwen2_moe", MoeExecutionConvention::HuggingFaceQwen);
}

#[test]
fn qwen3_moe_certifies() {
    // Qwen3-MoE: no shared expert, norm_topk_prob=true, router `mlp.router` →
    // Atenia convention; exercises the QWEN-MOE-CERT-1 router-detection fix.
    certify("qwen3_moe", MoeExecutionConvention::Atenia);
}

#[test]
fn qwen3_router_naming_is_detected() {
    // The fixture carries the real on-disk router name `mlp.router.weight`.
    let l = load("qwen3_moe");
    assert!(l.map.router_weight(0).is_some(), "qwen3 router must be mapped");
    assert!(l.map.router_weight(0).unwrap().name.contains("mlp.router"));
}

#[test]
fn family_fail_loud_still_active() {
    for model in ["qwen15_moe", "qwen2_moe", "qwen3_moe"] {
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
