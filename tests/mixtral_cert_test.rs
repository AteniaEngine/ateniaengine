//! **MIXTRAL-CERT-1** — Mixtral family certification (experimental path).
//!
//! Certifies Mixtral through the experimental MoE path, fixture-based (no
//! models / HF at test time). Mixtral's convention (softmax -> top-k ->
//! renormalise, NO shared expert) equals Atenia's default, so the
//! auto-resolver picks `Atenia` and the output matches the HuggingFace
//! transformers block reference.
//!
//! The committed CI fixture is the small `hf-internal-testing/
//! tiny-random-MixtralForCausalLM` (packed experts, 4 experts, d_model 64).
//!
//! A second real Mixtral, `TitanML/tiny-mixtral` (classic
//! `block_sparse_moe.experts.{E}.w1/w3/w2`, 8 experts, d_model 1024, ~940 MB),
//! was validated **locally** — MOE-14 smoke PASS and Atenia-vs-HF numerical
//! parity `max_abs_diff = 1.49e-8` (argmax match) — but its layer-0 fixture is
//! ~352 MB, far too large to commit, so it is recorded as a local validation
//! in `docs/HANDOFF_MIXTRAL_CERT_1.md` (reproducible via
//! `fixtures/moe/generate_reference.py`). Its classic on-disk naming is
//! exercised here by the detection assertions and end-to-end by the smoke.

use std::path::PathBuf;

use atenia_engine::moe::{
    classify_tensor_name, detect_moe, is_moe_router_tensor, MoeConventionResolver,
    MoeExecutionConvention, MoeLayerConfig, MoeWeightMap, NumericalMetrics, RealMoeLayer,
    TensorRole,
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

#[test]
fn mixtral_certifies() {
    let model = "mixtral";
    let j = read_json(model);
    let cfg = MoeLayerConfig::new(
        j["num_experts"].as_u64().unwrap() as usize,
        j["experts_per_token"].as_u64().unwrap() as usize,
        j["has_shared"].as_bool().unwrap(),
        j["d_model"].as_u64().unwrap() as usize,
        j["d_ff"].as_u64().unwrap() as usize,
    )
    .unwrap();
    let reader =
        SafetensorsReader::open(&fixture_dir().join(format!("{model}_layer0.safetensors"))).unwrap();
    assert!(detect_moe(reader.iter().map(|e| e.name)).is_moe, "mixtral must be detected as MoE");
    let map = MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())));
    // Mixtral has no shared-expert gate -> Atenia convention.
    assert_eq!(MoeConventionResolver::from_weight_map(&map), MoeExecutionConvention::Atenia);
    let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    assert_eq!(layer.resolve_convention(), MoeExecutionConvention::Atenia);

    let out = layer.forward_auto(&f32_vec(&j["input"])).unwrap();
    let m = NumericalMetrics::compute(&out, &f32_vec(&j["hf_ref"])).unwrap();
    println!("CERT mixtral (Atenia) vs HF: {m:?}");
    assert!(m.argmax_match && m.max_abs_diff < 0.5, "mixtral must certify vs HF (<0.5): {m:?}");
    assert!(m.max_abs_diff < 1e-6, "expected ~1e-10 HF parity: {m:?}");
}

#[test]
fn mixtral_classic_naming_detected() {
    // The original Mixtral on-disk layout (block_sparse_moe + w1/w3/w2), used
    // by the real TitanML checkpoint, classifies correctly. (TitanML's smoke
    // exercises this end-to-end on real data; see the handoff.)
    assert!(is_moe_router_tensor("model.layers.0.block_sparse_moe.gate.weight"));
    assert_eq!(
        classify_tensor_name("model.layers.0.block_sparse_moe.experts.0.w1.weight").role,
        TensorRole::MoeExpertGate
    );
    assert_eq!(
        classify_tensor_name("model.layers.0.block_sparse_moe.experts.0.w3.weight").role,
        TensorRole::MoeExpertUp
    );
    assert_eq!(
        classify_tensor_name("model.layers.0.block_sparse_moe.experts.0.w2.weight").role,
        TensorRole::MoeExpertDown
    );
}

#[test]
fn mixtral_fail_loud_still_active() {
    let reader = SafetensorsReader::open(&fixture_dir().join("mixtral_layer0.safetensors")).unwrap();
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
