//! **MOE-16** — numerical-equivalence validation against reference fixtures.
//!
//! Loads the committed reference fixtures (generated OFFLINE by
//! `fixtures/moe/generate_reference.py`, never in CI, never downloading), runs
//! Atenia's real `RealMoeLayer::forward` on the same input + weights, and
//! compares against two f64 references:
//!  - `atenia_ref` (PRIMARY, gates pass/fail): an independent f64
//!    reimplementation of the exact operation Atenia performs — proves the f32
//!    Rust code (incl. the MOE-15 packed gate/up split) is numerically
//!    correct, ADR-004 tolerance `max_abs_diff < 0.5`.
//!  - `hf_ref` (INFORMATIVE): the real HuggingFace transformers MoE block in
//!    f64 — reported, not asserted; surfaces convention gaps.
//!
//! No models downloaded; no HF/Python at test time; everything is fixture-based.

use std::path::PathBuf;

use atenia_engine::moe::{
    detect_moe, MoeLayerConfig, MoeNumericalReport, MoeWeightMap, NumericalMetrics, RealMoeLayer,
};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn read_json(model: &str) -> serde_json::Value {
    let path = fixture_dir().join(format!("{model}_layer0.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn f32_vec(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

/// Run Atenia's layer-0 MoE forward on a fixture and return
/// (report_vs_atenia_ref, metrics_vs_hf_ref, layers, experts).
fn run_model(model: &str) -> (MoeNumericalReport, NumericalMetrics) {
    let j = read_json(model);
    let num_experts = j["num_experts"].as_u64().unwrap() as usize;
    let experts_per_token = j["experts_per_token"].as_u64().unwrap() as usize;
    let d_model = j["d_model"].as_u64().unwrap() as usize;
    let d_ff = j["d_ff"].as_u64().unwrap() as usize;
    let has_shared = j["has_shared"].as_bool().unwrap();

    let input = f32_vec(&j["input"]);
    let atenia_ref = f32_vec(&j["atenia_ref"]);
    let hf_ref = f32_vec(&j["hf_ref"]);

    // Build the real layer from the fixture safetensors (F32).
    let st = fixture_dir().join(format!("{model}_layer0.safetensors"));
    let reader = SafetensorsReader::open(&st).unwrap();
    // Fail-loud detection still fires on the fixture.
    assert!(
        detect_moe(reader.iter().map(|e| e.name)).is_moe,
        "fixture must be detected as MoE"
    );
    let map = MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())));
    let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());

    let cfg = MoeLayerConfig::new(num_experts, experts_per_token, has_shared, d_model, d_ff)
        .unwrap();
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    let out = layer.forward(&input).unwrap();
    assert_eq!(out.len(), d_model);

    let m_atenia = NumericalMetrics::compute(&out, &atenia_ref)
        .expect("equal-length atenia_ref metrics");
    let m_hf = NumericalMetrics::compute(&out, &hf_ref).expect("equal-length hf_ref metrics");

    let report = MoeNumericalReport::new(model, 1, num_experts, m_atenia);
    println!("PRIMARY (Atenia f32 vs f64-of-same-op): {}", report.summary());
    println!(
        "INFORMATIVE (Atenia vs HF transformers f64): max_abs_diff={:.3e}, mean_abs_diff={:.3e}, rmse={:.3e}, argmax_match={}",
        m_hf.max_abs_diff, m_hf.mean_abs_diff, m_hf.rmse, m_hf.argmax_match
    );
    (report, m_hf)
}

#[test]
fn qwen15_moe_matches_reference() {
    let (report, _hf) = run_model("qwen15_moe");
    assert!(report.pass, "qwen1.5-moe must match its f64 reference: {}", report.summary());
}

#[test]
fn qwen2_moe_matches_reference() {
    let (report, _hf) = run_model("qwen2_moe");
    assert!(report.pass, "qwen2-moe must match its f64 reference: {}", report.summary());
}

#[test]
fn mixtral_matches_reference() {
    let (report, hf) = run_model("mixtral");
    assert!(report.pass, "mixtral must match its f64 reference: {}", report.summary());
    // Mixtral's convention (softmax -> top-k -> renormalise, no shared) equals
    // Atenia's, so Atenia also matches the HF transformers block tightly.
    assert!(
        hf.max_abs_diff < 0.5 && hf.argmax_match,
        "mixtral should also match HF block within tolerance: max_abs_diff={:.3e}",
        hf.max_abs_diff
    );
}

#[test]
fn validation_report_builds() {
    let (report, _hf) = run_model("mixtral");
    assert_eq!(report.layers, 1);
    assert!(report.experts >= 1);
    assert!(report.metrics.len >= 1);
    assert!(!report.summary().is_empty());
}

#[test]
fn metrics_are_deterministic() {
    let (r1, _) = run_model("qwen15_moe");
    let (r2, _) = run_model("qwen15_moe");
    assert_eq!(r1.metrics, r2.metrics, "metrics must be deterministic across runs");
}

#[test]
fn fail_loud_still_active() {
    // The fixtures are real MoE tensors; detection still fires (fail-loud
    // for the productive loader is unchanged by this validation path).
    for model in ["qwen15_moe", "qwen2_moe", "mixtral"] {
        let st = fixture_dir().join(format!("{model}_layer0.safetensors"));
        let reader = SafetensorsReader::open(&st).unwrap();
        assert!(detect_moe(reader.iter().map(|e| e.name)).is_moe);
    }
}

#[test]
fn dense_models_still_load() {
    // A dense-named in-memory listing yields no MoE map (sanity that the
    // validation path doesn't mis-classify dense weights).
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
}
