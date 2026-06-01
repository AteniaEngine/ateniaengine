//! **MOE-FULL-14** — partial / sub-reference certification from **real**
//! checkpoints. The `*_moe_layer0` fixtures are layer-0 MoE blocks sliced from
//! real HuggingFace checkpoints (Mixtral, Qwen1.5-MoE, Qwen2-MoE, Qwen3-MoE),
//! each with a committed HF f64 reference (`hf_ref`). This certifies the
//! family-distinguishing MoE block against a real checkpoint (the MOE-1
//! partial/sub-reference methodology — a full f64 forward at scale is
//! infeasible, but a single real layer is). No model downloaded.
//!
//! DeepSeek-MoE has no real-checkpoint layer-0 fixture; its block is certified
//! on a synthetic fixture (MOE-FULL-11) — documented in the certification
//! matrix / manifest.

use std::path::PathBuf;

use atenia_engine::moe::{
    detect_moe, MoeLayerConfig, MoeWeightMap, NumericalMetrics, RealMoeLayer,
};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn f32_vec(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

/// Certify one real-checkpoint layer-0 MoE block vs its HF f64 reference.
fn certify_layer0(model: &str) -> NumericalMetrics {
    let j: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join(format!("{model}_layer0.json"))).unwrap(),
    )
    .unwrap();
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
    assert!(detect_moe(reader.iter().map(|e| e.name)).is_moe, "{model} must be MoE");
    let map = MoeWeightMap::from_tensors(reader.iter().map(|e| (e.name, e.shape.to_vec())));
    let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve)
        .unwrap_or_else(|e| panic!("{model} assemble: {e}"));
    // forward_auto resolves the convention from the captured metadata
    // (HuggingFaceQwen when a shared-expert gate is present).
    let out = layer.forward_auto(&f32_vec(&j["input"])).unwrap();
    let m = NumericalMetrics::compute(&out, &f32_vec(&j["hf_ref"])).unwrap();
    eprintln!("PARTIAL CERT {model} (real checkpoint, layer 0) vs HF: max_abs_diff={:.3e} argmax_match={}", m.max_abs_diff, m.argmax_match);
    m
}

#[test]
fn mixtral_real_layer0_block_certifies() {
    let m = certify_layer0("mixtral");
    assert!(m.argmax_match);
    assert!(m.max_abs_diff < 1e-3, "Mixtral real block: {m:?}");
}

#[test]
fn qwen2_moe_real_layer0_block_certifies() {
    let m = certify_layer0("qwen2_moe");
    assert!(m.argmax_match);
    assert!(m.max_abs_diff < 1e-3, "Qwen2-MoE real block: {m:?}");
}

#[test]
fn qwen15_moe_real_layer0_block_certifies() {
    let m = certify_layer0("qwen15_moe");
    assert!(m.argmax_match);
    assert!(m.max_abs_diff < 1e-3, "Qwen1.5-MoE real block: {m:?}");
}

#[test]
fn qwen3_moe_real_layer0_block_certifies() {
    // Qwen3-MoE's *full transformer* is an unsupported variant (QK-norm
    // attention), but its MoE BLOCK is independent of attention and certifies.
    let m = certify_layer0("qwen3_moe");
    assert!(m.argmax_match);
    assert!(m.max_abs_diff < 1e-3, "Qwen3-MoE real block: {m:?}");
}
