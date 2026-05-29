//! **MOE-13** — integration tests for the real MoE checkpoint validation
//! harness, driven through the production `SafetensorsReader`.
//!
//! Builds a tiny safetensors checkpoint with **real Qwen-MoE tensor naming**
//! (router + routed experts + shared expert, multi-layer), opens it with the
//! production reader, and runs the MOE-13 validation harness end-to-end:
//! metadata → derived per-layer config → assembled stack → minimal forward →
//! report.
//!
//! No models downloaded. A real Qwen-MoE (~14B) / Mixtral (~47B) does not fit
//! CI and is NOT used; the harness is target-agnostic and validated here
//! against a synthetic-but-real-format Qwen-MoE-named checkpoint. The MOE-2
//! loader fail-loud guard is untouched (and re-asserted).

use std::collections::HashMap;

use atenia_engine::moe::{detect_moe, MoeWeightMap, RealMoeCheckpointValidation};
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

/// A multi-layer Qwen-MoE-named checkpoint with shared experts.
fn qwen_checkpoint(
    num_layers: usize,
    n: usize,
    d_model: usize,
    d_ff: usize,
    shared_ff: usize,
) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    for l in 0..num_layers {
        let lseed = (l as u64 + 1) * 4000;
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
    t
}

#[test]
fn validation_report_builds() {
    let reader = reader_with(&qwen_checkpoint(2, 4, 8, 16, 24));
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let report = RealMoeCheckpointValidation::validate(&map, 2, &resolve);
    assert!(report.is_moe());
    assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
    assert!(report.summary().contains("forward_ok=true"));
}

#[test]
fn validation_detects_experts() {
    let reader = reader_with(&qwen_checkpoint(3, 4, 8, 16, 24));
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let report = RealMoeCheckpointValidation::validate(&map, 2, &resolve);
    assert_eq!(report.layers_detected, 3);
    assert_eq!(report.experts_detected, 12);
    assert_eq!(report.shared_experts, 3);
    assert_eq!(report.d_model, Some(8));
}

#[test]
fn validation_builds_stack() {
    let reader = reader_with(&qwen_checkpoint(2, 4, 8, 16, 24));
    let map = weight_map(&reader);
    for lid in 0..2 {
        let cfg = RealMoeCheckpointValidation::derive_layer_config(&map, lid, 2).unwrap();
        assert_eq!(cfg.num_experts, 4);
        assert!(cfg.has_shared_expert);
        assert_eq!(cfg.d_model, 8);
        assert_eq!(cfg.d_ff, 16);
    }
}

#[test]
fn validation_runs_forward() {
    let reader = reader_with(&qwen_checkpoint(2, 4, 8, 16, 24));
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let report = RealMoeCheckpointValidation::validate(&map, 2, &resolve);
    assert!(report.forward_pass_ok, "forward must run; errors: {:?}", report.errors);
}

#[test]
fn fail_loud_still_active() {
    // The validation harness is an opt-in path; it does NOT lift the MOE-2
    // loader guard. Detection still fires on the same reader.
    let reader = reader_with(&qwen_checkpoint(1, 4, 8, 16, 24));
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
    let resolve = resolver(&reader);
    let report = RealMoeCheckpointValidation::validate(&map, 2, &resolve);
    assert!(!report.is_moe(), "dense checkpoint must yield a non-MoE report");
    assert!(!report.forward_pass_ok);
}
