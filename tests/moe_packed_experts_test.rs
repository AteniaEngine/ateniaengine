//! **MOE-15** — integration tests for packed/fused MoE expert support.
//!
//! Builds tiny safetensors checkpoints with the **packed** expert layout
//! (`mlp.experts.gate_up_proj` 3-D + `mlp.experts.down_proj` 3-D) observed in
//! modern Qwen2-MoE / Mixtral checkpoints, plus a classic per-expert
//! checkpoint, and confirms:
//!  - packed tensors are detected and mapped;
//!  - per-expert gate/up/down are sliced out of the packed tensors correctly
//!    (validated against an equivalent classic reference layer within 1e-5);
//!  - a packed layer assembles + runs a forward;
//!  - the classic per-expert path still passes;
//!  - a layer with BOTH formats is rejected as ambiguous;
//!  - fail-loud detection and dense loading are unaffected.
//!
//! No real models; no downloads.

use std::collections::HashMap;

use atenia_engine::moe::{
    build_packed_layer, detect_moe, MoeDenseExpert, MoeDenseLayer, MoeLayerConfig, MoeWeightMap,
    RealMoeLayer,
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

/// Build a packed single-layer checkpoint AND the equivalent classic
/// reference layer from the same per-expert weights.
///
/// Layout (assumed by MOE-15): `gate_up_proj` `[ne, 2*d_ff, d_model]` with the
/// first `d_ff` rows = gate, next `d_ff` = up; `down_proj` `[ne, d_model, d_ff]`.
fn packed_checkpoint(
    ne: usize,
    d_model: usize,
    d_ff: usize,
) -> (Vec<(String, Vec<usize>, Vec<f32>)>, MoeDenseLayer) {
    let mut gate_up_flat: Vec<f32> = Vec::with_capacity(ne * 2 * d_ff * d_model);
    let mut down_flat: Vec<f32> = Vec::with_capacity(ne * d_model * d_ff);
    let mut experts = Vec::with_capacity(ne);
    for e in 0..ne {
        let base = 500 + e as u64;
        let gate = seeded(base * 10 + 1, d_ff * d_model);
        let up = seeded(base * 10 + 2, d_ff * d_model);
        let down = seeded(base * 10 + 3, d_model * d_ff);
        // Packed gate_up block: [gate rows; up rows].
        gate_up_flat.extend_from_slice(&gate);
        gate_up_flat.extend_from_slice(&up);
        down_flat.extend_from_slice(&down);
        experts.push(MoeDenseExpert::new(d_model, d_ff, gate, up, down).unwrap());
    }
    let router = seeded(7, ne * d_model);
    let reference = MoeDenseLayer::new(d_model, d_ff, router.clone(), experts, 2).unwrap();

    let tensors = vec![
        ("model.layers.0.mlp.gate.weight".to_string(), vec![ne, d_model], router),
        (
            "model.layers.0.mlp.experts.gate_up_proj".to_string(),
            vec![ne, 2 * d_ff, d_model],
            gate_up_flat,
        ),
        (
            "model.layers.0.mlp.experts.down_proj".to_string(),
            vec![ne, d_model, d_ff],
            down_flat,
        ),
    ];
    (tensors, reference)
}

/// A classic Qwen-MoE per-expert single layer (no shared expert).
fn classic_checkpoint(ne: usize, d_model: usize, d_ff: usize) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    t.push(("model.layers.0.mlp.gate.weight".to_string(), vec![ne, d_model], seeded(3, ne * d_model)));
    for e in 0..ne {
        let base = 100 + e as u64;
        t.push((
            format!("model.layers.0.mlp.experts.{e}.gate_proj.weight"),
            vec![d_ff, d_model],
            seeded(base * 10 + 1, d_ff * d_model),
        ));
        t.push((
            format!("model.layers.0.mlp.experts.{e}.up_proj.weight"),
            vec![d_ff, d_model],
            seeded(base * 10 + 2, d_ff * d_model),
        ));
        t.push((
            format!("model.layers.0.mlp.experts.{e}.down_proj.weight"),
            vec![d_model, d_ff],
            seeded(base * 10 + 3, d_model * d_ff),
        ));
    }
    t
}

#[test]
fn detects_packed_gate_up() {
    let (tensors, _ref) = packed_checkpoint(4, 8, 16);
    let reader = reader_with(&tensors);
    let det = detect_moe(reader.iter().map(|e| e.name));
    assert!(det.is_moe, "packed checkpoint must be detected as MoE");
    let map = weight_map(&reader);
    let layer = map.layer(0).unwrap();
    assert!(layer.has_packed_experts());
    assert!(!layer.has_classic_experts());
}

#[test]
fn packed_metadata_extracts_num_experts() {
    let (tensors, _ref) = packed_checkpoint(6, 8, 12);
    let reader = reader_with(&tensors);
    let map = weight_map(&reader);
    let layer = map.layer(0).unwrap();
    let gu = layer.packed_gate_up.as_ref().unwrap();
    let dn = layer.packed_down.as_ref().unwrap();
    let dims = atenia_engine::moe::packed_dims(gu, dn).unwrap();
    assert_eq!(dims.num_experts, 6);
    assert_eq!(dims.d_model, 8);
    assert_eq!(dims.d_ff, 12);
}

#[test]
fn packed_binding_extracts_gate_up_down() {
    // The per-expert weights sliced from the packed tensors must equal the
    // reference experts exactly.
    let ne = 4;
    let (d_model, d_ff) = (8, 16);
    let (tensors, reference) = packed_checkpoint(ne, d_model, d_ff);
    let reader = reader_with(&tensors);
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let packed_layer = build_packed_layer(&map, 0, 2, &resolve).unwrap();
    assert_eq!(packed_layer.num_experts(), ne);
    for e in 0..ne {
        assert_eq!(packed_layer.experts[e].w_gate, reference.experts[e].w_gate);
        assert_eq!(packed_layer.experts[e].w_up, reference.experts[e].w_up);
        assert_eq!(packed_layer.experts[e].w_down, reference.experts[e].w_down);
    }
}

#[test]
fn packed_layer_forward() {
    // The packed-assembled layer's forward must match the classic reference
    // (built from the same per-expert weights) within 1e-5.
    let (d_model, d_ff) = (8, 16);
    let (tensors, reference) = packed_checkpoint(4, d_model, d_ff);
    let reader = reader_with(&tensors);
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let cfg = MoeLayerConfig::new(4, 2, false, d_model, d_ff).unwrap();
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    let x = seeded(123, d_model);
    let got = layer.forward(&x).unwrap();
    let expected = reference.forward_sparse(&x, 2).unwrap().output;
    assert_eq!(got.len(), d_model);
    for d in 0..d_model {
        assert!(
            (got[d] - expected[d]).abs() < 1e-5,
            "packed forward must match classic reference at {d}: {} vs {}",
            got[d],
            expected[d]
        );
    }
}

#[test]
fn classic_per_expert_still_passes() {
    let (d_model, d_ff) = (8, 16);
    let reader = reader_with(&classic_checkpoint(4, d_model, d_ff));
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let cfg = MoeLayerConfig::new(4, 2, false, d_model, d_ff).unwrap();
    let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
    assert_eq!(layer.num_experts(), 4);
    let x = seeded(55, d_model);
    let out = layer.forward(&x).unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn ambiguous_classic_and_packed_reports_error() {
    // A layer carrying BOTH classic per-expert AND packed tensors is
    // ambiguous; assembly must refuse rather than guess.
    let (d_model, d_ff) = (8, 16);
    let mut tensors = classic_checkpoint(4, d_model, d_ff);
    // Add packed tensors to the same layer 0.
    let (packed, _ref) = packed_checkpoint(4, d_model, d_ff);
    for t in packed.into_iter().filter(|(n, _, _)| n.contains("experts.")) {
        tensors.push(t);
    }
    let reader = reader_with(&tensors);
    let map = weight_map(&reader);
    let resolve = resolver(&reader);
    let layer = map.layer(0).unwrap();
    assert!(layer.has_classic_experts() && layer.has_packed_experts());
    let cfg = MoeLayerConfig::new(4, 2, false, d_model, d_ff).unwrap();
    let err = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap_err();
    assert!(
        matches!(err, atenia_engine::moe::MoeLayerError::MixedExpertFormat { layer_id: 0 }),
        "expected MixedExpertFormat, got {err:?}"
    );
}

#[test]
fn fail_loud_still_active() {
    let (tensors, _ref) = packed_checkpoint(4, 8, 16);
    let reader = reader_with(&tensors);
    let det = detect_moe(reader.iter().map(|e| e.name));
    assert!(det.is_moe, "packed MoE must still be detected (fail-loud preserved)");
}

#[test]
fn dense_models_still_load() {
    let reader = reader_with(&[
        ("model.embed_tokens.weight".to_string(), vec![16, 4], seeded(1, 64)),
        ("model.layers.0.mlp.gate_proj.weight".to_string(), vec![8, 4], seeded(3, 32)),
        ("model.layers.0.mlp.up_proj.weight".to_string(), vec![8, 4], seeded(4, 32)),
        ("model.layers.0.mlp.down_proj.weight".to_string(), vec![4, 8], seeded(5, 32)),
    ]);
    assert!(!detect_moe(reader.iter().map(|e| e.name)).is_moe);
    assert!(weight_map(&reader).is_empty());
}
