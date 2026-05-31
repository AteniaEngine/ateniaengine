//! **MOE-FULL-5** — integration test: a single experimental MoE decoder layer
//! (norm + self-attention + residual + `MoeRealLayerReference` + residual) is
//! assembled in the real AMG `Graph` with a **real** MoE sub-block built from
//! the committed Mixtral layer-0 fixture, and validated against an independent
//! imperative reference within 1e-5.
//!
//! Scope: one decoder layer, single token, CPU-only. No full model, no
//! generation, no KV cache, no multi-token, no loader/Adapter-Toolkit/CUDA/CLI.
//! The MOE-2 fail-loud guard is untouched (and re-asserted).

use std::path::PathBuf;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::moe::{
    decoder_layer_reference, detect_moe, register_and_build_decoder_layer, ExpAttnWeights,
    MoeLayerConfig, MoeWeightMap, NumericalMetrics, RealMoeLayer,
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

/// Assemble the real Mixtral layer-0 MoE block from the committed fixture.
fn mixtral_moe_layer() -> RealMoeLayer {
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
    RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap()
}

fn attn_weights(dm: usize) -> ExpAttnWeights {
    ExpAttnWeights {
        d_model: dm,
        norm1_gamma: seeded(101, dm),
        w_q: seeded(102, dm * dm),
        w_k: seeded(103, dm * dm),
        w_v: seeded(104, dm * dm),
        w_o: seeded(105, dm * dm),
        norm2_gamma: seeded(106, dm),
        rms_eps: 1e-6,
    }
}

fn run_graph(attn: &ExpAttnWeights, moe: RealMoeLayer, x: &[f32]) -> Vec<f32> {
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let (out_id, _id) = register_and_build_decoder_layer(&mut gb, x_id, attn, moe);
    gb.output(out_id);
    let mut g = gb.build();
    let t = Tensor::new_cpu(vec![1, attn.d_model], x.to_vec());
    let outs = g.execute(vec![t]);
    outs[0].as_cpu_slice().to_vec()
}

#[test]
fn decoder_layer_matches_reference() {
    let moe = mixtral_moe_layer();
    let dm = moe.config.d_model;
    let attn = attn_weights(dm);
    let x = seeded(7, dm);

    let reference = decoder_layer_reference(&x, &attn, &moe).unwrap();
    let got = run_graph(&attn, moe, &x);

    let m = NumericalMetrics::compute(&got, &reference).unwrap();
    assert!(
        m.max_abs_diff < 1e-5,
        "decoder layer graph vs reference: max={:.3e} mean={:.3e} rmse={:.3e}",
        m.max_abs_diff,
        m.mean_abs_diff,
        m.rmse
    );
}

#[test]
fn decoder_layer_is_deterministic() {
    let dm = mixtral_moe_layer().config.d_model;
    let attn = attn_weights(dm);
    let x = seeded(9, dm);
    let a = run_graph(&attn, mixtral_moe_layer(), &x);
    let b = run_graph(&attn, mixtral_moe_layer(), &x);
    assert_eq!(a, b);
    assert!(a.iter().all(|v| v.is_finite()));
}

#[test]
#[should_panic]
fn decoder_layer_rejects_bad_dims() {
    // Attention weights with the wrong d_model vs the MoE block / input must
    // fail when the graph executes (shape mismatch in a matmul).
    let moe = mixtral_moe_layer();
    let dm = moe.config.d_model;
    let bad_attn = attn_weights(dm + 4); // mismatched
    let x = seeded(11, dm);
    let _ = run_graph(&bad_attn, moe, &x);
}

#[test]
fn moe_graph_node_still_matches_reference() {
    // The MoE node inside the layer still equals RealMoeLayer::forward_auto on
    // its own input — i.e. MOE-FULL-4's contract holds within the layer.
    let moe = mixtral_moe_layer();
    let dm = moe.config.d_model;
    let x = seeded(13, dm);
    let direct = moe.forward_auto(&x).unwrap();

    // Build just the MoE node (input → MoeRealLayerReference → output).
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let id = atenia_engine::moe::register_real_moe_layer(moe);
    let moe_id = gb.moe_real_layer_reference(x_id, id);
    gb.output(moe_id);
    let mut g = gb.build();
    let t = Tensor::new_cpu(vec![dm], x.clone());
    let got = g.execute(vec![t])[0].as_cpu_slice().to_vec();

    let m = NumericalMetrics::compute(&got, &direct).unwrap();
    assert!(m.max_abs_diff < 1e-5, "moe node vs forward_auto: {m:?}");
}

#[test]
fn fail_loud_still_active() {
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
