//! **MOE-5** — integration test for the experimental fused sparse-MoE
//! graph op. Builds a graph containing `NodeType::MoeSparseReference`,
//! executes it on a small synthetic fixture, and asserts the graph output
//! equals the isolated `MoeDenseLayer::forward_sparse` reference (MOE-4).
//!
//! No real models, no loaders (except the MOE-2 fail-loud check, which is
//! a pure name-detection assertion), no CUDA.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::moe::dense::build_fixture_layer;
use atenia_engine::moe::{detect_moe, register_layer};
use atenia_engine::tensor::Tensor;

/// Deterministic token vector of length `d`.
fn token(d: usize) -> Vec<f32> {
    (0..d).map(|i| (i as f32) * 0.13 - 0.4).collect()
}

/// Build a graph `input -> MoeSparseReference(layer_id, k) -> output`,
/// execute it, and return the output slice.
fn run_graph(layer_id: u32, k: u32, x: &[f32]) -> Vec<f32> {
    let mut gb = GraphBuilder::new();
    let input = gb.input();
    let moe = gb.moe_sparse_reference(input, layer_id, k);
    let _ = gb.output(moe);
    let mut graph = gb.build();
    let tokens = Tensor::new_cpu(vec![x.len()], x.to_vec());
    let outputs = graph.execute(vec![tokens]);
    outputs[0].as_cpu_slice().to_vec()
}

#[test]
fn graph_moe_sparse_matches_reference() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);

    let graph_out = run_graph(id, 2, &x);
    let reference = layer.forward_sparse(&x, 2).unwrap().output;

    assert_eq!(graph_out.len(), reference.len());
    for d in 0..reference.len() {
        assert!(
            (graph_out[d] - reference[d]).abs() < 1e-5,
            "graph MoE output must equal forward_sparse reference at {d}: {} vs {}",
            graph_out[d],
            reference[d]
        );
    }
}

#[test]
fn graph_moe_sparse_is_deterministic() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    let a = run_graph(id, 2, &x);
    let b = run_graph(id, 2, &x);
    assert_eq!(a, b);
    assert!(a.iter().all(|v| v.is_finite()));
}

#[test]
fn graph_moe_sparse_k_equals_all_matches_dense() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    let graph_out = run_graph(id, layer.num_experts() as u32, &x);
    let dense = layer.forward(&x).unwrap();
    for d in 0..dense.len() {
        assert!((graph_out[d] - dense[d]).abs() < 1e-5);
    }
}

#[test]
fn graph_moe_sparse_rejects_bad_k() {
    // Bad k is rejected at the op boundary the graph executor calls
    // (k = 0 -> ZeroK). The graph arm would surface this as a panic on
    // execute; we assert the underlying contract directly.
    let layer = build_fixture_layer();
    let id = register_layer(layer);
    let x = token(8);
    let err = atenia_engine::moe::execute_sparse_reference(id, &x, 0).unwrap_err();
    assert!(format!("{err}").contains("k must be > 0"));
}

#[test]
fn graph_moe_sparse_preserves_existing_dense_graph() {
    // A normal (non-MoE) graph still executes correctly after adding the
    // MoE variant: input -> silu -> output.
    let mut gb = GraphBuilder::new();
    let input = gb.input();
    let silu = gb.silu(input);
    let _ = gb.output(silu);
    let mut graph = gb.build();
    let x = vec![-1.0_f32, 0.0, 1.0, 2.0];
    let out = graph.execute(vec![Tensor::new_cpu(vec![4], x.clone())]);
    let y = out[0].as_cpu_slice();
    // silu(0) == 0; silu is finite and monotonic-ish — just sanity check.
    assert_eq!(y.len(), 4);
    assert!((y[1] - 0.0).abs() < 1e-6, "silu(0) must be 0");
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn moe_checkpoint_still_fails_loud() {
    // MOE-2 detection is unchanged: a checkpoint with expert tensors is
    // still detected (so the loader returns MoeUnsupported).
    let names = vec![
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
    ];
    let det = detect_moe(names);
    assert!(det.is_moe, "MoE checkpoint must still be detected (fail-loud preserved)");
    // And a dense checkpoint is still NOT detected.
    let dense = vec![
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ];
    assert!(!detect_moe(dense).is_moe);
}
