//! **MOE-6** — integration tests for the primitive MoE graph ops
//! (`MoeRouterSoftmax`, `MoeTopK`, `MoeSparseCombine`). Validates each
//! primitive against the isolated `src/moe` reference helpers, and the
//! full primitive pipeline against the MOE-4/MOE-5 fused reference.
//!
//! No real models, no loaders, no CUDA. Synthetic fixture only.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::moe::dense::build_fixture_layer;
use atenia_engine::moe::{register_layer, softmax, top_k_routing};
use atenia_engine::tensor::Tensor;

fn token(d: usize) -> Vec<f32> {
    (0..d).map(|i| (i as f32) * 0.13 - 0.4).collect()
}

/// Naive `W·x` matvec (W row-major `[rows, cols]`), f64 accum.
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0_f32; rows];
    for r in 0..rows {
        let mut acc = 0.0_f64;
        for c in 0..cols {
            acc += (w[r * cols + c] as f64) * (x[c] as f64);
        }
        y[r] = acc as f32;
    }
    y
}

fn run_unary(node: impl FnOnce(&mut GraphBuilder, usize) -> usize, input: &[f32]) -> Vec<f32> {
    let mut gb = GraphBuilder::new();
    let inp = gb.input();
    let op = node(&mut gb, inp);
    let _ = gb.output(op);
    let mut graph = gb.build();
    let out = graph.execute(vec![Tensor::new_cpu(vec![input.len()], input.to_vec())]);
    out[0].as_cpu_slice().to_vec()
}

#[test]
fn graph_moe_router_softmax_matches_moe_softmax() {
    let logits = vec![1.0_f32, 2.0, 0.5, -1.0, 3.0];
    let got = run_unary(|gb, x| gb.moe_router_softmax(x), &logits);
    let expected = softmax(&logits);
    assert_eq!(got.len(), expected.len());
    for i in 0..expected.len() {
        assert!((got[i] - expected[i]).abs() < 1e-6);
    }
    assert!((got.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn graph_moe_topk_matches_sparse_topk() {
    let weights = vec![0.1_f32, 0.5, 0.2, 0.4];
    let got = run_unary(|gb, x| gb.moe_topk(x, 2), &weights);
    let sel = top_k_routing(&weights, 2).unwrap();
    // Flat layout [idx0, w0, idx1, w1, ...].
    assert_eq!(got.len(), sel.indices.len() * 2);
    for i in 0..sel.indices.len() {
        assert_eq!(got[i * 2] as usize, sel.indices[i]);
        assert!((got[i * 2 + 1] - sel.weights[i]).abs() < 1e-6);
    }
}

#[test]
fn graph_moe_topk_is_deterministic() {
    let weights = vec![0.25_f32, 0.25, 0.25, 0.25];
    let a = run_unary(|gb, x| gb.moe_topk(x, 2), &weights);
    let b = run_unary(|gb, x| gb.moe_topk(x, 2), &weights);
    assert_eq!(a, b);
    // Tie-break by lower index -> experts 0 and 1.
    assert_eq!(a[0] as usize, 0);
    assert_eq!(a[2] as usize, 1);
}

#[test]
fn graph_moe_sparse_combine_matches_manual() {
    // 3 experts, d_model = 2. expert_outputs concat = 6 values.
    let d_model = 2;
    let num_experts = 3;
    let expert_outputs = vec![
        1.0_f32, 2.0, // expert 0
        3.0, 4.0, // expert 1
        5.0, 6.0, // expert 2
    ];
    // Selection: experts 0 and 2 with weights 0.25 / 0.75.
    let selection = vec![0.0_f32, 0.25, 2.0, 0.75];

    let mut gb = GraphBuilder::new();
    let sel_in = gb.input();
    let outs_in = gb.input();
    let combine = gb.moe_sparse_combine(sel_in, outs_in, d_model, num_experts);
    let _ = gb.output(combine);
    let mut graph = gb.build();
    let out = graph.execute(vec![
        Tensor::new_cpu(vec![selection.len()], selection),
        Tensor::new_cpu(vec![expert_outputs.len()], expert_outputs),
    ]);
    let got = out[0].as_cpu_slice();
    // Manual: 0.25*[1,2] + 0.75*[5,6] = [0.25+3.75, 0.5+4.5] = [4.0, 5.0].
    assert!((got[0] - 4.0).abs() < 1e-6);
    assert!((got[1] - 5.0).abs() < 1e-6);
}

#[test]
fn primitive_pipeline_matches_fused_reference_for_fixture() {
    let layer = build_fixture_layer();
    let x = token(layer.d_model);
    let k = 2;
    let n = layer.num_experts();
    let dm = layer.d_model;

    // Inputs computed outside the graph (no dynamic dispatch yet):
    // router logits = W_router · x, and the concatenation of all experts.
    let logits = matvec(&layer.w_router, n, dm, &x);
    let mut expert_outs = Vec::with_capacity(n * dm);
    for e in 0..n {
        expert_outs.extend_from_slice(&layer.experts[e].forward(&x).unwrap());
    }

    // Primitive pipeline graph:
    //   logits -> RouterSoftmax -> TopK(k) ┐
    //   expert_outs ─────────────────────► SparseCombine -> output
    let mut gb = GraphBuilder::new();
    let logits_in = gb.input();
    let outs_in = gb.input();
    let router = gb.moe_router_softmax(logits_in);
    let topk = gb.moe_topk(router, k);
    let combine = gb.moe_sparse_combine(topk, outs_in, dm, n);
    let _ = gb.output(combine);
    let mut graph = gb.build();
    let out = graph.execute(vec![
        Tensor::new_cpu(vec![logits.len()], logits),
        Tensor::new_cpu(vec![expert_outs.len()], expert_outs),
    ]);
    let got = out[0].as_cpu_slice();

    // Reference: the certified sparse forward (MOE-4).
    let reference = layer.forward_sparse(&x, k).unwrap().output;
    assert_eq!(got.len(), reference.len());
    for d in 0..reference.len() {
        assert!(
            (got[d] - reference[d]).abs() < 1e-5,
            "primitive pipeline must equal fused reference at {d}: {} vs {}",
            got[d],
            reference[d]
        );
    }
}

#[test]
fn primitive_ops_reject_bad_k() {
    // Bad k is rejected at the boundary the TopK op calls.
    let weights = vec![0.5_f32, 0.5];
    assert!(top_k_routing(&weights, 0).is_err());
    assert!(top_k_routing(&weights, 5).is_err());
}

#[test]
fn existing_moe_sparse_reference_still_passes() {
    // MOE-5 fused op is intact.
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    let mut gb = GraphBuilder::new();
    let inp = gb.input();
    let moe = gb.moe_sparse_reference(inp, id, 2);
    let _ = gb.output(moe);
    let mut graph = gb.build();
    let out = graph.execute(vec![Tensor::new_cpu(vec![x.len()], x.clone())]);
    let got = out[0].as_cpu_slice();
    let reference = layer.forward_sparse(&x, 2).unwrap().output;
    for d in 0..reference.len() {
        assert!((got[d] - reference[d]).abs() < 1e-5);
    }
}

#[test]
fn existing_dense_graph_tests_still_pass() {
    // A normal dense graph still executes after adding the primitives.
    let mut gb = GraphBuilder::new();
    let inp = gb.input();
    let s = gb.silu(inp);
    let _ = gb.output(s);
    let mut graph = gb.build();
    let out = graph.execute(vec![Tensor::new_cpu(vec![3], vec![-1.0, 0.0, 1.0])]);
    let y = out[0].as_cpu_slice();
    assert_eq!(y.len(), 3);
    assert!((y[1] - 0.0).abs() < 1e-6);
    assert!(y.iter().all(|v| v.is_finite()));
}
