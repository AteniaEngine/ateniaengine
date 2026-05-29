//! **MOE-7** — integration tests for experimental dynamic expert dispatch.
//!
//! Validates that the `MoeDynamicDispatch` graph op (which executes ONLY
//! the selected experts) produces the same output as the MOE-6 primitive
//! pipeline, the MOE-5 fused op, and the MOE-4 sparse reference — and that
//! it really runs only `k` experts. Synthetic fixture only; no models, no
//! loaders, no CUDA.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::moe::dense::build_fixture_layer;
use atenia_engine::moe::{execute_dynamic_dispatch, register_layer, top_k_routing};
use atenia_engine::tensor::Tensor;

fn token(d: usize) -> Vec<f32> {
    (0..d).map(|i| (i as f32) * 0.13 - 0.4).collect()
}

/// Naive `W·x` matvec (row-major `[rows, cols]`).
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

#[test]
fn dynamic_dispatch_matches_sparse_reference() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    let k = 2;

    // Selection from the layer's own router (top-k of softmax).
    let routing = layer.route(&x).unwrap();
    let sel = top_k_routing(&routing.weights, k).unwrap();
    let mut selection = Vec::new();
    for i in 0..sel.indices.len() {
        selection.push(sel.indices[i] as f32);
        selection.push(sel.weights[i]);
    }

    let result = execute_dynamic_dispatch(id, &x, &selection).unwrap();
    let reference = layer.forward_sparse(&x, k).unwrap().output;
    for d in 0..reference.len() {
        assert!((result.output[d] - reference[d]).abs() < 1e-5);
    }
}

#[test]
fn dynamic_dispatch_matches_fused_reference() {
    // Full graph: input(logits) → RouterSoftmax → TopK → DynamicDispatch,
    // with the model input as a second graph input, compared to the fused
    // MoeSparseReference op output.
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    let k = 2;
    let n = layer.num_experts();
    let dm = layer.d_model;
    let logits = matvec(&layer.w_router, n, dm, &x);

    let mut gb = GraphBuilder::new();
    let logits_in = gb.input();
    let x_in = gb.input();
    let router = gb.moe_router_softmax(logits_in);
    let topk = gb.moe_topk(router, k);
    let dispatch = gb.moe_dynamic_dispatch(x_in, topk, id, dm, n);
    let _ = gb.output(dispatch);
    let mut graph = gb.build();
    let out = graph.execute(vec![
        Tensor::new_cpu(vec![logits.len()], logits),
        Tensor::new_cpu(vec![x.len()], x.clone()),
    ]);
    let got = out[0].as_cpu_slice();

    // Fused reference (MOE-5).
    let mut gb2 = GraphBuilder::new();
    let inp = gb2.input();
    let moe = gb2.moe_sparse_reference(inp, id, k as u32);
    let _ = gb2.output(moe);
    let mut g2 = gb2.build();
    let fused = g2.execute(vec![Tensor::new_cpu(vec![x.len()], x.clone())]);
    let fused_out = fused[0].as_cpu_slice();

    assert_eq!(got.len(), fused_out.len());
    for d in 0..fused_out.len() {
        assert!(
            (got[d] - fused_out[d]).abs() < 1e-5,
            "dynamic dispatch must equal fused op at {d}: {} vs {}",
            got[d],
            fused_out[d]
        );
    }
}

#[test]
fn dynamic_dispatch_executes_only_selected_experts() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    let k = 2;
    let routing = layer.route(&x).unwrap();
    let sel = top_k_routing(&routing.weights, k).unwrap();
    let mut selection = Vec::new();
    for i in 0..sel.indices.len() {
        selection.push(sel.indices[i] as f32);
        selection.push(sel.weights[i]);
    }
    let result = execute_dynamic_dispatch(id, &x, &selection).unwrap();
    // Exactly k experts executed, and they ARE the selected ones.
    assert_eq!(result.executed_experts.len(), k);
    assert_eq!(result.executed_experts, sel.indices);
    assert!(result.executed_experts.len() < layer.num_experts());
}

#[test]
fn dynamic_dispatch_rejects_bad_selection() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    // Odd-length selection.
    assert!(execute_dynamic_dispatch(id, &x, &[0.0, 0.5, 1.0]).is_err());
    // Out-of-range expert index.
    let bad = vec![(layer.num_experts() as f32) + 5.0, 1.0];
    assert!(execute_dynamic_dispatch(id, &x, &bad).is_err());
}

#[test]
fn dynamic_dispatch_rejects_unknown_layer() {
    let x = vec![0.0_f32; 8];
    assert!(execute_dynamic_dispatch(u32::MAX, &x, &[0.0, 1.0]).is_err());
}

#[test]
fn dynamic_dispatch_is_deterministic() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    let routing = layer.route(&x).unwrap();
    let sel = top_k_routing(&routing.weights, 2).unwrap();
    let mut selection = Vec::new();
    for i in 0..sel.indices.len() {
        selection.push(sel.indices[i] as f32);
        selection.push(sel.weights[i]);
    }
    let a = execute_dynamic_dispatch(id, &x, &selection).unwrap();
    let b = execute_dynamic_dispatch(id, &x, &selection).unwrap();
    assert_eq!(a, b);
}

#[test]
fn primitive_router_topk_dispatch_pipeline_matches_fused() {
    // Same as dynamic_dispatch_matches_fused_reference but also confirms
    // the full primitive chain equals forward_sparse (MOE-4).
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    let k = 2;
    let n = layer.num_experts();
    let dm = layer.d_model;
    let logits = matvec(&layer.w_router, n, dm, &x);

    let mut gb = GraphBuilder::new();
    let logits_in = gb.input();
    let x_in = gb.input();
    let router = gb.moe_router_softmax(logits_in);
    let topk = gb.moe_topk(router, k);
    let dispatch = gb.moe_dynamic_dispatch(x_in, topk, id, dm, n);
    let _ = gb.output(dispatch);
    let mut graph = gb.build();
    let out = graph.execute(vec![
        Tensor::new_cpu(vec![logits.len()], logits),
        Tensor::new_cpu(vec![x.len()], x.clone()),
    ]);
    let got = out[0].as_cpu_slice();
    let reference = layer.forward_sparse(&x, k).unwrap().output;
    for d in 0..reference.len() {
        assert!((got[d] - reference[d]).abs() < 1e-5);
    }
}

#[test]
fn existing_fused_op_still_passes() {
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
