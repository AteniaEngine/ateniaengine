//! **MOE-8** — integration tests for conditional expert subgraph
//! execution. Builds a graph where each expert is a `ConditionalExpert`
//! node that runs only when selected by `MoeTopK`, sums their
//! contributions, and validates the result against MOE-7 dynamic
//! dispatch, the MOE-5 fused op, MOE-4 sparse reference, and the MOE-3
//! dense oracle. Synthetic fixture only; no models, no loaders, no CUDA.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::moe::dense::build_fixture_layer;
use atenia_engine::moe::{
    execute_conditional_expert, register_layer, top_k_routing,
};
use atenia_engine::tensor::Tensor;

fn token(d: usize) -> Vec<f32> {
    (0..d).map(|i| (i as f32) * 0.13 - 0.4).collect()
}

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

/// Build the conditional-expert pipeline graph and run it:
///   logits → RouterSoftmax → TopK(k) ┐
///   x ───────────────────────────────┤→ N ConditionalExpert nodes → tree-sum → output
fn run_conditional_pipeline(
    layer_id: u32,
    k: usize,
    logits: &[f32],
    x: &[f32],
    n: usize,
    dm: usize,
) -> Vec<f32> {
    let mut gb = GraphBuilder::new();
    let logits_in = gb.input();
    let x_in = gb.input();
    let router = gb.moe_router_softmax(logits_in);
    let topk = gb.moe_topk(router, k);
    // One conditional-expert node per expert.
    let mut contribs: Vec<usize> = Vec::with_capacity(n);
    for e in 0..n {
        contribs.push(gb.moe_conditional_expert(x_in, topk, layer_id, e as u32, dm));
    }
    // Tree-sum the contributions.
    let mut acc = contribs[0];
    for &c in &contribs[1..] {
        acc = gb.add(acc, c);
    }
    let _ = gb.output(acc);
    let mut graph = gb.build();
    let out = graph.execute(vec![
        Tensor::new_cpu(vec![logits.len()], logits.to_vec()),
        Tensor::new_cpu(vec![x.len()], x.to_vec()),
    ]);
    out[0].as_cpu_slice().to_vec()
}

#[test]
fn conditional_expert_executes_when_selected() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    // Selection that includes expert 1.
    let selection = vec![1.0_f32, 0.6, 3.0, 0.4];
    let (contrib, executed) = execute_conditional_expert(id, 1, &x, &selection).unwrap();
    assert!(executed, "selected expert must execute");
    // Contribution = 0.6 * expert1(x).
    let expert_out = layer.experts[1].forward(&x).unwrap();
    for d in 0..layer.d_model {
        assert!((contrib[d] - 0.6 * expert_out[d]).abs() < 1e-5);
    }
}

#[test]
fn conditional_expert_skips_when_not_selected() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    // Selection that does NOT include expert 2.
    let selection = vec![0.0_f32, 0.5, 3.0, 0.5];
    let (contrib, executed) = execute_conditional_expert(id, 2, &x, &selection).unwrap();
    assert!(!executed, "unselected expert must be skipped");
    // Contribution must be all zeros (forward never ran).
    assert!(contrib.iter().all(|&v| v == 0.0));
    assert_eq!(contrib.len(), layer.d_model);
}

#[test]
fn executed_and_skipped_counts_are_correct() {
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
    let mut executed = 0;
    let mut skipped = 0;
    for e in 0..layer.num_experts() {
        let (_c, ran) = execute_conditional_expert(id, e as u32, &x, &selection).unwrap();
        if ran {
            executed += 1;
        } else {
            skipped += 1;
        }
    }
    assert_eq!(executed, k, "exactly k experts must execute");
    assert_eq!(skipped, layer.num_experts() - k);
}

#[test]
fn conditional_pipeline_matches_dynamic_dispatch() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    let k = 2;
    let n = layer.num_experts();
    let dm = layer.d_model;
    let logits = matvec(&layer.w_router, n, dm, &x);

    let conditional = run_conditional_pipeline(id, k, &logits, &x, n, dm);

    // MOE-7 dynamic dispatch (single graph op).
    let mut gb = GraphBuilder::new();
    let logits_in = gb.input();
    let x_in = gb.input();
    let router = gb.moe_router_softmax(logits_in);
    let topk = gb.moe_topk(router, k);
    let dispatch = gb.moe_dynamic_dispatch(x_in, topk, id, dm, n);
    let _ = gb.output(dispatch);
    let mut g = gb.build();
    let dd = g.execute(vec![
        Tensor::new_cpu(vec![logits.len()], logits.clone()),
        Tensor::new_cpu(vec![x.len()], x.clone()),
    ]);
    let dispatch_out = dd[0].as_cpu_slice();

    for d in 0..dm {
        assert!(
            (conditional[d] - dispatch_out[d]).abs() < 1e-5,
            "conditional pipeline must equal dynamic dispatch at {d}"
        );
    }
}

#[test]
fn conditional_pipeline_matches_sparse_reference() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    let k = 2;
    let n = layer.num_experts();
    let dm = layer.d_model;
    let logits = matvec(&layer.w_router, n, dm, &x);
    let conditional = run_conditional_pipeline(id, k, &logits, &x, n, dm);
    let reference = layer.forward_sparse(&x, k).unwrap().output;
    for d in 0..dm {
        assert!((conditional[d] - reference[d]).abs() < 1e-5);
    }
}

#[test]
fn conditional_pipeline_matches_dense_oracle() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    let k = 2;
    let n = layer.num_experts();
    let dm = layer.d_model;
    let logits = matvec(&layer.w_router, n, dm, &x);
    let conditional = run_conditional_pipeline(id, k, &logits, &x, n, dm);

    // Dense oracle restricted to the top-k experts (MOE-3/4).
    let routing = layer.route(&x).unwrap();
    let sel = top_k_routing(&routing.weights, k).unwrap();
    let oracle = layer.forward_dense_restricted(&x, &sel.indices).unwrap();
    for d in 0..dm {
        assert!((conditional[d] - oracle[d]).abs() < 1e-5);
    }
}

#[test]
fn conditional_pipeline_is_deterministic() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    let n = layer.num_experts();
    let dm = layer.d_model;
    let logits = matvec(&layer.w_router, n, dm, &x);
    let a = run_conditional_pipeline(id, 2, &logits, &x, n, dm);
    let b = run_conditional_pipeline(id, 2, &logits, &x, n, dm);
    assert_eq!(a, b);
}

#[test]
fn existing_fused_tests_still_pass() {
    // MOE-5 fused op + MOE-7 dispatch still behave.
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    let mut gb = GraphBuilder::new();
    let inp = gb.input();
    let moe = gb.moe_sparse_reference(inp, id, 2);
    let _ = gb.output(moe);
    let mut g = gb.build();
    let out = g.execute(vec![Tensor::new_cpu(vec![x.len()], x.clone())]);
    let got = out[0].as_cpu_slice();
    let reference = layer.forward_sparse(&x, 2).unwrap().output;
    for d in 0..reference.len() {
        assert!((got[d] - reference[d]).abs() < 1e-5);
    }
}

#[test]
fn conditional_expert_rejects_unknown_layer_and_expert() {
    let layer = build_fixture_layer();
    let id = register_layer(layer.clone());
    let x = token(layer.d_model);
    // Unknown layer.
    assert!(execute_conditional_expert(u32::MAX, 0, &x, &[0.0, 1.0]).is_err());
    // Expert id out of range.
    assert!(execute_conditional_expert(id, layer.num_experts() as u32, &x, &[0.0, 1.0]).is_err());
}
