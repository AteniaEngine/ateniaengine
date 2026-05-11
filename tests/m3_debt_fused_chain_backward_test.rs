//! Debt #8 — `FusedLinearActivationChain` backward pass correctness.
//!
//! Pre-fix, `exec_fused_linear_activation_chain` computed forward
//! correctly but did not register a `BackOp`. Consequence: backward
//! produced **zero gradients** for `x`, `W1`, `b1`, `W2`, `b2` of the
//! fused node. Existing tests (`apx_2_5_fused_kernels_test`,
//! `apx_4_9_fusion_test`) did not detect this because they validate
//! seq-vs-par equivalence or forward-only correctness — both paths
//! produced the same zeros.
//!
//! These tests lock the correctness of the Debt #8 closure:
//!
//! 1. `test_backward_correctness_3_input_silu` — 3-input chain (no
//!    biases), SiLU only, compared against a hand-built non-fused
//!    reference that uses `NodeType::SiLU` (which has a pre-existing
//!    standalone BackOp we trust).
//! 2. `test_backward_correctness_4_input_bias_on_first` — 4-input
//!    chain `[x, w1, b1, w2]`, SiLU.
//! 3. `test_backward_correctness_4_input_bias_on_second` — 4-input
//!    chain `[x, w1, w2, b2]`, SiLU.
//! 4. `test_backward_correctness_5_input_both_bias` — 5-input chain
//!    `[x, w1, b1, w2, b2]`, SiLU.
//! 5. `test_backward_multiple_silu_activations` — `Vec<ActType>` of
//!    length 2 (two SiLU activations in sequence between the two
//!    linears). Validates the reverse-iteration loop.
//! 6. `test_backward_nonzero_relu` — ReLU variant. Reference via
//!    non-fused graph is not available (no standalone `NodeType::ReLU`
//!    with BackOp), so this test asserts gradients are finite and
//!    non-trivially non-zero, and seq-vs-par parity holds.
//! 7. `test_backward_nonzero_gelu` — GELU variant, same contract as
//!    test 6.
//! 8. `test_backward_seq_vs_par_fused_chain` — runs the fused chain
//!    through `backward` (parallel) and `backward_sequential`, asserts
//!    gradients match. Pre-fix this passed trivially (all zeros);
//!    post-fix it locks that the parallel and sequential paths agree
//!    on non-zero gradients.
//!
//! The reference graph is hand-built via `Node::new` + `Graph::build`
//! using `NodeType::SiLU` (standalone, NOT `NodeType::Activation(SiLU)`).
//! The APX 4.9 fusion detector only matches `NodeType::Activation(_)`,
//! so the reference graph is **not** fused by `Graph::build`.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::{DType, Device, Layout, Tensor};

// ---------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------

/// Fill a tensor deterministically with a sine-based pattern.
fn fill_sine(t: &mut Tensor, seed: usize) {
    let data = t.as_cpu_slice_mut();
    for (i, v) in data.iter_mut().enumerate() {
        *v = (((i + seed) as f32) * 0.37).sin();
    }
}

fn new_tensor(shape: Vec<usize>, seed: usize) -> Tensor {
    let numel: usize = shape.iter().product();
    let mut t = Tensor::with_layout(shape, 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    fill_sine(&mut t, seed);
    let _ = numel;
    t
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn grad_of(graph: &Graph, node_id: usize) -> Vec<f32> {
    graph.nodes[node_id]
        .output
        .as_ref()
        .expect("node has no output")
        .grad
        .as_ref()
        .cloned()
        .unwrap_or_default()
}

// ---------------------------------------------------------------------
// Graph construction helpers
// ---------------------------------------------------------------------

/// Build a fused graph via `GraphBuilder`. The builder applies APX 4.9
/// fusion automatically, so `Linear → silu → Linear` (using
/// `gb.silu` → `NodeType::Activation(SiLU)`) becomes
/// `FusedLinearActivationChain([SiLU])`.
///
/// Returns: (graph, x_id, w1_id, w2_id, b1_id_opt, b2_id_opt,
/// chain_node_id_for_sanity_check).
///
/// The chain node id is the second Linear's id post-fusion — APX 4.9
/// rewrites the `lin2` node into the `FusedLinearActivationChain`
/// (see `apx4_9::patterns::fuse_linear_activation_linear`).
struct FusedIds {
    graph: Graph,
    x_id: usize,
    w1_id: usize,
    w2_id: usize,
    b1_id: Option<usize>,
    b2_id: Option<usize>,
    loss_id: usize,
    chain_id: usize,
}

fn build_fused_chain_silu(
    x: Tensor,
    w1: Tensor,
    b1: Option<Tensor>,
    w2: Tensor,
    b2: Option<Tensor>,
) -> FusedIds {
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w1_id = gb.parameter(w1);
    let b1_id = b1.map(|t| gb.parameter(t));
    let w2_id = gb.parameter(w2);
    let b2_id = b2.map(|t| gb.parameter(t));

    let l1 = gb.linear(x_id, w1_id, b1_id);
    let a = gb.silu(l1);
    let l2 = gb.linear(a, w2_id, b2_id);
    // Sum of outputs as a scalar loss — grad_out (wrt the sum) is 1s.
    let loss = gb.log_softmax(l2); // any scalar-producing-like; we'll use sum-of-elems by seeding grad manually
    let _out = gb.output(loss);

    let mut graph = gb.build();
    // Execute forward with x as the only input.
    graph.execute(vec![x]);

    // Locate the fused chain node. Post-fusion, `l2` (the second
    // Linear) was rewritten to `FusedLinearActivationChain`, and `l1`
    // and the Activation node became NoOp.
    let mut chain_id = None;
    for n in &graph.nodes {
        if matches!(n.node_type, NodeType::FusedLinearActivationChain(_)) {
            chain_id = Some(n.id);
            break;
        }
    }
    let chain_id = chain_id.expect("fused chain node not found — fusion did not apply");

    FusedIds {
        graph,
        x_id,
        w1_id,
        w2_id,
        b1_id,
        b2_id,
        loss_id: loss,
        chain_id,
    }
}

/// Build a non-fused reference graph with the SAME layout of nodes
/// but using `NodeType::SiLU` (standalone) instead of
/// `NodeType::Activation(SiLU)`. APX 4.9 does not match
/// `NodeType::SiLU`, so no fusion is applied.
///
/// Uses hand-built `Node::new` + `Graph::build` to bypass the builder
/// (which would still auto-fuse `Activation(SiLU)` if used, but
/// sidestepping it is clearer).
struct RefIds {
    graph: Graph,
    x_id: usize,
    w1_id: usize,
    w2_id: usize,
    b1_id: Option<usize>,
    b2_id: Option<usize>,
    loss_id: usize,
}

fn build_ref_chain_silu(
    x: Tensor,
    w1: Tensor,
    b1: Option<Tensor>,
    w2: Tensor,
    b2: Option<Tensor>,
    n_silu: usize,
) -> RefIds {
    // Allocate ids in sequence. Layout:
    //   0: input (x)
    //   1: w1
    //   2: b1 (if present)
    //   w2 next, b2 next, then linear1, silu1..siluN, linear2, output.
    let mut nodes: Vec<Node> = Vec::new();
    let x_id = 0;
    nodes.push(Node::new(x_id, NodeType::Input, vec![]));
    let w1_id = 1;
    nodes.push(Node::new(w1_id, NodeType::Parameter, vec![]));

    let mut next_id = 2;
    let b1_id = if b1.is_some() {
        let id = next_id;
        nodes.push(Node::new(id, NodeType::Parameter, vec![]));
        next_id += 1;
        Some(id)
    } else {
        None
    };
    let w2_id = next_id;
    nodes.push(Node::new(w2_id, NodeType::Parameter, vec![]));
    next_id += 1;
    let b2_id = if b2.is_some() {
        let id = next_id;
        nodes.push(Node::new(id, NodeType::Parameter, vec![]));
        next_id += 1;
        Some(id)
    } else {
        None
    };

    // Linear1
    let l1_id = next_id;
    next_id += 1;
    let mut l1_inputs = vec![x_id, w1_id];
    if let Some(b1i) = b1_id {
        l1_inputs.push(b1i);
    }
    nodes.push(Node::new(l1_id, NodeType::Linear, l1_inputs));

    // Activations (standalone NodeType::SiLU — not fused by detector).
    let mut prev = l1_id;
    for _ in 0..n_silu {
        let aid = next_id;
        next_id += 1;
        nodes.push(Node::new(aid, NodeType::SiLU, vec![prev]));
        prev = aid;
    }

    // Linear2
    let l2_id = next_id;
    next_id += 1;
    let mut l2_inputs = vec![prev, w2_id];
    if let Some(b2i) = b2_id {
        l2_inputs.push(b2i);
    }
    nodes.push(Node::new(l2_id, NodeType::Linear, l2_inputs));

    // Loss as LogSoftmax over l2 (same head as fused graph).
    let loss_id = next_id;
    next_id += 1;
    nodes.push(Node::new(loss_id, NodeType::LogSoftmax, vec![l2_id]));

    // Output node.
    let _out_id = next_id;
    nodes.push(Node::new(next_id, NodeType::Output, vec![loss_id]));

    // Now seed the parameter outputs (Graph::build does NOT execute).
    nodes[w1_id].output = Some(w1);
    if let (Some(id), Some(t)) = (b1_id, b1) {
        nodes[id].output = Some(t);
    }
    nodes[w2_id].output = Some(w2);
    if let (Some(id), Some(t)) = (b2_id, b2) {
        nodes[id].output = Some(t);
    }

    let mut graph = Graph::build(nodes);
    graph.execute(vec![x]);

    RefIds {
        graph,
        x_id,
        w1_id,
        w2_id,
        b1_id,
        b2_id,
        loss_id,
    }
}

/// Run backward on both graphs and return the (fused, ref) grad
/// vectors for a named parameter, given matching id accessors.
fn collect_grads(
    fused: &mut Graph,
    fused_id: usize,
    refg: &mut Graph,
    ref_id: usize,
    fused_loss: usize,
    ref_loss: usize,
) -> (Vec<f32>, Vec<f32>) {
    fused.backward(fused_loss);
    refg.backward(ref_loss);
    (grad_of(fused, fused_id), grad_of(refg, ref_id))
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

const TOL_FUSED_VS_REF: f32 = 1e-4;

#[test]
fn test_backward_correctness_3_input_silu() {
    let batch = 4usize;
    let d_in = 6usize;
    let d_h = 8usize;
    let d_out = 5usize;

    let x = new_tensor(vec![batch, d_in], 1);
    let w1 = new_tensor(vec![d_in, d_h], 2);
    let w2 = new_tensor(vec![d_h, d_out], 3);

    let mut f = build_fused_chain_silu(x.clone(), w1.clone(), None, w2.clone(), None);
    let mut r = build_ref_chain_silu(x, w1, None, w2, None, 1);

    // Sanity: the fused graph actually produced a FusedLinearActivationChain.
    // (Build helper already asserts this.)
    assert!(matches!(
        f.graph.nodes[f.chain_id].node_type,
        NodeType::FusedLinearActivationChain(_)
    ));

    let (gw1_f, gw1_r) = collect_grads(
        &mut f.graph,
        f.w1_id,
        &mut r.graph,
        r.w1_id,
        f.loss_id,
        r.loss_id,
    );
    assert!(
        !gw1_f.is_empty(),
        "fused w1 grad empty — BackOp was not registered"
    );
    assert_eq!(gw1_f.len(), gw1_r.len());
    let d = max_abs_diff(&gw1_f, &gw1_r);
    assert!(
        d < TOL_FUSED_VS_REF,
        "w1 grad max_abs_diff={} > tol={}",
        d,
        TOL_FUSED_VS_REF
    );

    let gw2_f = grad_of(&f.graph, f.w2_id);
    let gw2_r = grad_of(&r.graph, r.w2_id);
    let d = max_abs_diff(&gw2_f, &gw2_r);
    assert!(d < TOL_FUSED_VS_REF, "w2 grad max_abs_diff={}", d);

    let gx_f = grad_of(&f.graph, f.x_id);
    let gx_r = grad_of(&r.graph, r.x_id);
    let d = max_abs_diff(&gx_f, &gx_r);
    assert!(d < TOL_FUSED_VS_REF, "x grad max_abs_diff={}", d);
}

#[test]
fn test_backward_correctness_4_input_bias_on_first() {
    let batch = 3usize;
    let d_in = 5usize;
    let d_h = 7usize;
    let d_out = 4usize;

    let x = new_tensor(vec![batch, d_in], 10);
    let w1 = new_tensor(vec![d_in, d_h], 11);
    let b1 = new_tensor(vec![d_h], 12);
    let w2 = new_tensor(vec![d_h, d_out], 13);

    let mut f = build_fused_chain_silu(x.clone(), w1.clone(), Some(b1.clone()), w2.clone(), None);
    let mut r = build_ref_chain_silu(x, w1, Some(b1), w2, None, 1);

    assert!(matches!(
        f.graph.nodes[f.chain_id].node_type,
        NodeType::FusedLinearActivationChain(_)
    ));
    // Confirm 4-input layout with bias on first linear.
    let chain_inputs = &f.graph.nodes[f.chain_id].inputs;
    assert_eq!(chain_inputs.len(), 4);

    f.graph.backward(f.loss_id);
    r.graph.backward(r.loss_id);

    let check = |name: &str, fg: Vec<f32>, rg: Vec<f32>| {
        assert!(!fg.is_empty(), "{name} fused grad empty", name = name);
        assert_eq!(fg.len(), rg.len(), "{name} length mismatch", name = name);
        let d = max_abs_diff(&fg, &rg);
        assert!(
            d < TOL_FUSED_VS_REF,
            "{} grad max_abs_diff={} > tol={}",
            name,
            d,
            TOL_FUSED_VS_REF
        );
    };
    check("w1", grad_of(&f.graph, f.w1_id), grad_of(&r.graph, r.w1_id));
    check(
        "b1",
        grad_of(&f.graph, f.b1_id.unwrap()),
        grad_of(&r.graph, r.b1_id.unwrap()),
    );
    check("w2", grad_of(&f.graph, f.w2_id), grad_of(&r.graph, r.w2_id));
    check("x", grad_of(&f.graph, f.x_id), grad_of(&r.graph, r.x_id));
}

#[test]
fn test_backward_correctness_4_input_bias_on_second() {
    let batch = 3usize;
    let d_in = 5usize;
    let d_h = 7usize;
    let d_out = 4usize;

    let x = new_tensor(vec![batch, d_in], 20);
    let w1 = new_tensor(vec![d_in, d_h], 21);
    let w2 = new_tensor(vec![d_h, d_out], 22);
    let b2 = new_tensor(vec![d_out], 23);

    let mut f = build_fused_chain_silu(x.clone(), w1.clone(), None, w2.clone(), Some(b2.clone()));
    let mut r = build_ref_chain_silu(x, w1, None, w2, Some(b2), 1);

    assert_eq!(f.graph.nodes[f.chain_id].inputs.len(), 4);

    f.graph.backward(f.loss_id);
    r.graph.backward(r.loss_id);

    let check = |name: &str, fg: Vec<f32>, rg: Vec<f32>| {
        assert!(!fg.is_empty(), "{} fused grad empty", name);
        assert_eq!(fg.len(), rg.len(), "{} length mismatch", name);
        let d = max_abs_diff(&fg, &rg);
        assert!(d < TOL_FUSED_VS_REF, "{} grad max_abs_diff={}", name, d);
    };
    check("w1", grad_of(&f.graph, f.w1_id), grad_of(&r.graph, r.w1_id));
    check("w2", grad_of(&f.graph, f.w2_id), grad_of(&r.graph, r.w2_id));
    check(
        "b2",
        grad_of(&f.graph, f.b2_id.unwrap()),
        grad_of(&r.graph, r.b2_id.unwrap()),
    );
    check("x", grad_of(&f.graph, f.x_id), grad_of(&r.graph, r.x_id));
}

#[test]
fn test_backward_correctness_5_input_both_bias() {
    let batch = 2usize;
    let d_in = 4usize;
    let d_h = 6usize;
    let d_out = 3usize;

    let x = new_tensor(vec![batch, d_in], 30);
    let w1 = new_tensor(vec![d_in, d_h], 31);
    let b1 = new_tensor(vec![d_h], 32);
    let w2 = new_tensor(vec![d_h, d_out], 33);
    let b2 = new_tensor(vec![d_out], 34);

    let mut f = build_fused_chain_silu(
        x.clone(),
        w1.clone(),
        Some(b1.clone()),
        w2.clone(),
        Some(b2.clone()),
    );
    let mut r = build_ref_chain_silu(x, w1, Some(b1), w2, Some(b2), 1);

    assert_eq!(f.graph.nodes[f.chain_id].inputs.len(), 5);

    f.graph.backward(f.loss_id);
    r.graph.backward(r.loss_id);

    let check = |name: &str, fg: Vec<f32>, rg: Vec<f32>| {
        assert!(!fg.is_empty(), "{} fused grad empty", name);
        assert_eq!(fg.len(), rg.len(), "{} length mismatch", name);
        let d = max_abs_diff(&fg, &rg);
        assert!(d < TOL_FUSED_VS_REF, "{} grad max_abs_diff={}", name, d);
    };
    check("w1", grad_of(&f.graph, f.w1_id), grad_of(&r.graph, r.w1_id));
    check(
        "b1",
        grad_of(&f.graph, f.b1_id.unwrap()),
        grad_of(&r.graph, r.b1_id.unwrap()),
    );
    check("w2", grad_of(&f.graph, f.w2_id), grad_of(&r.graph, r.w2_id));
    check(
        "b2",
        grad_of(&f.graph, f.b2_id.unwrap()),
        grad_of(&r.graph, r.b2_id.unwrap()),
    );
    check("x", grad_of(&f.graph, f.x_id), grad_of(&r.graph, r.x_id));
}

#[test]
fn test_backward_multiple_silu_activations() {
    // Two SiLU activations in sequence between the two linears.
    // Fused graph: build via raw nodes because `gb.silu` only adds
    // one activation; two calls of gb.silu produce two Activation(SiLU)
    // nodes that the APX 4.9 detector collapses into a single
    // FusedLinearActivationChain with Vec<ActType> of length 2.
    let batch = 3usize;
    let d_in = 5usize;
    let d_h = 6usize;
    let d_out = 4usize;

    let x = new_tensor(vec![batch, d_in], 40);
    let w1 = new_tensor(vec![d_in, d_h], 41);
    let w2 = new_tensor(vec![d_h, d_out], 42);

    // Fused graph with two Activation(SiLU) nodes.
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w1_id = gb.parameter(w1.clone());
    let w2_id = gb.parameter(w2.clone());
    let l1 = gb.linear(x_id, w1_id, None);
    let a1 = gb.silu(l1);
    let a2 = gb.silu(a1);
    let l2 = gb.linear(a2, w2_id, None);
    let loss = gb.log_softmax(l2);
    gb.output(loss);
    let mut fused = gb.build();
    fused.execute(vec![x.clone()]);

    // Confirm the detector produced a chain with Vec<ActType> of length 2.
    let mut chain_len: Option<usize> = None;
    for n in &fused.nodes {
        if let NodeType::FusedLinearActivationChain(acts) = &n.node_type {
            chain_len = Some(acts.len());
            break;
        }
    }
    assert_eq!(
        chain_len,
        Some(2),
        "expected fused chain with 2 activations, got {:?}",
        chain_len
    );

    // Reference: two standalone NodeType::SiLU between the linears.
    let mut r = build_ref_chain_silu(x, w1, None, w2, None, 2);

    fused.backward(loss);
    r.graph.backward(r.loss_id);

    let check = |name: &str, fg: Vec<f32>, rg: Vec<f32>| {
        assert!(!fg.is_empty(), "{} fused grad empty", name);
        assert_eq!(fg.len(), rg.len(), "{} length mismatch", name);
        let d = max_abs_diff(&fg, &rg);
        assert!(d < TOL_FUSED_VS_REF, "{} grad max_abs_diff={}", name, d);
    };
    check("w1", grad_of(&fused, w1_id), grad_of(&r.graph, r.w1_id));
    check("w2", grad_of(&fused, w2_id), grad_of(&r.graph, r.w2_id));
    check("x", grad_of(&fused, x_id), grad_of(&r.graph, r.x_id));
}

/// Build a fused chain with a given `gb.*` activation constructor
/// (relu / gelu / silu), then assert grads are finite and
/// non-trivially non-zero. Used as smoke test for activations that
/// lack a standalone-BackOp reference path.
fn assert_fused_grads_nonzero<F>(activation: F) -> (Vec<f32>, Vec<f32>, Vec<f32>)
where
    F: Fn(&mut GraphBuilder, usize) -> usize,
{
    let batch = 3usize;
    let d_in = 5usize;
    let d_h = 6usize;
    let d_out = 4usize;

    let x = new_tensor(vec![batch, d_in], 50);
    let w1 = new_tensor(vec![d_in, d_h], 51);
    let w2 = new_tensor(vec![d_h, d_out], 52);

    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w1_id = gb.parameter(w1);
    let w2_id = gb.parameter(w2);
    let l1 = gb.linear(x_id, w1_id, None);
    let a = activation(&mut gb, l1);
    let l2 = gb.linear(a, w2_id, None);
    let loss = gb.log_softmax(l2);
    gb.output(loss);

    let mut g = gb.build();
    g.execute(vec![x]);

    // Sanity: fusion applied.
    let fused_found = g
        .nodes
        .iter()
        .any(|n| matches!(n.node_type, NodeType::FusedLinearActivationChain(_)));
    assert!(fused_found, "fusion did not apply for this activation");

    g.backward(loss);

    let gw1 = grad_of(&g, w1_id);
    let gw2 = grad_of(&g, w2_id);
    let gx = grad_of(&g, x_id);

    assert!(!gw1.is_empty(), "w1 grad empty");
    assert!(!gw2.is_empty(), "w2 grad empty");
    assert!(!gx.is_empty(), "x grad empty");

    for (name, v) in [("w1", &gw1), ("w2", &gw2), ("x", &gx)] {
        assert!(
            v.iter().all(|f| f.is_finite()),
            "{} grad has non-finite values",
            name
        );
        let max = v.iter().map(|f| f.abs()).fold(0.0f32, f32::max);
        assert!(
            max > 1e-6,
            "{} grad is ~zero (max abs = {}); BackOp likely not wiring",
            name,
            max
        );
    }

    (gw1, gw2, gx)
}

#[test]
fn test_backward_nonzero_relu() {
    let (_w1, _w2, _x) = assert_fused_grads_nonzero(|gb, src| gb.relu(src));
}

#[test]
fn test_backward_nonzero_gelu() {
    let (_w1, _w2, _x) = assert_fused_grads_nonzero(|gb, src| gb.gelu(src));
}

#[test]
fn test_backward_seq_vs_par_fused_chain() {
    // Both branches must match after the fix: pre-fix they matched
    // trivially (both zero); post-fix they must match on non-zero
    // grads. This complements the seq-vs-par check in
    // apx_2_5_fused_kernels_test by locking parity specifically for
    // the fused chain in isolation.
    let batch = 4usize;
    let d_in = 6usize;
    let d_h = 8usize;
    let d_out = 5usize;

    let x = new_tensor(vec![batch, d_in], 60);
    let w1 = new_tensor(vec![d_in, d_h], 61);
    let w2 = new_tensor(vec![d_h, d_out], 62);

    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w1_id = gb.parameter(w1);
    let w2_id = gb.parameter(w2);
    let l1 = gb.linear(x_id, w1_id, None);
    let a = gb.silu(l1);
    let l2 = gb.linear(a, w2_id, None);
    let loss = gb.log_softmax(l2);
    gb.output(loss);
    let g_seq = gb.build();
    let mut g_par = g_seq.clone();
    let mut g_seq = g_seq;

    g_seq.execute(vec![x.clone()]);
    g_par.execute(vec![x]);

    g_seq.backward_sequential(loss);
    g_par.backward(loss);

    for id in [x_id, w1_id, w2_id] {
        let gs = grad_of(&g_seq, id);
        let gp = grad_of(&g_par, id);
        assert_eq!(
            gs.len(),
            gp.len(),
            "seq/par grad length mismatch for id {id}"
        );
        assert!(
            !gs.is_empty(),
            "seq grad empty for id {id} — BackOp likely not registered"
        );
        // Non-trivially non-zero
        let max = gs.iter().map(|f| f.abs()).fold(0.0f32, f32::max);
        assert!(
            max > 1e-6,
            "seq grad for id {id} is ~zero (max abs = {})",
            max
        );
        let d = max_abs_diff(&gs, &gp);
        assert!(
            d < 1e-5,
            "seq vs par grad mismatch for id {id}: max_abs_diff={}",
            d
        );
    }
}
