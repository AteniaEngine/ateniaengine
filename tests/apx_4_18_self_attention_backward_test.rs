//! Equivalence check: the same self-attention graph run under
//! `ATENIA_APX_MODE=2.5` (no 4.x fusions) and `ATENIA_APX_MODE=4.18`
//! (FusedSelfAttention forward) must produce bit-close gradients on
//! `dX`, `dWq`, `dWk`, `dWv`.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::{DType, Device, Layout, Tensor};

fn assert_close(a: &Tensor, b: &Tensor, tol: f32) {
    assert_eq!(
        a.shape, b.shape,
        "shape mismatch: {:?} vs {:?}",
        a.shape, b.shape
    );
    assert_eq!(a.numel(), b.numel(), "len mismatch");
    for (i, (va, vb)) in a
        .as_cpu_slice()
        .iter()
        .zip(b.as_cpu_slice().iter())
        .enumerate()
    {
        let diff = (va - vb).abs();
        assert!(
            diff <= tol,
            "mismatch at idx {}: a={} b={} diff={} tol={}",
            i,
            va,
            vb,
            diff,
            tol
        );
    }
}

fn build_self_attention_graph() -> (GraphBuilder, usize, usize, usize, usize) {
    let mut gb = GraphBuilder::new();
    let x = gb.input();
    let wq = gb.input();
    let wk = gb.input();
    let wv = gb.input();

    let q = gb.linear(x, wq, None);
    let k = gb.linear(x, wk, None);
    let v = gb.linear(x, wv, None);

    let k_t = gb.transpose_last_two(k);
    let qk = gb.matmul(q, k_t);
    let att = gb.softmax(qk);
    let out = gb.matmul(att, v);
    let _ = gb.output(out);

    (gb, x, wq, wk, wv)
}

fn run_mode(mode: &str) -> (Tensor, Tensor, Tensor, Tensor) {
    if mode == "naive" {
        // Baseline completamente sin fusiones APX 4.x: forzamos 2.5
        unsafe {
            std::env::set_var("ATENIA_APX_MODE", "2.5");
        }
    } else {
        unsafe {
            std::env::set_var("ATENIA_APX_MODE", mode);
        }
    }

    let (gb, x_id, wq_id, wk_id, wv_id) = build_self_attention_graph();
    let mut graph = gb.build();

    let m = 2usize;
    let d = 4usize;

    let x = Tensor::with_layout(vec![m, d], 0.5, Device::CPU, Layout::Contiguous, DType::F32);
    let wq = Tensor::with_layout(vec![d, d], 0.1, Device::CPU, Layout::Contiguous, DType::F32);
    let wk = Tensor::with_layout(
        vec![d, d],
        -0.2,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let wv = Tensor::with_layout(vec![d, d], 0.3, Device::CPU, Layout::Contiguous, DType::F32);

    let inputs = vec![x.clone(), wq.clone(), wk.clone(), wv.clone()];

    let out_id = graph.last_output_id();
    let _outs = graph.execute(inputs);
    graph.backward(out_id);

    let x_t = graph.nodes[x_id]
        .output
        .as_ref()
        .expect("x output missing")
        .clone();
    let wq_t = graph.nodes[wq_id]
        .output
        .as_ref()
        .expect("wq output missing")
        .clone();
    let wk_t = graph.nodes[wk_id]
        .output
        .as_ref()
        .expect("wk output missing")
        .clone();
    let wv_t = graph.nodes[wv_id]
        .output
        .as_ref()
        .expect("wv output missing")
        .clone();

    let make_grad = |node_id: usize, proto: &Tensor| -> Tensor {
        let data = graph.nodes[node_id]
            .output
            .as_ref()
            .and_then(|t| t.grad.as_ref())
            .cloned()
            .unwrap_or_else(|| vec![0.0; proto.numel()]);
        let mut t = Tensor::new_cpu_with_layout(
            proto.shape.clone(),
            data,
            proto.device,
            proto.dtype,
            proto.layout,
        );
        t.strides = proto.strides.clone();
        t
    };

    let dx = make_grad(x_id, &x_t);
    let dwq = make_grad(wq_id, &wq_t);
    let dwk = make_grad(wk_id, &wk_t);
    let dwv = make_grad(wv_id, &wv_t);

    (dx, dwq, dwk, dwv)
}

/// Builds the self-attention graph twice and compares the resulting
/// parameter gradients between the two APX modes:
///
/// - `ATENIA_APX_MODE=2.5` ("naive"): no APX 4.x fusions; the eight
///   individual nodes (three `Linear`s for Q/K/V, `Transpose`,
///   `MatMul`, `Softmax`, `MatMul`) run their regular forward and
///   register their regular backward BackOps.
/// - `ATENIA_APX_MODE=4.18`: `FusedSelfAttention` materialises the
///   final output in a single shot on the last MatMul node.
///
/// APX 4.18 is a forward-only fusion: backward still flows through
/// the individual BackOps of the underlying nodes, so the two modes
/// must produce numerically equivalent gradients.
///
/// History: this test previously failed because the Linear dispatch
/// skipped BackOp registration on Q/K/V Linears under mode 4.18,
/// expecting a fused BackOp to cover them. That fused BackOp was
/// never implemented (see the "disabled for now" note on the
/// `FusedSelfAttention` arm of `exec_fused` in `src/amg/graph.rs`),
/// so the skip silently left the tape empty on those params and
/// backward produced zeros for `dX`, `dWq`, `dWk`, `dWv`. Naive mode
/// was fine because no fusion was active. The fix was to remove the
/// skip: both modes now register the same BackOps and the test
/// asserts their numeric equivalence.
#[test]
fn self_attention_backward_4_18_matches_naive() {
    let (dx_naive, dwq_naive, dwk_naive, dwv_naive) = run_mode("naive");
    let (dx_4_18, dwq_4_18, dwk_4_18, dwv_4_18) = run_mode("4.18");

    let tol = 1e-5;
    assert_close(&dx_naive, &dx_4_18, tol);
    assert_close(&dwq_naive, &dwq_4_18, tol);
    assert_close(&dwk_naive, &dwk_4_18, tol);
    assert_close(&dwv_naive, &dwv_4_18, tol);
}
