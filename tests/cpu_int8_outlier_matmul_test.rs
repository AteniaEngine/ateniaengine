//! **β.3** — end-to-end MatMul integration test for
//! [`TensorStorage::CpuInt8Outlier`].
//!
//! Design note. After β.2, the MatMul arm in `src/amg/graph.rs`
//! already accepts `CpuInt8Outlier` operands **transparently**:
//! the very first thing MatMul does on its operands is call
//! `Tensor::ensure_decoded`, and β.2 wired the
//! `CpuInt8Outlier => ensure_cpu()` arm in that function. The
//! `ensure_cpu` arm reconstructs the full `[K, N]` F32 matrix
//! (INT8 dequant + sidecar splice) and mutates the local clone's
//! storage to `Cpu(Vec<f32>)`. The downstream CPU matmul code
//! then runs unchanged.
//!
//! β.3 therefore does **not** add a new branch in `graph.rs` —
//! that would be redundant code that duplicates what β.2's
//! `ensure_cpu` already does correctly. Instead, this test suite
//! exercises the full Graph executor against `CpuInt8Outlier` RHS
//! operands to pin the integration contract and prove the
//! numerical improvement carries through the full matmul stack
//! (not just the isolated quantizer-spike level β.1 verified).
//!
//! If a future profiling pass shows that the transition cost in
//! `ensure_cpu` is significant on the hot path, the right answer
//! is a fused MatMul kernel that consumes `q + scales + sidecar`
//! directly — that is the β.6 CUDA milestone, not a CPU
//! redundancy here.

use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::quantizer::{
    absmax_per_group_symmetric, decompose_outliers_topk_by_absmax,
};
use atenia_engine::tensor::{Device, Tensor};

/// Build a row-major `[K, N]` weight with `outlier_cols` columns
/// scaled 1000× over the bulk — the same outlier topology β.1 and
/// β.2 tests use, so the numerical improvement margins compose
/// straight through.
fn build_outlier_weight(k: usize, n: usize, outlier_cols: &[usize]) -> Vec<f32> {
    let mut w = vec![0.0_f32; k * n];
    for row in 0..k {
        for col in 0..n {
            let base = ((row * n + col) as f32 * 0.01) - 0.5;
            w[row * n + col] = if outlier_cols.contains(&col) {
                base * 1000.0
            } else {
                base
            };
        }
    }
    w
}

/// Minimal `[m, k] x [k, n]` matmul graph with two `Input` nodes.
/// Returns the executed output for the given `a` and `b` tensors.
fn run_matmul_graph(a: Tensor, b: Tensor) -> Tensor {
    let mut nodes = Vec::new();
    nodes.push(Node::new(0, NodeType::Input, vec![]));
    nodes.push(Node::new(1, NodeType::Input, vec![]));
    nodes.push(Node::new(2, NodeType::MatMul, vec![0, 1]));
    nodes.push(Node::new(3, NodeType::Output, vec![2]));
    let mut g = Graph::new(nodes);
    let mut outs = g.execute(vec![a, b]);
    outs.pop().expect("graph must produce one output")
}

/// Naive CPU `[m, k] x [k, n]` reference. Used to compute the
/// "ground truth" matmul against the original F32 weight.
fn naive_matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

#[test]
fn matmul_accepts_cpu_int8_outlier_rhs() {
    let (m, k, n) = (4, 16, 8);
    let a = Tensor::randn(&[m, k], Device::CPU);
    let w = build_outlier_weight(k, n, &[1, 5]);
    let decomp = decompose_outliers_topk_by_absmax(&w, &[k, n], 8, 2).unwrap();
    let rhs = Tensor::from_outlier_decomposition(decomp);

    // The graph must accept the outlier-quantised RHS without
    // panicking, and produce an `[m, n]` F32 output.
    let out = run_matmul_graph(a, rhs);
    assert_eq!(out.shape, vec![m, n]);
    assert_eq!(out.dtype, atenia_engine::tensor::DType::F32);
}

#[test]
fn matmul_cpu_int8_outlier_matches_reconstructed_f32_rhs() {
    let (m, k, n) = (4, 16, 8);
    let a = Tensor::randn(&[m, k], Device::CPU);
    let w = build_outlier_weight(k, n, &[1, 5]);

    let decomp = decompose_outliers_topk_by_absmax(&w, &[k, n], 8, 2).unwrap();
    let recon_f32 = atenia_engine::tensor::quantizer::reconstruct_outlier_decomposition(&decomp);
    let rhs_outlier = Tensor::from_outlier_decomposition(decomp);
    let rhs_f32 = Tensor::new_cpu(vec![k, n], recon_f32);

    let out_outlier = run_matmul_graph(a.clone(), rhs_outlier);
    let out_f32 = run_matmul_graph(a, rhs_f32);

    assert_eq!(out_outlier.shape, out_f32.shape);
    // The two paths must produce bit-identical results because
    // ensure_decoded on CpuInt8Outlier reconstructs the very same
    // F32 buffer the other path receives directly.
    let lhs = out_outlier.copy_to_cpu_vec();
    let rhs = out_f32.copy_to_cpu_vec();
    assert_eq!(
        lhs, rhs,
        "matmul with CpuInt8Outlier RHS must equal matmul with reconstructed F32 RHS"
    );
}

#[test]
fn matmul_cpu_int8_outlier_improves_error_vs_plain_int8_rhs() {
    let (m, k, n) = (4, 16, 8);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013 - 0.5)).collect();
    let a = Tensor::new_cpu(vec![m, k], a_data.clone());
    let w = build_outlier_weight(k, n, &[0, 7]);

    // Ground truth: matmul against original F32 weight.
    let truth = naive_matmul(&a_data, m, k, &w, n);

    // Plain INT8 path (no sidecar).
    let (q_plain, scales_plain) = absmax_per_group_symmetric(&w, &[k, n], 8);
    let mut int8_recon = vec![0.0_f32; w.len()];
    for idx in 0..w.len() {
        let row = idx / n;
        let col = idx % n;
        let g = row / 8;
        int8_recon[idx] = (q_plain[idx] as f32) * scales_plain[g * n + col];
    }
    let int8_rhs = Tensor::new_cpu(vec![k, n], int8_recon);
    let out_int8 = run_matmul_graph(a.clone(), int8_rhs).copy_to_cpu_vec();

    // Outlier-decomposed path.
    let decomp = decompose_outliers_topk_by_absmax(&w, &[k, n], 8, 2).unwrap();
    let outlier_rhs = Tensor::from_outlier_decomposition(decomp);
    let out_outlier = run_matmul_graph(a, outlier_rhs).copy_to_cpu_vec();

    let max_err = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    };
    let plain_err = max_err(&out_int8, &truth);
    let outlier_err = max_err(&out_outlier, &truth);
    eprintln!(
        "β.3 matmul error: plain_int8 = {plain_err}, outlier = {outlier_err}, \
         improvement = {:.1}x",
        plain_err / outlier_err.max(1e-12)
    );
    assert!(
        outlier_err * 5.0 < plain_err,
        "β.3 expects outlier matmul error to be >=5x better than plain INT8 \
         (plain={plain_err}, outlier={outlier_err})"
    );
}

#[test]
fn matmul_cpu_int8_outlier_preserves_output_shape() {
    // Cover several non-square shapes to pin the [m, n] = [a.shape[0], rhs.shape[1]]
    // contract under outlier RHS.
    for (m, k, n) in [(1, 8, 4), (3, 16, 1), (5, 32, 7), (2, 12, 9)] {
        let a = Tensor::randn(&[m, k], Device::CPU);
        let w = build_outlier_weight(k, n, &[0]);
        let g = if k <= 8 { k } else { 8 };
        let decomp = decompose_outliers_topk_by_absmax(&w, &[k, n], g, 1).unwrap();
        let rhs = Tensor::from_outlier_decomposition(decomp);
        let out = run_matmul_graph(a, rhs);
        assert_eq!(
            out.shape,
            vec![m, n],
            "shape contract for ({m},{k},{n}) — got {:?}",
            out.shape
        );
    }
}

#[test]
fn matmul_cpu_int8_outlier_does_not_change_plain_cpu_matmul() {
    // No-regression pin: a graph with both operands as plain
    // Cpu(Vec<f32>) must behave bit-identically to the same graph
    // run before β.3 wired CpuInt8Outlier into MatMul. We compare
    // against the naive matmul reference, which has not changed
    // since the test was written.
    let (m, k, n) = (4, 16, 8);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013 - 0.5)).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.007 + 0.1)).collect();
    let a = Tensor::new_cpu(vec![m, k], a_data.clone());
    let b = Tensor::new_cpu(vec![k, n], b_data.clone());

    let truth = naive_matmul(&a_data, m, k, &b_data, n);
    let out = run_matmul_graph(a, b).copy_to_cpu_vec();

    let max_err = out
        .iter()
        .zip(&truth)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-3,
        "plain Cpu × Cpu MatMul drifted (max_err = {max_err}); β.3 must not \
         affect the existing CPU path"
    );
}

#[test]
fn matmul_cpu_int8_outlier_with_k_zero_matches_plain_int8_reconstruction() {
    // k = 0 outliers — the decomposition collapses to pure
    // per-group INT8 (empty sidecar). MatMul output must equal the
    // plain INT8 reconstruction path.
    let (m, k, n) = (4, 16, 8);
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013 - 0.5)).collect();
    let a = Tensor::new_cpu(vec![m, k], a_data.clone());
    let w = build_outlier_weight(k, n, &[]); // no outliers in the source either

    let decomp = decompose_outliers_topk_by_absmax(&w, &[k, n], 8, 0).unwrap();
    assert_eq!(decomp.outlier_cols.len(), 0);
    let rhs_outlier = Tensor::from_outlier_decomposition(decomp);

    // Plain INT8 reconstruction for comparison.
    let (q_plain, scales_plain) = absmax_per_group_symmetric(&w, &[k, n], 8);
    let mut int8_recon = vec![0.0_f32; w.len()];
    for idx in 0..w.len() {
        let row = idx / n;
        let col = idx % n;
        let g = row / 8;
        int8_recon[idx] = (q_plain[idx] as f32) * scales_plain[g * n + col];
    }
    let rhs_int8 = Tensor::new_cpu(vec![k, n], int8_recon);

    let out_outlier = run_matmul_graph(a.clone(), rhs_outlier).copy_to_cpu_vec();
    let out_int8 = run_matmul_graph(a, rhs_int8).copy_to_cpu_vec();

    assert_eq!(
        out_outlier, out_int8,
        "k=0 must produce the same matmul output as plain INT8 reconstruction"
    );
}
