//! APX v20 M3-d.3.C — integration tests for `ensure_gpu` + backward.
//!
//! These tests validate the post-M3-d.3 invariant: when every cached
//! forward output is migrated to `TensorStorage::Cuda`, calling
//! `Graph::backward` still succeeds because the pre-pass in
//! `backward_checked` migrates them back to CPU before any backward
//! closure runs. The second test exercises a graph whose op produces
//! `Tensor` intermediates inside its backward closure (category
//! "problematic" per the M3-d.3.1b audit), so the closure-level
//! `ensure_cpu().expect(...)` guards are also on the code path.
//!
//! # Coverage scope
//!
//! - Happy path: `ensure_gpu` forward → backward → correct gradients.
//! - Both the graph-level pre-pass and the closure-level defense are
//!   exercised, although the closure-level defense is a no-op today
//!   because every op used inside the closures produces CPU tensors
//!   (`nn_linear::matmul`, `transpose_2d`, `reshape_back`, etc.). The
//!   defense future-proofs against ops that are rewired to the GPU in
//!   later milestones.
//!
//! # Limitations
//!
//! - The error path of `backward_checked` (returning
//!   `Err(StorageTransferError::...)`) is not covered here. Exercising it
//!   requires injecting a failure into `gpu_engine()` or into a
//!   `TensorGPU::to_cpu` call, and no mock infrastructure exists for
//!   either today. Coverage will be added alongside such infrastructure
//!   in a future milestone.
//!
//! Both MatMul and Linear are covered: `test_matmul_backward_now_works`
//! and `test_matmul_backward_with_ensure_gpu` exercise the MatMul path
//! after the tape-registration gap was closed (the GPU-plan intercept
//! now skips when `record_tape` is active, letting MatMul fall through
//! to the dispatch that registers a backward entry).
//! `test_backward_works_after_ensure_gpu_with_linear` exercises Linear.
//! `test_backward_works_with_problematic_closure` exercises Reshape.
//!
//! Every test graceful-skips if the singleton engine is unavailable,
//! matching the pattern used by M3-d.1 and M3-d.2 tests.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::gpu::gpu_engine;
use atenia_engine::tensor::{DType, Device, Layout, Tensor};

fn require_gpu(test_name: &str) -> bool {
    if gpu_engine().is_some() {
        true
    } else {
        println!(
            "[TEST:{}] no GPU available (gpu_engine() = None) -> graceful skip",
            test_name
        );
        false
    }
}

fn tensor_from(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    let mut t = Tensor::with_layout(shape, 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    t.as_cpu_slice_mut().copy_from_slice(&data);
    t
}

/// Migrates every cached `node.output` in the graph to `TensorStorage::Cuda`.
/// Used to set up the pre-pass trigger in `backward_checked`.
fn migrate_all_outputs_to_gpu(graph: &mut atenia_engine::amg::graph::Graph) {
    for node in graph.nodes.iter_mut() {
        if let Some(ref mut out) = node.output {
            out.ensure_gpu()
                .expect("test setup: ensure_gpu must succeed");
        }
    }
}

#[test]
fn test_backward_works_after_ensure_gpu_with_linear() {
    if !require_gpu("test_backward_works_after_ensure_gpu_with_linear") {
        return;
    }

    // Uses Linear instead of MatMul because the GPU plan interception
    // in execute_single_inner treats MatMul as a segment-start and
    // executes it via exec_gpu_segment without registering backward
    // tape. This is a pre-existing issue tracked for M3-d.4 (the
    // sub-milestone that refactors gpu_executor.rs). Linear is also
    // a "problematic closure" per the M3-d.3 audit (it generates
    // grad_a, grad_b Tensor intermediates inside its backward
    // closure), so this test still validates the pre-pass plus the
    // intra-closure ensure_cpu patches for the problematic closures
    // path. The MatMul + backward case will be validated in a test
    // added after M3-d.4 resolves the planner tape-registration gap.
    //
    // Graph: out = linear(x, w) with x [1x2], w [2x1] (no bias). Linear
    // computes out = x @ w, so with x=[[1, 2]] and w=[[3], [4]] we get
    // out=[[1*3 + 2*4]] = [[11]]. With grad_out seeded to ones:
    //   grad_x = grad_out @ w^T = [[3, 4]]      (shape [1, 2])
    //   grad_w = x^T @ grad_out = [[1], [2]]    (shape [2, 1])
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w_id = gb.input();
    let lin_id = gb.linear(x_id, w_id, None);
    let out_id = gb.output(lin_id);

    let mut graph = gb.build();

    let x = tensor_from(vec![1, 2], vec![1.0, 2.0]);
    let w = tensor_from(vec![2, 1], vec![3.0, 4.0]);

    let outputs = graph.execute(vec![x, w]);
    assert_eq!(outputs.len(), 1);
    // Forward sanity: x @ w = 1*3 + 2*4 = 11.
    assert!((outputs[0].as_cpu_slice()[0] - 11.0).abs() < 1e-5);

    // Push every cached output to GPU to force the pre-pass in
    // backward_checked to do real D->H transfers.
    migrate_all_outputs_to_gpu(&mut graph);

    // Must not panic: pre-pass migrates outputs back to CPU.
    graph.backward(out_id);

    let x_grad = graph.nodes[x_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("x grad missing");
    let w_grad = graph.nodes[w_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("w grad missing");

    assert_eq!(x_grad.len(), 2);
    assert!((x_grad[0] - 3.0).abs() < 1e-5, "x_grad[0] = {}", x_grad[0]);
    assert!((x_grad[1] - 4.0).abs() < 1e-5, "x_grad[1] = {}", x_grad[1]);

    assert_eq!(w_grad.len(), 2);
    assert!((w_grad[0] - 1.0).abs() < 1e-5, "w_grad[0] = {}", w_grad[0]);
    assert!((w_grad[1] - 2.0).abs() < 1e-5, "w_grad[1] = {}", w_grad[1]);
}

#[test]
fn test_backward_works_with_problematic_closure() {
    if !require_gpu("test_backward_works_with_problematic_closure") {
        return;
    }

    // Graph: out = reshape(x, [3, 2]). Reshape is a "problematic" op
    // (its backward closure builds a Tensor via reshape_back and then
    // calls as_cpu_slice on it, exercising the closure-level defense).
    // Loss seed is ones of shape [3, 2]; grad_x must be ones [2, 3].
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let r_id = gb.reshape(x_id, vec![3, 2]);
    let out_id = gb.output(r_id);

    let mut graph = gb.build();

    let x = tensor_from(vec![2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    let outputs = graph.execute(vec![x]);
    assert_eq!(outputs.len(), 1);
    // Forward sanity: reshape preserves buffer order, so the [3,2]
    // view exposes the same six values in row-major order.
    assert_eq!(
        outputs[0].as_cpu_slice(),
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );

    // Trigger the pre-pass by making every cached output GPU-resident.
    migrate_all_outputs_to_gpu(&mut graph);

    graph.backward(out_id);

    let x_grad = graph.nodes[x_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("x grad missing");
    assert_eq!(x_grad.len(), 6, "grad must match x's numel");
    for (i, g) in x_grad.iter().enumerate() {
        assert!(
            (g - 1.0).abs() < 1e-6,
            "x_grad[{}] = {}, expected 1.0 (seed propagates through reshape identity)",
            i,
            g
        );
    }
}

#[test]
fn test_matmul_backward_now_works() {
    if !require_gpu("test_matmul_backward_now_works") {
        return;
    }

    // Graph: out = matmul(x, w), with x [1x2] and w [2x1] → out [1x1].
    // Analytic gradients (grad_out seeded to ones by default):
    //   grad_x = grad_out @ w^T = [[1*3, 1*4]]             = [3.0, 4.0]
    //   grad_w = x^T @ grad_out = [[1*1], [2*1]]           = [1.0, 2.0]
    //
    // Before M3-d.4.B this failed because the GPU-plan intercept ran
    // exec_gpu_segment for the MatMul node without registering a
    // backward tape entry, leaving grads at None. The fix makes the
    // intercept skip when record_tape is active.
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w_id = gb.input();
    let mm_id = gb.matmul(x_id, w_id);
    let out_id = gb.output(mm_id);

    let mut graph = gb.build();

    let x = tensor_from(vec![1, 2], vec![1.0, 2.0]);
    let w = tensor_from(vec![2, 1], vec![3.0, 4.0]);

    let outputs = graph.execute(vec![x, w]);
    assert_eq!(outputs.len(), 1);
    // Forward sanity: x @ w = 1*3 + 2*4 = 11.
    assert!((outputs[0].as_cpu_slice()[0] - 11.0).abs() < 1e-5);

    graph.backward(out_id);

    let x_grad = graph.nodes[x_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("x grad missing — tape was not registered for MatMul");
    let w_grad = graph.nodes[w_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("w grad missing — tape was not registered for MatMul");

    assert_eq!(x_grad.len(), 2);
    assert!((x_grad[0] - 3.0).abs() < 1e-5, "x_grad[0] = {}", x_grad[0]);
    assert!((x_grad[1] - 4.0).abs() < 1e-5, "x_grad[1] = {}", x_grad[1]);

    assert_eq!(w_grad.len(), 2);
    assert!((w_grad[0] - 1.0).abs() < 1e-5, "w_grad[0] = {}", w_grad[0]);
    assert!((w_grad[1] - 2.0).abs() < 1e-5, "w_grad[1] = {}", w_grad[1]);
}

#[test]
fn test_matmul_backward_with_ensure_gpu() {
    if !require_gpu("test_matmul_backward_with_ensure_gpu") {
        return;
    }

    // Same graph and analytic gradients as test_matmul_backward_now_works,
    // but with every cached forward output migrated to VRAM between
    // forward and backward. This exercises both the tape gap fix from
    // M3-d.4.B and the backward pre-pass from M3-d.3 in a single path.
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w_id = gb.input();
    let mm_id = gb.matmul(x_id, w_id);
    let out_id = gb.output(mm_id);

    let mut graph = gb.build();

    let x = tensor_from(vec![1, 2], vec![1.0, 2.0]);
    let w = tensor_from(vec![2, 1], vec![3.0, 4.0]);

    let outputs = graph.execute(vec![x, w]);
    assert_eq!(outputs.len(), 1);
    assert!((outputs[0].as_cpu_slice()[0] - 11.0).abs() < 1e-5);

    // Force every cached node.output onto VRAM. backward's pre-pass
    // must migrate them back before the backward closures run.
    migrate_all_outputs_to_gpu(&mut graph);

    graph.backward(out_id);

    let x_grad = graph.nodes[x_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("x grad missing after ensure_gpu + backward");
    let w_grad = graph.nodes[w_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("w grad missing after ensure_gpu + backward");

    assert_eq!(x_grad.len(), 2);
    assert!((x_grad[0] - 3.0).abs() < 1e-5, "x_grad[0] = {}", x_grad[0]);
    assert!((x_grad[1] - 4.0).abs() < 1e-5, "x_grad[1] = {}", x_grad[1]);

    assert_eq!(w_grad.len(), 2);
    assert!((w_grad[0] - 1.0).abs() < 1e-5, "w_grad[0] = {}", w_grad[0]);
    assert!((w_grad[1] - 2.0).abs() < 1e-5, "w_grad[1] = {}", w_grad[1]);
}
