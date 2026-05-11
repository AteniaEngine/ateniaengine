//! RoPE NodeType tests (M4.5-a).
//!
//! Test 1 — canary: forward output matches the PyTorch / HuggingFace
//! reference fixture in `tests/fixtures/rope_reference/`. If this test
//! passes, the half-split layout is correct.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::rope::{apply_rope, apply_rope_backward};
use atenia_engine::tensor::Tensor;
use std::fs;
use std::path::PathBuf;

fn load_json(path: &PathBuf) -> serde_json::Value {
    let s = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", path.display(), e));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("could not parse {}: {}", path.display(), e))
}

fn parse_usize_vec(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .expect("expected array")
        .iter()
        .map(|x| x.as_u64().expect("expected u64") as usize)
        .collect()
}

fn parse_f32_vec(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("expected array")
        .iter()
        .map(|x| x.as_f64().expect("expected f64") as f32)
        .collect()
}

#[test]
fn rope_forward_matches_huggingface_reference() {
    let fixture_dir = PathBuf::from("tests/fixtures/rope_reference");
    let inputs_json = load_json(&fixture_dir.join("inputs.json"));
    let outputs_json = load_json(&fixture_dir.join("expected_outputs.json"));

    let input_shape: Vec<usize> = parse_usize_vec(&inputs_json["shape"]);
    let head_dim = inputs_json["head_dim"].as_u64().expect("head_dim u64") as usize;
    let base_freq = inputs_json["base_freq"].as_u64().expect("base_freq u64") as u32;
    let input_values: Vec<f32> = parse_f32_vec(&inputs_json["values"]);
    let expected_values: Vec<f32> = parse_f32_vec(&outputs_json["values"]);
    let expected_shape: Vec<usize> = parse_usize_vec(&outputs_json["shape"]);

    // Sanity on the fixture itself.
    assert_eq!(input_shape, expected_shape, "fixture shape mismatch");
    assert_eq!(
        input_values.len(),
        input_shape.iter().product::<usize>(),
        "fixture input length / shape mismatch"
    );

    // Build graph: input -> RoPE -> output.
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let rope_id = gb.rope(x_id, head_dim, base_freq);
    let _out_id = gb.output(rope_id);
    let mut g = gb.build();

    let input_tensor = Tensor::new_cpu(input_shape.clone(), input_values);
    let outputs = g.execute(vec![input_tensor]);
    assert_eq!(outputs.len(), 1, "expected one Output tensor");
    let y = &outputs[0];

    assert_eq!(y.shape, input_shape, "output shape mismatch");
    assert_eq!(y.numel(), expected_values.len(), "output numel mismatch");

    // Bit-close vs PyTorch reference. f32 sin/cos round-trip + multiply-add
    // can drift by a few ULP; 1e-6 is the right scale here (every fixture
    // value is O(1)).
    let tol = 1e-6_f32;
    for (i, (got, want)) in y
        .as_cpu_slice()
        .iter()
        .zip(expected_values.iter())
        .enumerate()
    {
        let diff = (got - want).abs();
        assert!(
            diff < tol,
            "RoPE forward mismatch at flat index {}: got {}, want {}, diff {} (tol {})",
            i,
            got,
            want,
            diff,
            tol
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2 — rotation properties (3 sub-tests)
// ---------------------------------------------------------------------------

/// Position 0 has angle = 0 for every frequency, so cos = 1, sin = 0
/// and RoPE collapses to the identity. If this fails, the position
/// index is being applied incorrectly (off-by-one, wrong axis, etc.).
#[test]
fn rope_position_zero_is_identity() {
    let head_dim = 8;
    let base_freq = 10000;
    let input_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();

    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let rope_id = gb.rope(x_id, head_dim, base_freq);
    let _ = gb.output(rope_id);
    let mut graph = gb.build();

    let input = Tensor::new_cpu(vec![1, 1, 1, 8], input_data.clone());
    let outputs = graph.execute(vec![input]);
    let output_slice = outputs[0].as_cpu_slice();

    for i in 0..8 {
        let diff = (output_slice[i] - input_data[i]).abs();
        assert!(
            diff < 1e-7,
            "Position 0 should be identity at index {}: got {}, expected {}, diff {}",
            i,
            output_slice[i],
            input_data[i],
            diff
        );
    }
}

/// Each `(i, i+half)` pair is a 2D rotation, which preserves L2 norm.
/// Property check across all positions and pairs.
#[test]
fn rope_preserves_l2_norm_per_pair() {
    let head_dim = 4;
    let base_freq = 10000;
    let seq_len = 8;
    let half = head_dim / 2;

    let n_elements = 1 * seq_len * 1 * head_dim;
    let input_data: Vec<f32> = (0..n_elements)
        .map(|i| ((i as f32) * 0.3 - 0.5).sin())
        .collect();

    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let rope_id = gb.rope(x_id, head_dim, base_freq);
    let _ = gb.output(rope_id);
    let mut graph = gb.build();

    let input = Tensor::new_cpu(vec![1, seq_len, 1, head_dim], input_data.clone());
    let outputs = graph.execute(vec![input]);
    let output_slice = outputs[0].as_cpu_slice();

    for s in 0..seq_len {
        let base = s * head_dim;
        for i in 0..half {
            let in_norm_sq = input_data[base + i].powi(2) + input_data[base + i + half].powi(2);
            let out_norm_sq =
                output_slice[base + i].powi(2) + output_slice[base + i + half].powi(2);
            let diff = (in_norm_sq - out_norm_sq).abs();
            assert!(
                diff < 1e-5,
                "L2 norm not preserved at seq={}, pair {}: in^2={}, out^2={}, diff={}",
                s,
                i,
                in_norm_sq,
                out_norm_sq,
                diff
            );
        }
    }
}

/// Position > 0 should differ from input. Sanity check that RoPE is
/// actually doing something (catches a degenerate "all-zero rotation"
/// implementation that would still pass `rope_position_zero_is_identity`
/// trivially).
#[test]
fn rope_position_nonzero_changes_values() {
    let head_dim = 4;
    let base_freq = 10000;
    let input_data: Vec<f32> = vec![
        1.0, 0.0, 0.0, 1.0, // pos 0
        1.0, 0.0, 0.0, 1.0, // pos 1
    ];

    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let rope_id = gb.rope(x_id, head_dim, base_freq);
    let _ = gb.output(rope_id);
    let mut graph = gb.build();

    let input = Tensor::new_cpu(vec![1, 2, 1, 4], input_data.clone());
    let outputs = graph.execute(vec![input]);
    let output_slice = outputs[0].as_cpu_slice();

    // Position 0: identity (sanity).
    for i in 0..4 {
        assert!((output_slice[i] - input_data[i]).abs() < 1e-7);
    }
    // Position 1: at least one element must visibly differ from input.
    let any_different = (4..8).any(|i| (output_slice[i] - input_data[i]).abs() > 1e-4);
    assert!(
        any_different,
        "Position 1 should not be identity (output should differ from input)"
    );
}

// ---------------------------------------------------------------------------
// Test 3 — backward via finite differences
// ---------------------------------------------------------------------------

const FD_H: f32 = 1e-3;
const FD_REL_TOL: f32 = 1e-2;

fn finite_diff_grad<F>(base: &[f32], forward: F) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let mut g = vec![0.0_f32; base.len()];
    let mut perturbed = base.to_vec();
    for i in 0..base.len() {
        let orig = perturbed[i];
        perturbed[i] = orig + FD_H;
        let f_plus = forward(&perturbed);
        perturbed[i] = orig - FD_H;
        let f_minus = forward(&perturbed);
        perturbed[i] = orig;
        g[i] = (f_plus - f_minus) / (2.0 * FD_H);
    }
    g
}

fn assert_grad_close(analytical: &[f32], numerical: &[f32], ctx: &str) {
    assert_eq!(analytical.len(), numerical.len(), "{}: len mismatch", ctx);
    for (i, (&a, &n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        let diff = (a - n).abs();
        let scale = a.abs().max(n.abs()).max(1e-4_f32);
        let rel = diff / scale;
        assert!(
            rel < FD_REL_TOL,
            "{}: idx {}: analytical={} numerical={} rel_err={} > tol={}",
            ctx,
            i,
            a,
            n,
            rel,
            FD_REL_TOL
        );
    }
}

/// Analytical backward (via [`apply_rope_backward`]) must match the
/// central-difference numerical gradient of `loss = sum(apply_rope(x))`.
///
/// With `loss = sum(y)`, `out_grad` is a tensor of ones, so
/// analytical grad_x is exactly `apply_rope_backward(ones, shape, ..)`.
#[test]
fn rope_backward_matches_finite_diff_helper() {
    let head_dim = 4;
    let base_freq = 10000;
    let shape = vec![1, 2, 1, head_dim];
    let n: usize = shape.iter().product();
    let input_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    // Numerical gradient: scalar loss = sum(apply_rope(x)).
    let numerical = finite_diff_grad(&input_data, |x_slice| {
        let t = Tensor::new_cpu(shape.clone(), x_slice.to_vec());
        let y = apply_rope(&t, head_dim, base_freq);
        y.as_cpu_slice().iter().sum::<f32>()
    });

    // Analytical gradient: out_grad = ones (since loss = sum).
    let ones = vec![1.0_f32; n];
    let analytical = apply_rope_backward(&ones, &shape, head_dim, base_freq);

    assert_grad_close(&analytical, &numerical, "rope_backward (helper)");
}

/// Same property, but routed through the graph tape: builds the graph,
/// executes forward (records tape), runs `graph.backward`, and reads
/// `grad` off the input node. Validates the tape integration in
/// `src/amg/graph.rs`, not just the helper math.
#[test]
fn rope_backward_via_graph_tape_matches_helper() {
    let head_dim = 4;
    let base_freq = 10000;
    let shape = vec![1, 2, 1, head_dim];
    let n: usize = shape.iter().product();
    let input_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    // Helper analytical reference.
    let ones = vec![1.0_f32; n];
    let helper_grad = apply_rope_backward(&ones, &shape, head_dim, base_freq);

    // Graph: input -> rope -> output. Loss = sum(output) is what
    // graph.backward() seeds (out_grad of ones into the Output node).
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let rope_id = gb.rope(x_id, head_dim, base_freq);
    let out_id = gb.output(rope_id);
    let mut graph = gb.build();

    let input = Tensor::new_cpu(shape.clone(), input_data);
    let _ = graph.execute(vec![input]);
    graph.backward(out_id);

    let x_grad: &[f32] = graph.nodes[x_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("x_id grad missing")
        .as_slice();

    assert_eq!(x_grad.len(), helper_grad.len());
    for (i, (&g, &h)) in x_grad.iter().zip(helper_grad.iter()).enumerate() {
        let diff = (g - h).abs();
        assert!(
            diff < 1e-6,
            "graph-tape grad differs from helper at idx {}: graph={}, helper={}, diff={}",
            i,
            g,
            h,
            diff
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4 — multi-head independence
// ---------------------------------------------------------------------------

/// Heads share frequencies (head_dim is global), so identical input
/// across two heads must produce identical output across those heads.
#[test]
fn rope_applies_independently_per_head() {
    let head_dim = 4;
    let base_freq = 10000;
    let n_heads = 2;
    let seq_len = 3;

    let total = 1 * seq_len * n_heads * head_dim;
    let mut input_data = vec![0.0_f32; total];
    for s in 0..seq_len {
        for h in 0..n_heads {
            for d in 0..head_dim {
                let idx = s * n_heads * head_dim + h * head_dim + d;
                // Same value for every head (h doesn't appear in the formula).
                input_data[idx] = (s as f32) * 0.1 + (d as f32) * 0.05;
            }
        }
    }

    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let rope_id = gb.rope(x_id, head_dim, base_freq);
    let _ = gb.output(rope_id);
    let mut graph = gb.build();

    let input = Tensor::new_cpu(vec![1, seq_len, n_heads, head_dim], input_data);
    let outputs = graph.execute(vec![input]);
    let out_slice = outputs[0].as_cpu_slice();

    for s in 0..seq_len {
        for d in 0..head_dim {
            let head0_idx = s * n_heads * head_dim + 0 * head_dim + d;
            let head1_idx = s * n_heads * head_dim + 1 * head_dim + d;
            let diff = (out_slice[head0_idx] - out_slice[head1_idx]).abs();
            assert!(
                diff < 1e-7,
                "Head 0 and head 1 should match at seq={}, dim={}: {} vs {}",
                s,
                d,
                out_slice[head0_idx],
                out_slice[head1_idx]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 5 — integration in a multi-node graph
// ---------------------------------------------------------------------------

/// Compose RoPE with another op (`Mul`) in the same graph and confirm
/// the graph executes end-to-end with correct shape propagation. This
/// validates that the RoPE node's output tensor (shape, dtype, device)
/// is consumable by downstream ops, not just by `Output`.
#[test]
fn rope_integrates_with_mul_in_graph() {
    let head_dim = 4;
    let base_freq = 10000;
    let shape = vec![1, 3, 2, head_dim]; // [batch, seq, n_heads, head_dim]
    let n: usize = shape.iter().product();

    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05).collect();
    let scale_data: Vec<f32> = vec![2.0_f32; n];

    // Graph: x -> rope -> mul(rope_out, scale) -> output
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let scale_id = gb.input();
    let rope_id = gb.rope(x_id, head_dim, base_freq);
    let scaled_id = gb.mul(rope_id, scale_id);
    let _ = gb.output(scaled_id);
    let mut graph = gb.build();

    let input = Tensor::new_cpu(shape.clone(), input_data.clone());
    let scale = Tensor::new_cpu(shape.clone(), scale_data);
    let outputs = graph.execute(vec![input.clone(), scale]);
    assert_eq!(outputs.len(), 1, "expected one Output");
    assert_eq!(
        outputs[0].shape, shape,
        "shape must propagate through rope+mul"
    );

    // Independent reference: apply rope directly, then *2.
    let rope_ref = apply_rope(&input, head_dim, base_freq);
    let expected: Vec<f32> = rope_ref.as_cpu_slice().iter().map(|v| v * 2.0).collect();

    let got = outputs[0].as_cpu_slice();
    for (i, (&a, &b)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-6,
            "mismatch at idx {}: graph={} expected={} diff={}",
            i,
            a,
            b,
            diff
        );
    }
}
