//! Tests for `NodeType::Permute` (M4.5-b0).
//!
//! Coverage:
//!   1. identity perm
//!   2. 2D transpose parity (matches `Transpose2D`)
//!   3. attention layout swap `[b,s,h,d] → [b,h,s,d]`
//!   4. full reverse perm
//!   5. double application via inverse perm = identity
//!   6. backward grad check vs central difference
//!   7. validate rejects malformed perms (3 sub-cases)

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::Tensor;

fn execute_permute(input: &Tensor, perm: Vec<usize>) -> Tensor {
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let p_id = gb.permute(x_id, perm);
    let _ = gb.output(p_id);
    let mut g = gb.build();
    let out = g.execute(vec![input.clone()]);
    out.into_iter().next().expect("expected one output")
}

#[test]
fn permute_identity_is_input() {
    let shape = vec![2, 3, 4, 5];
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 1.5).collect();
    let input = Tensor::new_cpu(shape.clone(), data.clone());
    let out = execute_permute(&input, vec![0, 1, 2, 3]);
    assert_eq!(out.shape, shape);
    assert_eq!(out.as_cpu_slice(), data.as_slice());
}

#[test]
fn permute_2d_matches_transpose2d_semantics() {
    let rows = 3_usize;
    let cols = 4_usize;
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let input = Tensor::new_cpu(vec![rows, cols], data.clone());

    let out = execute_permute(&input, vec![1, 0]);
    assert_eq!(out.shape, vec![cols, rows]);

    // Reference: out[c, r] == data[r * cols + c]
    let out_slice = out.as_cpu_slice();
    for c in 0..cols {
        for r in 0..rows {
            let want = data[r * cols + c];
            let got = out_slice[c * rows + r];
            assert!(
                (got - want).abs() < 1e-7,
                "permute(1,0) mismatch at (c={}, r={}): got {}, want {}",
                c,
                r,
                got,
                want
            );
        }
    }
}

#[test]
fn permute_attention_swap_b_s_h_d_to_b_h_s_d() {
    // The exact shape needed by multi-head attention.
    let (b, s, h, d) = (2_usize, 5, 3, 4);
    let n = b * s * h * d;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let input = Tensor::new_cpu(vec![b, s, h, d], data.clone());

    let out = execute_permute(&input, vec![0, 2, 1, 3]);
    assert_eq!(out.shape, vec![b, h, s, d]);

    let out_slice = out.as_cpu_slice();
    for bi in 0..b {
        for hi in 0..h {
            for si in 0..s {
                for di in 0..d {
                    let in_idx = ((bi * s + si) * h + hi) * d + di;
                    let out_idx = ((bi * h + hi) * s + si) * d + di;
                    assert!(
                        (out_slice[out_idx] - data[in_idx]).abs() < 1e-7,
                        "[b,s,h,d]→[b,h,s,d] mismatch at b={},h={},s={},d={}",
                        bi,
                        hi,
                        si,
                        di
                    );
                }
            }
        }
    }
}

#[test]
fn permute_reverse_all_dims() {
    let shape = vec![2_usize, 3, 4, 5];
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let input = Tensor::new_cpu(shape.clone(), data.clone());

    let out = execute_permute(&input, vec![3, 2, 1, 0]);
    assert_eq!(out.shape, vec![5, 4, 3, 2]);

    // Spot check: out[i3, i2, i1, i0] == in[i0, i1, i2, i3].
    let in_strides = [
        shape[1] * shape[2] * shape[3],
        shape[2] * shape[3],
        shape[3],
        1,
    ];
    let out_strides = [shape[2] * shape[1] * shape[0], shape[1] * shape[0], shape[0], 1];
    let out_slice = out.as_cpu_slice();
    for &(i0, i1, i2, i3) in &[
        (0, 0, 0, 0),
        (1, 2, 3, 4),
        (0, 1, 2, 3),
        (1, 0, 1, 0),
    ] {
        let in_lin = i0 * in_strides[0] + i1 * in_strides[1] + i2 * in_strides[2] + i3 * in_strides[3];
        let out_lin =
            i3 * out_strides[0] + i2 * out_strides[1] + i1 * out_strides[2] + i0 * out_strides[3];
        assert!(
            (out_slice[out_lin] - data[in_lin]).abs() < 1e-7,
            "reverse-all mismatch at ({},{},{},{})",
            i0,
            i1,
            i2,
            i3
        );
    }
}

#[test]
fn permute_double_application_is_identity() {
    // permute(permute(x, p), inv(p)) == x — exercises the
    // inverse_perm path used by the backward closure.
    let shape = vec![2_usize, 3, 4, 5];
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.13 - 1.0).collect();
    let input = Tensor::new_cpu(shape.clone(), data.clone());

    let perm = vec![2, 0, 3, 1];
    let inv = {
        // explicit reference for `inverse_perm` semantics
        let mut iv = vec![0_usize; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            iv[p] = i;
        }
        iv
    };
    let intermediate = execute_permute(&input, perm);
    let recovered = execute_permute(&intermediate, inv);

    assert_eq!(recovered.shape, shape);
    let rec_slice = recovered.as_cpu_slice();
    for (i, (g, w)) in rec_slice.iter().zip(data.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-7,
            "double-permute not identity at {}: got {}, want {}",
            i,
            g,
            w
        );
    }
}

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
            "{}: idx {}: analytical={} numerical={} rel_err={}",
            ctx,
            i,
            a,
            n,
            rel
        );
    }
}

#[test]
fn permute_backward_matches_finite_diff() {
    let shape = vec![2_usize, 3, 4];
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let perm = vec![2_usize, 0, 1];

    // Numerical gradient: scalar loss = sum(permute(x, perm)).
    let numerical = finite_diff_grad(&data, |x_slice| {
        let t = Tensor::new_cpu(shape.clone(), x_slice.to_vec());
        let y = execute_permute(&t, perm.clone());
        y.as_cpu_slice().iter().sum::<f32>()
    });

    // Analytical gradient via the graph tape.
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let p_id = gb.permute(x_id, perm);
    let out_id = gb.output(p_id);
    let mut g = gb.build();
    let _ = g.execute(vec![Tensor::new_cpu(shape.clone(), data)]);
    g.backward(out_id);
    let analytical: &[f32] = g.nodes[x_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("input grad missing")
        .as_slice();

    assert_grad_close(analytical, &numerical, "permute backward");
}

// The 3 sub-cases below all expect a panic from the helper validation.
// We exercise them through a graph build → execute path so the actual
// runtime panic surfaces.

#[test]
#[should_panic(expected = "perm has duplicate index")]
fn permute_validate_rejects_duplicate() {
    let input = Tensor::new_cpu(vec![2, 3, 4], vec![0.0; 24]);
    let _ = execute_permute(&input, vec![0, 1, 1]);
}

#[test]
#[should_panic(expected = "perm index 5 out of range")]
fn permute_validate_rejects_out_of_range() {
    let input = Tensor::new_cpu(vec![2, 3, 4], vec![0.0; 24]);
    let _ = execute_permute(&input, vec![0, 5, 2]);
}

#[test]
#[should_panic]
fn permute_validate_rejects_wrong_length() {
    // Wrong length is also caught by the graph's arity validator
    // (`Permute { .. }` is a 1-input op, but the perm itself must
    // match the runtime input rank). The helper assertion fires
    // first and yields a clear message.
    let input = Tensor::new_cpu(vec![2, 3, 4], vec![0.0; 24]);
    let _ = execute_permute(&input, vec![0, 1]); // rank 3 input, perm len 2
}
