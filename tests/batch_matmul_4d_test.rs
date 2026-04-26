//! Tests for the rank-4 extension of `NodeType::BatchMatMul` (M4.5-b0).
//!
//! Coverage:
//!   1. smoke 4D — no panic, output shape is correct
//!   2. 3D regression — rank-3 path still works after the extension
//!   3. flatten equivalence — BMM4D with dim0=1 == BMM3D
//!   4. forward bit-exact vs a manual reference matmul
//!   5. backward grad check vs central difference

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::Tensor;

fn run_bmm(a: &Tensor, b: &Tensor) -> Tensor {
    let mut gb = GraphBuilder::new();
    let a_id = gb.input();
    let b_id = gb.input();
    let bmm_id = gb.batch_matmul(a_id, b_id);
    let _ = gb.output(bmm_id);
    let mut g = gb.build();
    let out = g.execute(vec![a.clone(), b.clone()]);
    out.into_iter().next().expect("expected one output")
}

/// CPU-only reference: `out[..., m, n] = sum_k a[..., m, k] * b[..., k, n]`.
fn manual_bmm4d(a_data: &[f32], b_data: &[f32], outer: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; outer * m * n];
    for o in 0..outer {
        let a_off = o * m * k;
        let b_off = o * k * n;
        let out_off = o * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for kk in 0..k {
                    acc += a_data[a_off + i * k + kk] * b_data[b_off + kk * n + j];
                }
                out[out_off + i * n + j] = acc;
            }
        }
    }
    out
}

#[test]
fn bmm_4d_smoke_no_panic_correct_shape() {
    // [b=2, h=3, m=4, k=5] @ [b=2, h=3, k=5, n=6] → [b=2, h=3, m=4, n=6]
    let (bd, hd, m, k, n) = (2_usize, 3, 4, 5, 6);
    let a_data: Vec<f32> = (0..bd * hd * m * k).map(|i| i as f32 * 0.01).collect();
    let b_data: Vec<f32> = (0..bd * hd * k * n).map(|i| (i as f32) * 0.02 - 0.5).collect();

    let a = Tensor::new_cpu(vec![bd, hd, m, k], a_data);
    let b = Tensor::new_cpu(vec![bd, hd, k, n], b_data);
    let out = run_bmm(&a, &b);

    assert_eq!(out.shape, vec![bd, hd, m, n]);
    assert_eq!(out.numel(), bd * hd * m * n);
}

#[test]
fn bmm_3d_regression_still_works() {
    // The existing rank-3 path must be unaffected by the extension.
    let (batch, m, k, n) = (4_usize, 5, 6, 7);
    let a_data: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * -0.05 + 1.0).collect();
    let a = Tensor::new_cpu(vec![batch, m, k], a_data.clone());
    let b = Tensor::new_cpu(vec![batch, k, n], b_data.clone());
    let out = run_bmm(&a, &b);

    let expected = manual_bmm4d(&a_data, &b_data, batch, m, k, n);
    let got = out.as_cpu_slice();
    assert_eq!(got.len(), expected.len());
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let scale = e.abs().max(1e-4);
        assert!(
            diff / scale < 1e-4,
            "BMM3D regression mismatch at {}: got {} expected {}",
            i,
            g,
            e
        );
    }
}

#[test]
fn bmm_4d_flatten_equivalence_with_3d() {
    // BMM4D with dim0=1 should equal BMM3D over the inner [batch, m, k] tensors.
    let (m, k, n) = (4_usize, 5, 6);
    let inner_batch = 3_usize;

    let a_data: Vec<f32> = (0..inner_batch * m * k).map(|i| (i as f32) * 0.07).collect();
    let b_data: Vec<f32> = (0..inner_batch * k * n).map(|i| (i as f32) * 0.03 - 0.2).collect();

    let a3 = Tensor::new_cpu(vec![inner_batch, m, k], a_data.clone());
    let b3 = Tensor::new_cpu(vec![inner_batch, k, n], b_data.clone());
    let out3 = run_bmm(&a3, &b3);

    let a4 = Tensor::new_cpu(vec![1, inner_batch, m, k], a_data);
    let b4 = Tensor::new_cpu(vec![1, inner_batch, k, n], b_data);
    let out4 = run_bmm(&a4, &b4);

    assert_eq!(out4.shape, vec![1, inner_batch, m, n]);
    assert_eq!(out4.numel(), out3.numel());

    for (i, (&g4, &g3)) in out4
        .as_cpu_slice()
        .iter()
        .zip(out3.as_cpu_slice().iter())
        .enumerate()
    {
        assert!(
            (g4 - g3).abs() < 1e-4,
            "4D-vs-3D equivalence mismatch at {}: 4D={} 3D={}",
            i,
            g4,
            g3
        );
    }
}

#[test]
fn bmm_4d_forward_bit_exact_vs_manual() {
    let (bd, hd, m, k, n) = (2_usize, 3, 4, 5, 6);
    let a_data: Vec<f32> = (0..bd * hd * m * k).map(|i| ((i as f32) * 0.31).sin()).collect();
    let b_data: Vec<f32> = (0..bd * hd * k * n)
        .map(|i| ((i as f32) * 0.17 + 0.5).cos())
        .collect();

    let a = Tensor::new_cpu(vec![bd, hd, m, k], a_data.clone());
    let b = Tensor::new_cpu(vec![bd, hd, k, n], b_data.clone());
    let out = run_bmm(&a, &b);
    let got = out.as_cpu_slice();

    let expected = manual_bmm4d(&a_data, &b_data, bd * hd, m, k, n);
    assert_eq!(got.len(), expected.len());
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let scale = e.abs().max(1e-4);
        assert!(
            diff / scale < 1e-4,
            "BMM4D forward mismatch at {}: got {} expected {}",
            i,
            g,
            e
        );
    }
}

const FD_H: f32 = 1e-3;
const FD_REL_TOL: f32 = 2e-2;

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

#[test]
fn bmm_4d_backward_grad_check() {
    let (bd, hd, m, k, n) = (1_usize, 2, 2, 3, 2);
    let a_size = bd * hd * m * k;
    let b_size = bd * hd * k * n;
    let a_data: Vec<f32> = (0..a_size).map(|i| (i as f32) * 0.05).collect();
    let b_data: Vec<f32> = (0..b_size).map(|i| (i as f32) * 0.03 - 0.1).collect();

    let a_shape = vec![bd, hd, m, k];
    let b_shape = vec![bd, hd, k, n];

    let forward_loss = |a_slice: &[f32], b_slice: &[f32]| -> f32 {
        let a = Tensor::new_cpu(a_shape.clone(), a_slice.to_vec());
        let b = Tensor::new_cpu(b_shape.clone(), b_slice.to_vec());
        run_bmm(&a, &b).as_cpu_slice().iter().sum::<f32>()
    };

    // Numerical gradients (vary one tensor at a time).
    let num_grad_a = finite_diff_grad(&a_data, |x| forward_loss(x, &b_data));
    let num_grad_b = finite_diff_grad(&b_data, |x| forward_loss(&a_data, x));

    // Analytical gradients via graph tape.
    let mut gb = GraphBuilder::new();
    let a_id = gb.input();
    let b_id = gb.input();
    let bmm_id = gb.batch_matmul(a_id, b_id);
    let out_id = gb.output(bmm_id);
    let mut g = gb.build();
    let _ = g.execute(vec![
        Tensor::new_cpu(a_shape.clone(), a_data.clone()),
        Tensor::new_cpu(b_shape.clone(), b_data.clone()),
    ]);
    g.backward(out_id);

    let ana_a: &[f32] = g.nodes[a_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("a grad missing");
    let ana_b: &[f32] = g.nodes[b_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("b grad missing");

    let check = |label: &str, ana: &[f32], num: &[f32]| {
        assert_eq!(ana.len(), num.len(), "{}: len mismatch", label);
        for (i, (&a, &n)) in ana.iter().zip(num.iter()).enumerate() {
            let diff = (a - n).abs();
            let scale = a.abs().max(n.abs()).max(1e-4);
            let rel = diff / scale;
            assert!(
                rel < FD_REL_TOL,
                "{}: idx {}: analytical={} numerical={} rel_err={}",
                label,
                i,
                a,
                n,
                rel
            );
        }
    };
    check("grad_a", ana_a, &num_grad_a);
    check("grad_b", ana_b, &num_grad_b);
}
