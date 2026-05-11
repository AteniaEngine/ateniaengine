//! Tests for `NodeType::BroadcastMul` (prerequisite for M4.5-b1).
//!
//! Coverage:
//!   1. same-shape: equivalent to element-wise Mul
//!   2. last-dim broadcast on rank-3
//!   3. RMSNorm-gamma case `[batch, seq, hidden]` × `[1, 1, hidden]`
//!   4. forward bit-exact vs hand computation
//!   5. backward grad check vs central diff (input A)
//!   6. backward grad check vs central diff (input B with broadcast)

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::Tensor;

fn run_broadcast_mul(a: &Tensor, b: &Tensor) -> Tensor {
    let mut gb = GraphBuilder::new();
    let a_id = gb.input();
    let b_id = gb.input();
    let m_id = gb.broadcast_mul(a_id, b_id);
    let _ = gb.output(m_id);
    let mut g = gb.build();
    let out = g.execute(vec![a.clone(), b.clone()]);
    out.into_iter().next().expect("one output")
}

#[test]
fn broadcast_mul_same_shape_matches_elementwise_mul() {
    let shape = vec![2_usize, 3, 4];
    let n: usize = shape.iter().product();
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * -0.07 + 0.3).collect();
    let a = Tensor::new_cpu(shape.clone(), a_data.clone());
    let b = Tensor::new_cpu(shape.clone(), b_data.clone());

    let got = run_broadcast_mul(&a, &b);
    assert_eq!(got.shape, shape);

    for i in 0..n {
        let want = a_data[i] * b_data[i];
        let g = got.as_cpu_slice()[i];
        assert!(
            (g - want).abs() < 1e-7,
            "idx {}: got {} want {}",
            i,
            g,
            want
        );
    }
}

#[test]
fn broadcast_mul_last_dim_broadcast_on_rank3() {
    // a: [2, 3, 4]; b: [1, 1, 4] (broadcast over the first two dims).
    let shape_a = vec![2_usize, 3, 4];
    let shape_b = vec![1_usize, 1, 4];
    let a_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let b_data: Vec<f32> = vec![10.0, 100.0, 1000.0, 10000.0];
    let a = Tensor::new_cpu(shape_a.clone(), a_data.clone());
    let b = Tensor::new_cpu(shape_b.clone(), b_data.clone());

    let got = run_broadcast_mul(&a, &b);
    assert_eq!(got.shape, shape_a);

    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                let lin = i * 12 + j * 4 + k;
                let want = a_data[lin] * b_data[k];
                let g = got.as_cpu_slice()[lin];
                assert!(
                    (g - want).abs() < 1e-7,
                    "({},{},{}): got {} want {}",
                    i,
                    j,
                    k,
                    g,
                    want
                );
            }
        }
    }
}

#[test]
fn broadcast_mul_rmsnorm_gamma_pattern() {
    // [batch=2, seq=4, hidden=8] × [1, 1, hidden=8] — the exact shape
    // pattern the TinyLlama builder will use for `RMSNorm × γ`.
    let (b, s, h) = (2_usize, 4, 8);
    let n = b * s * h;
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32 % 7.0) * 0.13).collect();
    let g_data: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let a = Tensor::new_cpu(vec![b, s, h], a_data.clone());
    let gamma = Tensor::new_cpu(vec![1, 1, h], g_data.clone());

    let got = run_broadcast_mul(&a, &gamma);
    for bi in 0..b {
        for si in 0..s {
            for hi in 0..h {
                let lin = (bi * s + si) * h + hi;
                let want = a_data[lin] * g_data[hi];
                let g = got.as_cpu_slice()[lin];
                assert!(
                    (g - want).abs() < 1e-7,
                    "({},{},{}): got {} want {}",
                    bi,
                    si,
                    hi,
                    g,
                    want
                );
            }
        }
    }
}

#[test]
fn broadcast_mul_forward_bit_exact_hand_computed() {
    // Tiny case verifiable by hand.
    // a = [[[1, 2], [3, 4]]] shape [1, 2, 2]
    // b = [[[10, 100]]]      shape [1, 1, 2]   (broadcast along dim 1)
    // out[b,s,h] = a[b,s,h] * b[0, 0, h]
    // Expected:
    //   (0,0,0)=1*10=10   (0,0,1)=2*100=200
    //   (0,1,0)=3*10=30   (0,1,1)=4*100=400
    let a = Tensor::new_cpu(vec![1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Tensor::new_cpu(vec![1, 1, 2], vec![10.0, 100.0]);
    let got = run_broadcast_mul(&a, &b);
    assert_eq!(got.shape, vec![1, 2, 2]);
    let expected = vec![10.0_f32, 200.0, 30.0, 400.0];
    for (i, (g, e)) in got.as_cpu_slice().iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-7, "idx {}: got {} want {}", i, g, e);
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

fn assert_grad_close(label: &str, ana: &[f32], num: &[f32]) {
    assert_eq!(ana.len(), num.len(), "{}: len", label);
    for (i, (&a, &n)) in ana.iter().zip(num.iter()).enumerate() {
        let diff = (a - n).abs();
        let scale = a.abs().max(n.abs()).max(1e-4);
        let rel = diff / scale;
        assert!(
            rel < FD_REL_TOL,
            "{}: idx {}: ana={} num={} rel={}",
            label,
            i,
            a,
            n,
            rel
        );
    }
}

#[test]
fn broadcast_mul_backward_grad_a_matches_finite_diff() {
    let shape_a = vec![1_usize, 3, 4];
    let shape_b = vec![1_usize, 1, 4];
    let a_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1).collect();
    let b_data: Vec<f32> = vec![0.5, -0.3, 0.7, 0.2];

    // Numerical: scalar loss = sum(broadcast_mul(a, b))
    let num = finite_diff_grad(&a_data, |x| {
        let a = Tensor::new_cpu(shape_a.clone(), x.to_vec());
        let b = Tensor::new_cpu(shape_b.clone(), b_data.clone());
        run_broadcast_mul(&a, &b).as_cpu_slice().iter().sum::<f32>()
    });

    // Analytical via tape.
    let mut gb = GraphBuilder::new();
    let a_id = gb.input();
    let b_id = gb.input();
    let m_id = gb.broadcast_mul(a_id, b_id);
    let out_id = gb.output(m_id);
    let mut g = gb.build();
    let _ = g.execute(vec![
        Tensor::new_cpu(shape_a.clone(), a_data.clone()),
        Tensor::new_cpu(shape_b.clone(), b_data.clone()),
    ]);
    g.backward(out_id);
    let ana: &[f32] = g.nodes[a_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("grad_a missing");

    assert_grad_close("grad_a", ana, &num);
}

#[test]
fn broadcast_mul_backward_grad_b_reduces_over_broadcast_dims() {
    let shape_a = vec![1_usize, 3, 4];
    let shape_b = vec![1_usize, 1, 4]; // broadcast over dim 1
    let a_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 + 0.05).collect();
    let b_data: Vec<f32> = vec![0.5, -0.3, 0.7, 0.2];

    // Numerical grad of b: for each b_k, sum over (b,s) of out_grad[b,s,k] * a[b,s,k]
    // where out_grad = ones. Equivalent to sum_{b,s} a[b,s,k].
    // Verify the tape computes this.
    let num = finite_diff_grad(&b_data, |x| {
        let a = Tensor::new_cpu(shape_a.clone(), a_data.clone());
        let b = Tensor::new_cpu(shape_b.clone(), x.to_vec());
        run_broadcast_mul(&a, &b).as_cpu_slice().iter().sum::<f32>()
    });

    let mut gb = GraphBuilder::new();
    let a_id = gb.input();
    let b_id = gb.input();
    let m_id = gb.broadcast_mul(a_id, b_id);
    let out_id = gb.output(m_id);
    let mut g = gb.build();
    let _ = g.execute(vec![
        Tensor::new_cpu(shape_a.clone(), a_data.clone()),
        Tensor::new_cpu(shape_b.clone(), b_data.clone()),
    ]);
    g.backward(out_id);
    let ana: &[f32] = g.nodes[b_id]
        .output
        .as_ref()
        .and_then(|t| t.grad.as_ref())
        .expect("grad_b missing");

    // Sanity: ana[k] should equal sum over (i,j) of a[i,j,k]
    for k in 0..4 {
        let mut sum = 0.0_f32;
        for i in 0..3 {
            sum += a_data[i * 4 + k];
        }
        assert!(
            (ana[k] - sum).abs() < 1e-4,
            "grad_b[{}]: ana {} != hand-sum {}",
            k,
            ana[k],
            sum
        );
    }

    assert_grad_close("grad_b", ana, &num);
}
