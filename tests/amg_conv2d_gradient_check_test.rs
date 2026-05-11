//! Gradient check: AMG's `Conv2D` analytical backward vs central-difference
//! numerical gradient on a small hand-chosen setup.
//!
//! Technique: for loss `L(x) = sum(Conv2D(x, ...))`, the numerical
//! gradient at each element `x_i` is
//!   `(L(x + h*e_i) - L(x - h*e_i)) / (2h)`,
//! and the analytical gradient is
//!   `execute_conv2d_backward(..., out_grad=ones).grad_*`.
//!
//! Tolerance is a relative error of `1e-2`. Gradient checking with
//! f32 is inherently noisy (cancellation in the central difference);
//! 1e-2 is the standard threshold that avoids false positives without
//! hiding real bugs. A larger discrepancy indicates a backward bug.
//!
//! Inputs are small (≤ 36 elements per tensor) so the O(N) forward
//! calls of finite differences remain fast.

use atenia_engine::amg::nodes::Conv2DConfig;
use atenia_engine::amg::ops::conv2d::{
    execute_conv2d as amg_conv2d, execute_conv2d_backward as amg_conv2d_backward,
};
use atenia_engine::tensor::Tensor as AmgTensor;

const H: f32 = 1e-3;
const REL_TOL: f32 = 1e-2;

fn amg_tensor(shape: Vec<usize>, data: Vec<f32>) -> AmgTensor {
    AmgTensor::new_cpu(shape, data)
}

/// Central-difference gradient of a scalar function `f: Vec<f32> -> f32`
/// around `base`, with step `H`. Non-destructive (restores perturbed
/// slot after each probe).
fn finite_diff_grad<F>(base: &[f32], forward: F) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let mut g = vec![0.0_f32; base.len()];
    let mut perturbed = base.to_vec();
    for i in 0..base.len() {
        let orig = perturbed[i];
        perturbed[i] = orig + H;
        let f_plus = forward(&perturbed);
        perturbed[i] = orig - H;
        let f_minus = forward(&perturbed);
        perturbed[i] = orig;
        g[i] = (f_plus - f_minus) / (2.0 * H);
    }
    g
}

/// Compare analytical vs numerical gradients with mixed abs/rel
/// tolerance. The `1e-4` floor on `scale` keeps small gradient
/// components from inflating the relative error.
fn assert_grad_close(analytical: &[f32], numerical: &[f32], ctx: &str) {
    assert_eq!(analytical.len(), numerical.len(), "{}: len mismatch", ctx);
    for (i, (&a, &n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        let diff = (a - n).abs();
        let scale = a.abs().max(n.abs()).max(1e-4_f32);
        let rel = diff / scale;
        assert!(
            rel < REL_TOL,
            "{}: idx {}: analytical={} numerical={} rel_err={} > tol={}",
            ctx,
            i,
            a,
            n,
            rel,
            REL_TOL
        );
    }
}

/// Fixed setup used by all three gradient checks in this file.
fn fixtures() -> (
    Vec<usize>,
    Vec<f32>,
    Vec<usize>,
    Vec<f32>,
    Vec<f32>,
    Conv2DConfig,
) {
    // 1 sample, 2 input channels, 3x3 spatial.
    let input_shape = vec![1, 2, 3, 3];
    let input_data: Vec<f32> = (0..18).map(|i| (i as f32 + 1.0) * 0.05).collect();

    // 2 output channels, 2 input channels, 2x2 kernel.
    let weight_shape = vec![2, 2, 2, 2];
    let weight_data: Vec<f32> = vec![
        0.5, -0.3, 0.2, 0.1, // oc=0, ic=0
        -0.1, 0.4, 0.3, -0.2, // oc=0, ic=1
        0.2, 0.1, -0.4, 0.5, // oc=1, ic=0
        0.3, -0.2, 0.1, 0.4, // oc=1, ic=1
    ];
    let bias_data: Vec<f32> = vec![0.1, -0.2];

    let cfg = Conv2DConfig::new((1, 1), (0, 0));
    (
        input_shape,
        input_data,
        weight_shape,
        weight_data,
        bias_data,
        cfg,
    )
}

#[test]
fn conv2d_grad_input_matches_finite_diff() {
    let (input_shape, input_data, weight_shape, weight_data, _, cfg) = fixtures();

    let weight = amg_tensor(weight_shape, weight_data);

    // Analytical gradient: loss = sum(out) ⇒ out_grad = ones.
    let input = amg_tensor(input_shape.clone(), input_data.clone());
    let out = amg_conv2d(&input, &weight, None, &cfg);
    let ones = amg_tensor(out.shape.clone(), vec![1.0; out.numel()]);
    let grads = amg_conv2d_backward(&input, &weight, None, &ones, &cfg);

    // Numerical gradient: perturb each input element and sum forward output.
    let grad_num = finite_diff_grad(&input_data, |x| {
        let t = amg_tensor(input_shape.clone(), x.to_vec());
        amg_conv2d(&t, &weight, None, &cfg)
            .as_cpu_slice()
            .iter()
            .sum()
    });

    assert_grad_close(&grads.grad_input, &grad_num, "grad_input");
}

#[test]
fn conv2d_grad_weight_matches_finite_diff() {
    let (input_shape, input_data, weight_shape, weight_data, _, cfg) = fixtures();

    let input = amg_tensor(input_shape, input_data);

    // Analytical.
    let weight = amg_tensor(weight_shape.clone(), weight_data.clone());
    let out = amg_conv2d(&input, &weight, None, &cfg);
    let ones = amg_tensor(out.shape.clone(), vec![1.0; out.numel()]);
    let grads = amg_conv2d_backward(&input, &weight, None, &ones, &cfg);

    // Numerical: perturb each weight element.
    let grad_num = finite_diff_grad(&weight_data, |w| {
        let wt = amg_tensor(weight_shape.clone(), w.to_vec());
        amg_conv2d(&input, &wt, None, &cfg)
            .as_cpu_slice()
            .iter()
            .sum()
    });

    assert_grad_close(&grads.grad_weight, &grad_num, "grad_weight");
}

#[test]
fn conv2d_grad_bias_matches_finite_diff() {
    let (input_shape, input_data, weight_shape, weight_data, bias_data, cfg) = fixtures();

    let input = amg_tensor(input_shape, input_data);
    let weight = amg_tensor(weight_shape, weight_data);
    let bias = amg_tensor(vec![bias_data.len()], bias_data.clone());

    // Analytical.
    let out = amg_conv2d(&input, &weight, Some(&bias), &cfg);
    let ones = amg_tensor(out.shape.clone(), vec![1.0; out.numel()]);
    let grads = amg_conv2d_backward(&input, &weight, Some(&bias), &ones, &cfg);
    let grad_bias = grads
        .grad_bias
        .as_ref()
        .expect("grad_bias must be present when bias was provided");

    // Numerical.
    let grad_num = finite_diff_grad(&bias_data, |b| {
        let bt = amg_tensor(vec![b.len()], b.to_vec());
        amg_conv2d(&input, &weight, Some(&bt), &cfg)
            .as_cpu_slice()
            .iter()
            .sum()
    });

    assert_grad_close(grad_bias, &grad_num, "grad_bias");
}
