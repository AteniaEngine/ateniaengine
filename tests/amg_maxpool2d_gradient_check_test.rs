//! Gradient check: AMG's `MaxPool2D` analytical backward vs
//! central-difference numerical gradient.
//!
//! MaxPool's gradient is piecewise constant: each input element
//! either is the argmax of some output window (gradient = out_grad
//! of that window) or it isn't (gradient = 0). For the loss
//! `L = sum(out)` with `out_grad = ones`, the gradient is either
//! exactly 1 or exactly 0 per input element.
//!
//! **Tie-free input design**: we pick input values well-separated
//! (minimum spacing of 0.1) so that the `H = 1e-3` perturbation
//! cannot change the argmax. If two values in a window were within
//! 2*H of each other, a perturbation could flip the argmax and the
//! numerical gradient would jump discontinuously, producing a false
//! test failure that does not reflect a backward bug.
//!
//! Tolerance is `1e-2` relative — same rationale as the conv2d
//! gradient check: f32 finite differences are inherently noisy.

use atenia_engine::amg::nodes::MaxPool2DConfig;
use atenia_engine::amg::ops::maxpool2d::{
    execute_maxpool2d as amg_maxpool2d, execute_maxpool2d_backward as amg_maxpool2d_backward,
};
use atenia_engine::tensor::Tensor as AmgTensor;

const H: f32 = 1e-3;
const REL_TOL: f32 = 1e-2;

fn amg_tensor(shape: Vec<usize>, data: Vec<f32>) -> AmgTensor {
    AmgTensor::new_cpu(shape, data)
}

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

fn assert_grad_close(analytical: &[f32], numerical: &[f32], ctx: &str) {
    assert_eq!(analytical.len(), numerical.len(), "{}: len mismatch", ctx);
    for (i, (&a, &n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        let diff = (a - n).abs();
        let scale = a.abs().max(n.abs()).max(1e-4_f32);
        let rel = diff / scale;
        assert!(
            rel < REL_TOL,
            "{}: idx {}: analytical={} numerical={} rel_err={} > tol={}",
            ctx, i, a, n, rel, REL_TOL
        );
    }
}

#[test]
fn maxpool2d_grad_input_matches_finite_diff_non_overlapping() {
    // 4x4 input, all distinct values at least 0.1 apart → tie-free
    // at H = 1e-3 perturbation granularity.
    let input_shape = vec![1, 1, 4, 4];
    let input_data: Vec<f32> = (0..16).map(|i| 0.1 + i as f32 * 0.1).collect();
    let cfg = MaxPool2DConfig::non_overlapping((2, 2));

    // Analytical gradient: loss = sum(out) ⇒ out_grad = ones.
    let input = amg_tensor(input_shape.clone(), input_data.clone());
    let out = amg_maxpool2d(&input, &cfg);
    let ones = amg_tensor(out.shape.clone(), vec![1.0; out.numel()]);
    let grad_analytical = amg_maxpool2d_backward(&input, &ones, &cfg);

    // Numerical.
    let grad_num = finite_diff_grad(&input_data, |x| {
        let t = amg_tensor(input_shape.clone(), x.to_vec());
        amg_maxpool2d(&t, &cfg).as_cpu_slice().iter().sum()
    });

    assert_grad_close(&grad_analytical, &grad_num, "grad_input (non-overlapping)");
}

#[test]
fn maxpool2d_grad_input_matches_finite_diff_with_padding() {
    // 3x3 input, kernel 2x2, stride 1, padding 1 → 4x4 output.
    // Values spaced at 0.1 to avoid ties under H perturbation.
    let input_shape = vec![1, 1, 3, 3];
    let input_data: Vec<f32> = (0..9).map(|i| 0.1 + i as f32 * 0.1).collect();
    let cfg = MaxPool2DConfig::new((2, 2), (1, 1), (1, 1));

    let input = amg_tensor(input_shape.clone(), input_data.clone());
    let out = amg_maxpool2d(&input, &cfg);
    let ones = amg_tensor(out.shape.clone(), vec![1.0; out.numel()]);
    let grad_analytical = amg_maxpool2d_backward(&input, &ones, &cfg);

    let grad_num = finite_diff_grad(&input_data, |x| {
        let t = amg_tensor(input_shape.clone(), x.to_vec());
        amg_maxpool2d(&t, &cfg).as_cpu_slice().iter().sum()
    });

    assert_grad_close(&grad_analytical, &grad_num, "grad_input (with padding)");
}
