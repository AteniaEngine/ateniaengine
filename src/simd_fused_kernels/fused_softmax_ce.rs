#[inline]
pub fn fused_softmax_ce_forward(
    logits: &[f32],
    targets: &[usize],
    batch: usize,
    classes: usize,
) -> Vec<f32> {
    #[cfg(debug_assertions)]
    eprintln!("[APX-2.5] fused_softmax_ce_backward invoked");
    #[cfg(debug_assertions)]
    eprintln!("[APX-2.5] fused_softmax_ce_forward invoked");
    let mut losses = vec![0.0; batch];

    for b in 0..batch {
        let start = b * classes;
        let row = &logits[start..start + classes];

        let mut max_val = f32::NEG_INFINITY;
        for &v in row {
            if v > max_val {
                max_val = v;
            }
        }

        let mut sum = 0.0;
        let mut exp_row = vec![0.0; classes];

        for i in 0..classes {
            exp_row[i] = (row[i] - max_val).exp();
            sum += exp_row[i];
        }

        let softmax_target = exp_row[targets[b]] / sum;
        losses[b] = -softmax_target.ln();
    }

    losses
}

#[inline]
pub fn fused_softmax_ce_backward(
    logits: &[f32],
    targets: &[usize],
    grad_out: &[f32],
    batch: usize,
    classes: usize,
) -> Vec<f32> {
    let mut grad = vec![0.0; batch * classes];

    for b in 0..batch {
        let start = b * classes;
        let row = &logits[start..start + classes];

        let mut max_val = f32::NEG_INFINITY;
        for &v in row {
            if v > max_val {
                max_val = v;
            }
        }

        let mut sum = 0.0;
        let mut exp_row = vec![0.0; classes];
        for i in 0..classes {
            exp_row[i] = (row[i] - max_val).exp();
            sum += exp_row[i];
        }

        for i in 0..classes {
            grad[start + i] = (exp_row[i] / sum) * grad_out[b];
        }

        grad[start + targets[b]] -= grad_out[b];
    }

    grad
}
