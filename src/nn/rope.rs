//! Rotary Positional Embedding (RoPE) — half-split layout (HuggingFace
//! convention).
//!
//! See [`crate::amg::nodes::NodeType::RoPE`] for the architectural
//! contract (layout, positions, parameters).

use crate::tensor::Tensor;

/// Compute inverse frequencies `theta_i = base_freq^(-2i/head_dim)` for
/// `i in 0..head_dim/2`. Shared by [`apply_rope`] and
/// [`apply_rope_backward`] so that any change to the schedule applies
/// to both directions.
fn compute_inv_freqs(head_dim: usize, base_freq: u32) -> Vec<f32> {
    let half = head_dim / 2;
    let base_freq_f32 = base_freq as f32;
    (0..half)
        .map(|i| {
            let exp = (i as f32) * 2.0 / (head_dim as f32);
            1.0_f32 / base_freq_f32.powf(exp)
        })
        .collect()
}

fn validate_shape(shape: &[usize], head_dim: usize, op: &str) {
    assert_eq!(
        shape.len(),
        4,
        "{}: expects 4D tensor [batch, seq_len, n_heads, head_dim], got {:?}",
        op,
        shape
    );
    assert_eq!(
        shape[3], head_dim,
        "{}: tensor last dim ({}) does not match head_dim ({})",
        op, shape[3], head_dim
    );
    assert!(head_dim > 0, "{}: head_dim must be positive", op);
    assert!(
        head_dim % 2 == 0,
        "{}: head_dim must be even, got {}",
        op,
        head_dim
    );
}

/// Apply RoPE (half-split layout) to input tensor.
///
/// Input shape:  `[batch, seq_len, n_heads, head_dim]`.
/// Output shape: same as input.
/// Positions:    implicit `[0..seq_len)`.
///
/// Half-split formula (per `(b, s, h)` slice, for `i in 0..head_dim/2`):
/// ```text
///   angle    = s * base_freq^(-2i/head_dim)
///   y[i]            =  x[i]      * cos(angle) - x[i + half] * sin(angle)
///   y[i + half]     =  x[i + half]* cos(angle) + x[i]       * sin(angle)
/// ```
///
/// Frequencies are computed on the fly. For typical TinyLlama-class
/// shapes the cost is ~1.3 ms per forward pass, which is negligible
/// relative to the matmul-dominated workload around it.
///
/// # Panics
/// See [`validate_shape`].
pub fn apply_rope(x: &Tensor, head_dim: usize, base_freq: u32) -> Tensor {
    let shape = &x.shape;
    validate_shape(shape, head_dim, "apply_rope");

    let batch = shape[0];
    let seq_len = shape[1];
    let n_heads = shape[2];
    let half = head_dim / 2;
    let inv_freqs = compute_inv_freqs(head_dim, base_freq);

    let x_slice = x.as_cpu_slice();
    let mut output = vec![0.0_f32; x_slice.len()];

    for b in 0..batch {
        for s in 0..seq_len {
            let s_f32 = s as f32;
            for h in 0..n_heads {
                let base_offset =
                    b * seq_len * n_heads * head_dim + s * n_heads * head_dim + h * head_dim;
                for i in 0..half {
                    let angle = s_f32 * inv_freqs[i];
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    let x_first = x_slice[base_offset + i];
                    let x_second = x_slice[base_offset + i + half];

                    output[base_offset + i] = x_first * cos_a - x_second * sin_a;
                    output[base_offset + i + half] = x_second * cos_a + x_first * sin_a;
                }
            }
        }
    }

    Tensor::new_cpu(shape.clone(), output)
}

/// Backward pass for RoPE.
///
/// Given the upstream gradient `out_grad` (same shape as the forward
/// output) and the forward configuration, returns `grad_x` (same shape
/// as the input). RoPE has no learnable parameters, so this is the
/// only gradient produced.
///
/// Derivation: forward applies a 2D rotation by `+angle` on each
/// `(i, i+half)` pair. The Jacobian of a rotation matrix is its
/// transpose, i.e. a rotation by `-angle`. Therefore:
/// ```text
///   grad_x[i]      =  out_grad[i]      * cos(angle) + out_grad[i+half] * sin(angle)
///   grad_x[i+half] = -out_grad[i]      * sin(angle) + out_grad[i+half] * cos(angle)
/// ```
/// `head_dim` and `base_freq` are model constants and have no gradient.
///
/// # Panics
/// Panics if `out_grad`'s shape does not satisfy the same constraints
/// as the forward input.
pub fn apply_rope_backward(out_grad: &[f32], shape: &[usize], head_dim: usize, base_freq: u32) -> Vec<f32> {
    validate_shape(shape, head_dim, "apply_rope_backward");
    let expected: usize = shape.iter().product();
    assert_eq!(
        out_grad.len(),
        expected,
        "apply_rope_backward: out_grad length {} does not match shape product {} (shape = {:?})",
        out_grad.len(),
        expected,
        shape
    );

    let batch = shape[0];
    let seq_len = shape[1];
    let n_heads = shape[2];
    let half = head_dim / 2;
    let inv_freqs = compute_inv_freqs(head_dim, base_freq);

    let mut grad_x = vec![0.0_f32; out_grad.len()];

    for b in 0..batch {
        for s in 0..seq_len {
            let s_f32 = s as f32;
            for h in 0..n_heads {
                let base_offset =
                    b * seq_len * n_heads * head_dim + s * n_heads * head_dim + h * head_dim;
                for i in 0..half {
                    let angle = s_f32 * inv_freqs[i];
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    let g_first = out_grad[base_offset + i];
                    let g_second = out_grad[base_offset + i + half];

                    // Rotate grad by -angle (transpose of forward rotation).
                    grad_x[base_offset + i] = g_first * cos_a + g_second * sin_a;
                    grad_x[base_offset + i + half] = -g_first * sin_a + g_second * cos_a;
                }
            }
        }
    }

    grad_x
}
