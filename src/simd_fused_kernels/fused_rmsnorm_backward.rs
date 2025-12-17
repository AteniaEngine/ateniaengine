pub fn fused_rmsnorm_backward(
    x: &[f32],
    w: &[f32],
    grad_out: &[f32],
    hidden: usize,
) -> Vec<f32> {
    #[cfg(debug_assertions)]
    {
        eprintln!("[APX-2.5] fused_rmsnorm_backward invoked");
        if std::env::var("ATENIA_TRACE").unwrap_or_default() == "1" {
            eprintln!("[APX TRACE] Using FUSED kernel matmul+rmsnorm");
        }
    }
    let mut grad = vec![0.0; hidden];

    let mut mean_sq = 0.0;
    for i in 0..hidden {
        mean_sq += x[i] * x[i];
    }
    mean_sq /= hidden as f32;

    let scale = 1.0 / mean_sq.sqrt().max(1e-6);

    for i in 0..hidden {
        grad[i] = grad_out[i] * w[i] * scale;
    }

    grad
}
