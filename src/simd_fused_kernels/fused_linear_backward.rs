pub fn fused_linear_backward_dx(
    dy: &[f32],
    w: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    #[cfg(debug_assertions)]
    eprintln!("[APX-2.5] fused_linear_backward_dw invoked");
    let mut dx = vec![0.0; batch * in_dim];

    for b in 0..batch {
        for i in 0..in_dim {
            let mut acc = 0.0;
            for o in 0..out_dim {
                acc += dy[b * out_dim + o] * w[i * out_dim + o];
            }
            dx[b * in_dim + i] = acc;
        }
    }

    dx
}

pub fn fused_linear_backward_dw(
    x: &[f32],
    dy: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut dw = vec![0.0; in_dim * out_dim];

    for i in 0..in_dim {
        for o in 0..out_dim {
            let mut acc = 0.0;
            for b in 0..batch {
                acc += x[b * in_dim + i] * dy[b * out_dim + o];
            }
            dw[i * out_dim + o] = acc;
        }
    }

    dw
}

pub fn fused_linear_backward_db(dy: &[f32], batch: usize, out_dim: usize) -> Vec<f32> {
    let mut db = vec![0.0; out_dim];

    for b in 0..batch {
        for o in 0..out_dim {
            db[o] += dy[b * out_dim + o];
        }
    }

    db
}
