use atenia_engine::gpu::tensor::manager::GpuTensorManager;
use atenia_engine::gpu::autodiff::linear_backward::LinearBackwardGPU;

#[test]
fn test_linear_backward_real_gpu() {
    let mgr = match GpuTensorManager::new() {
        Ok(m) => m,
        Err(_) => return, // sin GPU -> skip
    };

    // X [2x2], W [2x2], dOut [2x2]
    let x = vec![
        1.0f32, 2.0,
        3.0,    4.0,
    ];
    let w = vec![
        5.0f32, 6.0,
        7.0,    8.0,
    ];
    let dout = vec![
        1.0f32, -1.0,
        2.0,    0.5,
    ];

    let x_gpu = match mgr.from_cpu_vec(&x, 2, 2) {
        Ok(v) => v,
        Err(_) => return,
    };
    let w_gpu = match mgr.from_cpu_vec(&w, 2, 2) {
        Ok(v) => v,
        Err(_) => return,
    };
    let dout_gpu = match mgr.from_cpu_vec(&dout, 2, 2) {
        Ok(v) => v,
        Err(_) => return,
    };

    let (d_x_gpu, d_w_gpu, d_b_gpu) = match LinearBackwardGPU::run(&mgr, &x_gpu, &w_gpu, &dout_gpu) {
        Ok(v) => v,
        Err(_) => return,
    };

    let d_x = match mgr.to_cpu_vec(&d_x_gpu) {
        Ok(v) => v,
        Err(_) => return,
    };
    let d_w = match mgr.to_cpu_vec(&d_w_gpu) {
        Ok(v) => v,
        Err(_) => return,
    };
    let d_b = match mgr.to_cpu_vec(&d_b_gpu) {
        Ok(v) => v,
        Err(_) => return,
    };

    // CPU reference
    let m = 2;
    let k = 2;
    let n = 2;

    // dX_ref = dOut * W^T
    let mut d_x_ref = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            let mut acc = 0.0;
            for col in 0..n {
                acc += dout[i * n + col] * w[j * n + col];
            }
            d_x_ref[i * k + j] = acc;
        }
    }

    // dW_ref = X^T * dOut
    let mut d_w_ref = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            let mut acc = 0.0;
            for row in 0..m {
                acc += x[row * k + i] * dout[row * n + j];
            }
            d_w_ref[i * n + j] = acc;
        }
    }

    // dB_ref = sum_i dOut[i, :]
    let mut d_b_ref = vec![0.0f32; n];
    for j in 0..n {
        let mut acc = 0.0;
        for i in 0..m {
            acc += dout[i * n + j];
        }
        d_b_ref[j] = acc;
    }

    // Compare with a small FP tolerance
    for i in 0..d_x.len() {
        assert!((d_x[i] - d_x_ref[i]).abs() < 1e-4, "dX mismatch at {}: {} vs {}", i, d_x[i], d_x_ref[i]);
    }
    for i in 0..d_w.len() {
        assert!((d_w[i] - d_w_ref[i]).abs() < 1e-4, "dW mismatch at {}: {} vs {}", i, d_w[i], d_w_ref[i]);
    }
    for i in 0..d_b.len() {
        assert!((d_b[i] - d_b_ref[i]).abs() < 1e-4, "dB mismatch at {}: {} vs {}", i, d_b[i], d_b_ref[i]);
    }
}
