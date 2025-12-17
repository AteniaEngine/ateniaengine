use atenia_engine::apx6::matmul_tiled_6_3b::matmul_tiled_6_3b;

fn matmul_scalar(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

#[test]
fn correctness_6_3b_small() {
    let sizes = [
        (32usize, 32usize, 32usize),
        (64usize, 64usize, 64usize),
        (48usize, 40usize, 32usize),
    ];

    for (m, k, n) in sizes {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.271).cos()).collect();

        let ref_out = matmul_scalar(&a, &b, m, k, n);
        let mut tiled_out = vec![0.0f32; m * n];
        matmul_tiled_6_3b(&a, &b, &mut tiled_out, m, k, n);

        for i in 0..ref_out.len() {
            assert!(
                (ref_out[i] - tiled_out[i]).abs() < 1e-5,
                "mismatch at {}: ref={} tiled={}",
                i,
                ref_out[i],
                tiled_out[i]
            );
        }
    }
}
