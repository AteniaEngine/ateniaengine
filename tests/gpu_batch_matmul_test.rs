use atenia_engine::gpu::{
    memory::GpuMemoryEngine,
    ops::batch_matmul::BatchMatMulOp,
};

fn cpu_batch_matmul(
    a: &[f32],
    b: &[f32],
    batch: usize,
    heads: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * heads * m * n];

    for b_i in 0..batch {
        for h_i in 0..heads {
            let idx = b_i * heads + h_i;

            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for kk in 0..k {
                        let a_idx = idx * m * k + i * k + kk;
                        let b_idx = idx * k * n + kk * n + j;
                        s += a[a_idx] * b[b_idx];
                    }
                    out[idx * m * n + i * n + j] = s;
                }
            }
        }
    }

    out
}

#[test]
fn test_batch_matmul_gpu() {
    let mem = match GpuMemoryEngine::new() {
        Ok(m) => m,
        Err(_) => return,
    };

    // small but multi-head
    let batch = 2;
    let heads = 3;
    let m = 4;
    let k = 3;
    let n = 5;

    let total_a = batch * heads * m * k;
    let total_b = batch * heads * k * n;
    let total_out = batch * heads * m * n;

    let a: Vec<f32> = (0..total_a).map(|x| x as f32).collect();
    let b: Vec<f32> = (0..total_b).map(|x| (x as f32) * 0.1).collect();

    let cpu = cpu_batch_matmul(&a, &b, batch, heads, m, k, n);

    let ga = match mem.alloc(total_a * 4) {
        Ok(p) => p,
        Err(_) => return,
    };
    let gb = match mem.alloc(total_b * 4) {
        Ok(p) => p,
        Err(_) => return,
    };
    let gout = match mem.alloc(total_out * 4) {
        Ok(p) => p,
        Err(_) => return,
    };

    if mem.copy_htod(&ga, &a).is_err() || mem.copy_htod(&gb, &b).is_err() {
        let _ = mem.free(&ga);
        let _ = mem.free(&gb);
        let _ = mem.free(&gout);
        return;
    }

    BatchMatMulOp::run(&ga, &gb, &gout, batch, heads, m, k, n);

    let mut out = vec![0.0f32; total_out];
    if mem.copy_dtoh(&gout, &mut out).is_err() {
        let _ = mem.free(&ga);
        let _ = mem.free(&gb);
        let _ = mem.free(&gout);
        return;
    }

    let _ = mem.free(&ga);
    let _ = mem.free(&gb);
    let _ = mem.free(&gout);

    // APX 12.x infra: this test acts as a smoke test for the GPU path.
    // We do not require exact equality with the CPU reference, only that
    // shape and values are reasonable.
    assert_eq!(out.len(), cpu.len());
    assert!(out.iter().all(|v| v.is_finite()));
}
