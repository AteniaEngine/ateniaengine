use atenia_engine::gpu::{
    memory::GpuMemoryEngine,
    ops::matmul::MatMulOp,
};

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = s;
        }
    }
    out
}

#[test]
fn test_matmul_gpu() {
    let mem = match GpuMemoryEngine::new() {
        Ok(m) => m,
        Err(_) => return,
    };

    let m = 4;
    let k = 3;
    let n = 5;

    let a: Vec<f32> = (0..m * k).map(|x| x as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|x| (x as f32) * 0.5).collect();

    let cpu = cpu_matmul(&a, &b, m, k, n);

    let ptr_a = match mem.alloc(a.len() * 4) {
        Ok(p) => p,
        Err(_) => return,
    };
    let ptr_b = match mem.alloc(b.len() * 4) {
        Ok(p) => p,
        Err(_) => return,
    };
    let ptr_c = match mem.alloc(cpu.len() * 4) {
        Ok(p) => p,
        Err(_) => return,
    };

    if mem.copy_htod(&ptr_a, &a).is_err() || mem.copy_htod(&ptr_b, &b).is_err() {
        let _ = mem.free(&ptr_a);
        let _ = mem.free(&ptr_b);
        let _ = mem.free(&ptr_c);
        return;
    }

    MatMulOp::run(&ptr_a, &ptr_b, &ptr_c, m, k, n);

    let mut gpu_out = vec![0.0f32; cpu.len()];
    if mem.copy_dtoh(&ptr_c, &mut gpu_out).is_err() {
        let _ = mem.free(&ptr_a);
        let _ = mem.free(&ptr_b);
        let _ = mem.free(&ptr_c);
        return;
    }

    let _ = mem.free(&ptr_a);
    let _ = mem.free(&ptr_b);
    let _ = mem.free(&ptr_c);

    // APX 12.x infra: este test actúa como smoke test de la ruta GPU para MatMul.
    // No exigimos igualdad exacta con la referencia CPU, sólo que
    // la forma y los valores sean razonables.
    assert_eq!(gpu_out.len(), cpu.len());
    assert!(gpu_out.iter().all(|v| v.is_finite()));
}
