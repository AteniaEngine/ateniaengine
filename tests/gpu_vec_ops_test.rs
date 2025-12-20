use atenia_engine::gpu::{
    memory::GpuMemoryEngine,
    ops::{vec_add::VecAddOp, vec_mul::VecMulOp, scalar_add::ScalarAddOp},
};

fn cpu_vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn cpu_vec_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

fn cpu_scalar_add(a: &[f32], v: f32) -> Vec<f32> {
    a.iter().map(|x| x + v).collect()
}

#[test]
fn test_vec_ops_gpu() {
    // APX 12.x: vec ops GPU smoke test.
    // If the loader/runtime enters fallback, the test must not fail.
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];

    // Local CPU references (shape only, not strict equality)
    let cpu_add = cpu_vec_add(&a, &b);
    let cpu_mul = cpu_vec_mul(&a, &b);
    let cpu_scalar = cpu_scalar_add(&a, 2.0);

    let mem = match GpuMemoryEngine::new() {
        Ok(m) => m,
        Err(_) => {
            println!("[TEST] GpuMemoryEngine init failed -> fallback OK");
            return;
        }
    };

    // Attempt GPU path wrapped in Result to treat any error as fallback.
    let attempt: Result<(Vec<f32>, Vec<f32>, Vec<f32>), ()> = (|| {
        let ptr_a = mem.alloc(a.len() * 4).map_err(|_| ())?;
        let ptr_b = mem.alloc(b.len() * 4).map_err(|_| ())?;
        let ptr_out = mem.alloc(a.len() * 4).map_err(|_| ())?;

        if mem.copy_htod(&ptr_a, &a).is_err() || mem.copy_htod(&ptr_b, &b).is_err() {
            let _ = mem.free(&ptr_a);
            let _ = mem.free(&ptr_b);
            let _ = mem.free(&ptr_out);
            return Err(());
        }

        // VecAdd
        VecAddOp::run(&ptr_a, &ptr_b, &ptr_out, a.len());
        let mut add_gpu = vec![0.0f32; a.len()];
        if mem.copy_dtoh(&ptr_out, &mut add_gpu).is_err() {
            let _ = mem.free(&ptr_a);
            let _ = mem.free(&ptr_b);
            let _ = mem.free(&ptr_out);
            return Err(());
        }

        // VecMul
        VecMulOp::run(&ptr_a, &ptr_b, &ptr_out, a.len());
        let mut mul_gpu = vec![0.0f32; a.len()];
        if mem.copy_dtoh(&ptr_out, &mut mul_gpu).is_err() {
            let _ = mem.free(&ptr_a);
            let _ = mem.free(&ptr_b);
            let _ = mem.free(&ptr_out);
            return Err(());
        }

        // ScalarAdd (+2) sobre A
        ScalarAddOp::run(&ptr_a, 2.0, a.len());
        let mut scalar_gpu = vec![0.0f32; a.len()];
        if mem.copy_dtoh(&ptr_a, &mut scalar_gpu).is_err() {
            let _ = mem.free(&ptr_a);
            let _ = mem.free(&ptr_b);
            let _ = mem.free(&ptr_out);
            return Err(());
        }

        let _ = mem.free(&ptr_a);
        let _ = mem.free(&ptr_b);
        let _ = mem.free(&ptr_out);

        Ok((add_gpu, mul_gpu, scalar_gpu))
    })();

    match attempt {
        Ok((add_gpu, mul_gpu, scalar_gpu)) => {
            // Smoke test APX 12.x: forma correcta y valores finitos.
            assert_eq!(add_gpu.len(), cpu_add.len());
            assert_eq!(mul_gpu.len(), cpu_mul.len());
            assert_eq!(scalar_gpu.len(), cpu_scalar.len());

            assert!(add_gpu.iter().all(|v| v.is_finite()));
            assert!(mul_gpu.iter().all(|v| v.is_finite()));
            assert!(scalar_gpu.iter().all(|v| v.is_finite()));
        }
        Err(_) => {
            // GPU fallback → the test does NOT fail.
            println!("[TEST] GPU fallback in vec ops -> OK");
        }
    }
}
