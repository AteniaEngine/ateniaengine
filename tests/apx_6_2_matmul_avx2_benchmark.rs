use atenia_engine::matmul_dispatcher;
use atenia_engine::apx6_2::avx2_matmul;

#[test]
fn bench_avx2_vs_baseline() {
    use std::time::Instant;
    let sizes = [128, 256, 512, 1024];

    for &s in &sizes {
        let (m, k, n) = (s, s, s);

        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut out = vec![0.0f32; m * n];

        // baseline
        let t0 = Instant::now();
        matmul_dispatcher::matmul_dispatch(&a, &b, &mut out, m, k, n);
        let baseline = t0.elapsed().as_micros();

        // avx2
        let t1 = Instant::now();
        unsafe {
            avx2_matmul::matmul_avx2_f32(
                a.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
                m,
                k,
                n,
            );
        }
        let avx2 = t1.elapsed().as_micros();

        println!(
            "[APX 6.2 BENCH] size {}x{}x{} -> baseline={}us | avx2={}us | speedup={:.2}x",
            m,
            k,
            n,
            baseline,
            avx2,
            baseline as f32 / avx2 as f32,
        );
    }
}
