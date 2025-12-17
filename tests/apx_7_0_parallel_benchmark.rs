use atenia_engine::tensor::{Tensor, Device};

#[test]
fn apx_7_0_bench_parallel_vs_seq() {
    let a = Tensor::randn(&[512, 512], Device::CPU);
    let b = Tensor::randn(&[512, 512], Device::CPU);

    let t0 = std::time::Instant::now();
    let _ = a.matmul(&b);
    let seq = t0.elapsed();

    let t1 = std::time::Instant::now();
    let _ = a.matmul_parallel(&b);
    let par = t1.elapsed();

    println!("[APX 7.0] seq={:?} par={:?}", seq, par);
}
