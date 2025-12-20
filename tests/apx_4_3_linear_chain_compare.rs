use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::linear::linear;

#[test]
fn apx_4_3_linear_chain_gpu_matches_cpu() {
    unsafe {
        std::env::set_var("APX_TRACE", "1");
        std::env::set_var("APX_GPU", "1");
    }

    let mut gb = GraphBuilder::new();
    let x = gb.input();
    let mut last = x;

    // Build a small chain of 3 compatible linear layers.
    let mut layers: Vec<(Tensor, Tensor)> = Vec::new();
    let m = 1usize;
    let k = 32usize;
    let n = 32usize;

    for _ in 0..3 {
        let w = Tensor::randn(&[k, n], Device::CPU);
        let b = Tensor::randn(&[n], Device::CPU); // bias 1D [n]
        let w_id = gb.parameter(w.clone());
        let b_id = gb.parameter(b.clone());
        last = gb.linear(last, w_id, Some(b_id));
        layers.push((w, b));
    }

    gb.output(last);
    let mut graph = gb.build();

    // Common CPU/GPU input
    let x0 = Tensor::randn(&[m, k], Device::CPU);

    // CPU: apply the same chain of linear layers with the same weights and bias.
    let mut cpu = x0.clone();
    for (w, b) in &layers {
        cpu = linear(&cpu, w, Some(b));
    }

    // GPU: execute the graph (APX 4.3 should route linear layers to CUDA when possible).
    let out_vec = graph.execute(vec![x0.clone()]);
    assert_eq!(out_vec.len(), 1, "graph must produce a single output tensor");
    let gpu = &out_vec[0];

    assert_eq!(cpu.shape, gpu.shape, "CPU/GPU shapes must match");

    for i in 0..cpu.data.len() {
        let d = (cpu.data[i] - gpu.data[i]).abs();
        assert!(d < 1e-3, "diff too large at {}: {}", i, d);
    }
}
