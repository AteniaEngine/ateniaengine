use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::{Tensor, Device};

#[test]
fn apx_4_6_full_pipeline_runs_gpu_segments_correctly() {
    unsafe {
        std::env::set_var("APX_TRACE", "1");
        std::env::set_var("APX_CUDA_FORCE", "1");
    }

    let mut gb = GraphBuilder::new();

    let x = gb.input();           // batch 4, shape vendrá del input runtime
    let w1 = gb.parameter(Tensor::randn(&[32, 32], Device::CPU));
    let b1 = gb.parameter(Tensor::randn(&[32], Device::CPU));
    let h1 = gb.linear(x, w1, Some(b1));

    let w2 = gb.parameter(Tensor::randn(&[32, 32], Device::CPU));
    let b2 = gb.parameter(Tensor::randn(&[32], Device::CPU));
    let h2 = gb.linear(h1, w2, Some(b2));

    let w3 = gb.parameter(Tensor::randn(&[32, 32], Device::CPU));
    let b3 = gb.parameter(Tensor::randn(&[32], Device::CPU));
    let out = gb.linear(h2, w3, Some(b3));

    gb.output(out);

    let mut graph = gb.build();
    let mut cpu_graph = graph.clone();

    let input = Tensor::randn(&[4, 32], Device::CPU);

    let gpu_out = graph.execute(vec![input.clone()]);
    let cpu_out = cpu_graph.execute(vec![input]);

    let g = &gpu_out[0].data;
    let c = &cpu_out[0].data;

    let mut max_diff: f32 = 0.0;

    for i in 0..g.len() {
        max_diff = max_diff.max((g[i] - c[i]).abs());
    }

    println!("max diff = {}", max_diff);

    assert!(max_diff < 1e-3);
}
