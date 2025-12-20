use std::time::Instant;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::{apx_set_silent_mode, init_apx};

fn build_linear_stack_graph(
    layers: usize,
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
) -> atenia_engine::amg::graph::Graph {
    let mut gb = GraphBuilder::new();

    // Input
    let x_id = gb.input();

    let mut prev_id = x_id;
    let mut in_d = in_dim;
    for l in 0..layers {
        let out_d = if l == layers - 1 { out_dim } else { hidden_dim };

        let w = Tensor::randn(&[in_d, out_d], Device::CPU);
        let b = Tensor::randn(&[out_d], Device::CPU);

        let w_id = gb.parameter(w);
        let b_id = gb.parameter(b);

        prev_id = gb.linear(prev_id, w_id, Some(b_id));
        in_d = out_d;
    }

    let _out_id = gb.output(prev_id);

    gb.build()
}

fn run_linear_bench(label: &str, layers: usize, batch: usize, in_dim: usize, hidden_dim: usize, out_dim: usize) {
    let x = Tensor::randn(&[batch, in_dim], Device::CPU);

    // Pure CPU benchmark.
    let mut g_cpu = build_linear_stack_graph(layers, in_dim, hidden_dim, out_dim);

    // Warm-up
    let _ = g_cpu.execute(vec![x.clone()]);

    let iters = 50u32;
    let start_cpu = Instant::now();
    for _ in 0..iters {
        let _ = g_cpu.execute(vec![x.clone()]);
    }
    let elapsed_cpu = start_cpu.elapsed();
    let cpu_per_it = elapsed_cpu.as_secs_f64() * 1e3 / iters as f64;

    // Benchmark with GPU hooks (depends only on CUDA availability and shapes).
    let mut g_gpu = build_linear_stack_graph(layers, in_dim, hidden_dim, out_dim);

    let _ = g_gpu.execute(vec![x.clone()]);

    let start_gpu = Instant::now();
    for _ in 0..iters {
        let _ = g_gpu.execute(vec![x.clone()]);
    }
    let elapsed_gpu = start_gpu.elapsed();
    let gpu_per_it = elapsed_gpu.as_secs_f64() * 1e3 / iters as f64;

    let speedup = cpu_per_it / gpu_per_it;

    println!(
        "[APX 4.11 BENCH] {label}: CPU={:.4} ms | GPUHook={:.4} ms | Speedup={:.2}x",
        cpu_per_it,
        gpu_per_it,
        speedup,
    );
}

fn main() {
    // Initialize APX facilities (includes APX 4.12 GPU memory pool).
    init_apx();

    // Silence the entire APX engine to avoid traces during the benchmark.
    apx_set_silent_mode(true);

    // Some typical sizes. hidden_dim is used only for intermediate layers; the last is in_dim->out_dim.
    let configs = [
        ("Linear 64→64", 20usize, 32usize, 64usize, 64usize, 64usize),
        ("Linear 128→64", 20usize, 32usize, 128usize, 64usize, 64usize),
        ("Linear 256→128", 20usize, 32usize, 256usize, 128usize, 128usize),
    ];

    for (label, layers, batch, in_dim, hidden_dim, out_dim) in configs.into_iter() {
        run_linear_bench(label, layers, batch, in_dim, hidden_dim, out_dim);
    }
}
