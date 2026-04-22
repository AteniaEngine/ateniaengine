use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::{Device, Layout, Tensor};

use std::sync::Mutex;

/// Serialize the two tests in this file. Both mutate the global
/// `ATENIA_APX_MODE` env var, which `crate::apx_mode()` reads on every
/// `Graph::execute` call. Without this lock, when cargo runs tests in
/// parallel (intra-file, two threads), one test can flip the mode mid-
/// execution of the other and cause spurious failures.
///
/// NOTE: this only protects against intra-file races. Other test
/// binaries that also touch `ATENIA_APX_MODE` are out of our control
/// here; the performance test below is written to be robust against
/// that residual noise (warmup + best-of-N + wide margin + large
/// enough work to dominate init overhead).
static ENV_MODE_LOCK: Mutex<()> = Mutex::new(());

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.as_cpu_slice()
        .iter()
        .zip(b.as_cpu_slice().iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |acc, d| if d > acc { d } else { acc })
}

fn make_simple_graph() -> Graph {
    // Graph:
    // 0: Input A
    // 1: Input B
    // 2: Add(A, B) -> C
    // 3: SiLU(C)    -> D
    // 4: Output(D)  -> E
    let mut nodes = Vec::new();
    nodes.push(Node::new(0, NodeType::Input, vec![]));
    nodes.push(Node::new(1, NodeType::Input, vec![]));
    nodes.push(Node::new(2, NodeType::Add, vec![0, 1]));
    nodes.push(Node::new(3, NodeType::SiLU, vec![2]));
    nodes.push(Node::new(4, NodeType::Output, vec![3]));

    Graph::build(nodes)
}

/// Small inputs for the correctness check: equivalence between modes
/// does not depend on shape, and a 2x2 tensor keeps the test fast.
fn make_small_inputs() -> Vec<Tensor> {
    let a = Tensor::with_layout(
        vec![2, 2],
        1.0,
        Device::CPU,
        Layout::Contiguous,
        atenia_engine::tensor::DType::F32,
    );
    let b = Tensor::with_layout(
        vec![2, 2],
        2.0,
        Device::CPU,
        Layout::Contiguous,
        atenia_engine::tensor::DType::F32,
    );
    vec![a, b]
}

/// Bigger inputs for the performance sanity check: a 2x2 tensor runs
/// in microseconds, where HPGE runtime-init costs dominate and give
/// wildly variable ratios. A 512x512 tensor puts each `execute` on the
/// milliseconds scale, where the real per-op work dominates and the
/// ratio between modes becomes stable enough for a sanity bound.
fn make_large_inputs() -> Vec<Tensor> {
    let a = Tensor::with_layout(
        vec![512, 512],
        1.0,
        Device::CPU,
        Layout::Contiguous,
        atenia_engine::tensor::DType::F32,
    );
    let b = Tensor::with_layout(
        vec![512, 512],
        2.0,
        Device::CPU,
        Layout::Contiguous,
        atenia_engine::tensor::DType::F32,
    );
    vec![a, b]
}

#[test]
fn apx_7_5_hpge_equivalence_with_sequential() {
    let _lock = ENV_MODE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let inputs = make_small_inputs();

    // Run in sequential mode (7.4)
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.4");
    }
    let mut g_seq = make_simple_graph();
    let out_seq = g_seq.execute(inputs.clone());

    // Run with HPGE (7.5)
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.5");
    }
    let mut g_par = make_simple_graph();
    let out_par = g_par.execute(inputs);

    assert_eq!(out_seq.len(), 1);
    assert_eq!(out_par.len(), 1);
    assert!(max_abs_diff(&out_seq[0], &out_par[0]) < 1e-5);
}

#[test]
fn apx_7_5_hpge_sanity_performance() {
    let _lock = ENV_MODE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    // Robust perf sanity: this is not a benchmark, it's a guard against
    // HPGE being catastrophically slower than sequential. The original
    // single-shot 2x2 measurement was dominated by parallel-runtime
    // init noise (microsecond scale, 10x+ variance depending on CPU
    // warmth and scheduler). We now measure with large enough inputs
    // that per-op work dominates, warm up both modes, take best-of-N,
    // and use a generous margin that only trips on real regressions.

    fn time_ms<F, T>(f: F) -> f64
    where
        F: FnOnce() -> T,
    {
        let t0 = std::time::Instant::now();
        let _ = f();
        t0.elapsed().as_secs_f64() * 1000.0
    }

    /// Warmup + best-of-N wall-clock timing of `Graph::execute` under
    /// the current `ATENIA_APX_MODE`. Fresh `Graph` per iteration so
    /// caches (plan, fusion, etc.) are exercised on each call the
    /// same way they would be in production.
    fn measure_best_of(inputs: &[Tensor], warmups: usize, samples: usize) -> f64 {
        for _ in 0..warmups {
            let mut g = make_simple_graph();
            let _ = g.execute(inputs.to_vec());
        }
        let mut best = f64::INFINITY;
        for _ in 0..samples {
            let mut g = make_simple_graph();
            let t = time_ms(|| g.execute(inputs.to_vec()));
            if t < best {
                best = t;
            }
        }
        best
    }

    let inputs = make_large_inputs();

    // Sequential (7.4)
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.4");
    }
    let t_seq = measure_best_of(&inputs, /* warmups */ 2, /* samples */ 5);

    // HPGE (7.5)
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.5");
    }
    let t_par = measure_best_of(&inputs, /* warmups */ 2, /* samples */ 5);

    // Sanity bound: HPGE on a 2-op graph is not expected to *win* over
    // sequential (the graph is too shallow to amortize parallel
    // dispatch), but it must not be catastrophically slower. 3x is a
    // wide debug-build margin that passes when the code is healthy
    // and trips only on real regressions. We also accept tiny absolute
    // differences (< 1 ms) unconditionally to absorb scheduler jitter
    // when both paths happen to be extremely fast.
    let absolute_floor_ms = 1.0_f64;
    let ratio_bound = 3.0_f64;
    let diff_ms = t_par - t_seq;
    let within_ratio = t_par <= t_seq * ratio_bound;
    let within_floor = diff_ms <= absolute_floor_ms;
    assert!(
        within_ratio || within_floor,
        "HPGE (7.5) looks catastrophically slower than sequential (7.4): \
         t_seq={:.3}ms, t_par={:.3}ms, ratio={:.2}x (bound {}x), \
         abs_diff={:.3}ms (floor {:.3}ms)",
        t_seq,
        t_par,
        t_par / t_seq.max(1e-9),
        ratio_bound,
        diff_ms,
        absolute_floor_ms,
    );
}
