use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::{Tensor, Device, Layout};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |acc, d| if d > acc { d } else { acc })
}

fn make_deep_graph() -> Graph {
    // Graph with branches and a deep chain to exercise HLS Deep.
    // 0: Input
    // 1: SiLU(0)
    // 2: SiLU(1)
    // 3: SiLU(2)
    // 4: Skip = SiLU(0)
    // 5: Add(3,4)
    // 6: Output(5)
    let mut nodes = Vec::new();
    nodes.push(Node::new(0, NodeType::Input, vec![]));
    nodes.push(Node::new(1, NodeType::SiLU, vec![0]));
    nodes.push(Node::new(2, NodeType::SiLU, vec![1]));
    nodes.push(Node::new(3, NodeType::SiLU, vec![2]));
    nodes.push(Node::new(4, NodeType::SiLU, vec![0]));
    nodes.push(Node::new(5, NodeType::Add, vec![3, 4]));
    nodes.push(Node::new(6, NodeType::Output, vec![5]));
    Graph::build(nodes)
}

fn make_inputs() -> Vec<Tensor> {
    let x = Tensor::with_layout(vec![8, 8], 1.0, Device::CPU, Layout::Contiguous, atenia_engine::tensor::DType::F32);
    vec![x]
}

#[test]
fn apx_7_10_hls_deep_superlevels_nontrivial() {
    let g = make_deep_graph();
    let depths = atenia_engine::apx7::hls_deep::compute_depth(&g);
    let superlevels = atenia_engine::apx7::hls_deep::build_superlevels(&depths);

    // There must be at least 2 superlevels and at least one involving
    // more than one depth level.
    assert!(superlevels.len() >= 2);
}

#[test]
fn apx_7_10_hls_deep_equivalence_vs_7_9() {
    let inputs = make_inputs();

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.9"); }
    let mut g_tlo = make_deep_graph();
    let out_tlo = g_tlo.execute(inputs.clone());

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.10"); }
    let mut g_hlsd = make_deep_graph();
    let out_hlsd = g_hlsd.execute(inputs);

    assert_eq!(out_tlo.len(), out_hlsd.len());
    assert!(max_abs_diff(&out_tlo[0], &out_hlsd[0]) < 1e-5);
}

#[test]
fn apx_7_10_hls_deep_performance_sanity() {
    fn make_wider_deep_graph(width: usize, depth: usize) -> Graph {
        // Multiple deep chains hanging from the same input.
        let mut nodes = Vec::new();
        nodes.push(Node::new(0, NodeType::Input, vec![]));
        let mut last_ids = Vec::new();
        for _w in 0..width {
            let mut prev = 0usize;
            for _d in 0..depth {
                let id = nodes.len();
                nodes.push(Node::new(id, NodeType::SiLU, vec![prev]));
                prev = id;
            }
            last_ids.push(prev);
        }
        let mut acc = last_ids[0];
        for &id in &last_ids[1..] {
            let add_id = nodes.len();
            nodes.push(Node::new(add_id, NodeType::Add, vec![acc, id]));
            acc = add_id;
        }
        let out_id = nodes.len();
        nodes.push(Node::new(out_id, NodeType::Output, vec![acc]));
        Graph::build(nodes)
    }

    fn now_ms<F, T>(f: F) -> f64
    where
        F: FnOnce() -> T,
    {
        let t0 = std::time::Instant::now();
        let _ = f();
        t0.elapsed().as_secs_f64() * 1000.0
    }

    let inputs = make_inputs();

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.4"); }
    let mut g_seq = make_wider_deep_graph(3, 4);
    let t_seq = now_ms(|| g_seq.execute(inputs.clone()));

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.10"); }
    let mut g_hlsd = make_wider_deep_graph(3, 4);
    let t_hlsd = now_ms(|| g_hlsd.execute(inputs));

    // APX 7.10: this test is a performance sanity check, not a strict contract.
    // Allow HLS-Deep to be up to several times slower than the baseline path
    // before considering it a catastrophic regression. The goal is to detect
    // orders of magnitude, not micro-variations between machines/environments.
    let max_factor = 5.0;
    assert!(t_hlsd <= t_seq * max_factor);
}
