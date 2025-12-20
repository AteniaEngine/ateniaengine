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

fn make_inputs() -> Vec<Tensor> {
    let a = Tensor::with_layout(vec![2, 2], 1.0, Device::CPU, Layout::Contiguous, atenia_engine::tensor::DType::F32);
    let b = Tensor::with_layout(vec![2, 2], 2.0, Device::CPU, Layout::Contiguous, atenia_engine::tensor::DType::F32);
    vec![a, b]
}

#[test]
fn apx_7_5_hpge_equivalence_with_sequential() {
    let inputs = make_inputs();

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
    let inputs = make_inputs();

    fn now_ms<F, T>(f: F) -> f64
    where
        F: FnOnce() -> T,
    {
        let t0 = std::time::Instant::now();
        let _ = f();
        t0.elapsed().as_secs_f64() * 1000.0
    }

    // Sequential (7.4)
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.4");
    }
    let mut g_seq = make_simple_graph();
    let t_seq = now_ms(|| g_seq.execute(inputs.clone()));

    // HPGE (7.5)
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "7.5");
    }
    let mut g_par = make_simple_graph();
    let t_par = now_ms(|| g_par.execute(inputs));

    // Sanity: HPGE should not be much worse than sequential. In debug we use
    // a wide margin.
    assert!(t_par <= t_seq * 1.5);
}
