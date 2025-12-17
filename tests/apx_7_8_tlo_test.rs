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

fn make_two_branch_graph() -> Graph {
    // 0: Input X
    // 1: A = SiLU(X)
    // 2: B = SiLU(X)
    // 3: C = Add(A, B)
    // 4: Output(C)
    let mut nodes = Vec::new();
    nodes.push(Node::new(0, NodeType::Input, vec![]));
    nodes.push(Node::new(1, NodeType::SiLU, vec![0]));
    nodes.push(Node::new(2, NodeType::SiLU, vec![0]));
    nodes.push(Node::new(3, NodeType::Add, vec![1, 2]));
    nodes.push(Node::new(4, NodeType::Output, vec![3]));
    Graph::build(nodes)
}

fn make_inputs() -> Vec<Tensor> {
    let x = Tensor::with_layout(vec![16, 16], 1.0, Device::CPU, Layout::Contiguous, atenia_engine::tensor::DType::F32);
    vec![x]
}

#[test]
fn apx_7_8_tlo_equivalence_with_sequential() {
    let inputs = make_inputs();

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.4"); }
    let mut g_seq = make_two_branch_graph();
    let out_seq = g_seq.execute(inputs.clone());

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.8"); }
    let mut g_tlo = make_two_branch_graph();
    let out_tlo = g_tlo.execute(inputs);

    assert_eq!(out_seq.len(), out_tlo.len());
    assert!(max_abs_diff(&out_seq[0], &out_tlo[0]) < 1e-6);
}

#[test]
fn apx_7_8_tlo_reorders_ready_nodes_structural() {
    // No observamos directamente ready_queue, pero aseguramos que la
    // construcción de hints no paniquea y que TLO se puede invocar sin
    // romper nada en un grafo sencillo con dos ramas.
    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.8"); }
    let mut g = make_two_branch_graph();
    let _ = g.execute(make_inputs());
}

#[test]
fn apx_7_8_tlo_performance_sanity() {
    let inputs = make_inputs();

    fn now_ms<F, T>(f: F) -> f64
    where
        F: FnOnce() -> T,
    {
        let t0 = std::time::Instant::now();
        let _ = f();
        t0.elapsed().as_secs_f64() * 1000.0
    }

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.5"); }
    let mut g_hpge = make_two_branch_graph();
    let t_hpge = now_ms(|| g_hpge.execute(inputs.clone()));

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.8"); }
    let mut g_tlo = make_two_branch_graph();
    let t_tlo = now_ms(|| g_tlo.execute(inputs));

    // En debug dejamos margen amplio, sólo verificamos que no sea muy peor.
    assert!(t_tlo <= t_hpge * 1.3);
}
