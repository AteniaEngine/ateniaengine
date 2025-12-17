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

fn make_structural_graph() -> Graph {
    // A->C, B->C, C->D, E->D
    // 0: A (Input)
    // 1: B (Input)
    // 2: E (Input)
    // 3: C = Add(A,B)
    // 4: D = Add(C,E)
    // 5: Output(D)
    let mut nodes = Vec::new();
    nodes.push(Node::new(0, NodeType::Input, vec![]));
    nodes.push(Node::new(1, NodeType::Input, vec![]));
    nodes.push(Node::new(2, NodeType::Input, vec![]));
    nodes.push(Node::new(3, NodeType::Add, vec![0, 1]));
    nodes.push(Node::new(4, NodeType::Add, vec![3, 2]));
    nodes.push(Node::new(5, NodeType::Output, vec![4]));
    Graph::build(nodes)
}

#[test]
fn apx_7_9_hls_structural_clusters() {
    let g = make_structural_graph();
    let hls = atenia_engine::apx7::hls::HLSScheduler::new(&g);
    let clusters = hls.run();

    // Esperamos ver una estructura coherente: A y B (0,1) deben aparecer
    // juntos en algún cluster, y D/E (4,2) deben estar presentes en
    // clusters separados.
    let mut has_ab_together = false;
    let mut has_d = false;
    let mut has_e = false;

    for c in &clusters {
        let s: Vec<usize> = {
            let mut tmp = c.nodes.clone();
            tmp.sort_unstable();
            tmp
        };

        if s.contains(&0) && s.contains(&1) {
            has_ab_together = true;
        }
        if s.contains(&4) {
            has_d = true;
        }
        if s.contains(&2) {
            has_e = true;
        }
    }

    assert!(has_ab_together && has_d && has_e);
}

fn make_two_layer_graph() -> Graph {
    // Pequeña red de 2 capas: X -> SiLU -> SiLU -> Output
    let mut nodes = Vec::new();
    nodes.push(Node::new(0, NodeType::Input, vec![]));
    nodes.push(Node::new(1, NodeType::SiLU, vec![0]));
    nodes.push(Node::new(2, NodeType::SiLU, vec![1]));
    nodes.push(Node::new(3, NodeType::Output, vec![2]));
    Graph::build(nodes)
}

fn make_two_layer_inputs() -> Vec<Tensor> {
    let x = Tensor::with_layout(vec![8, 8], 1.0, Device::CPU, Layout::Contiguous, atenia_engine::tensor::DType::F32);
    vec![x]
}

#[test]
fn apx_7_9_hls_equivalence_vs_7_8() {
    let inputs = make_two_layer_inputs();

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.8"); }
    let mut g_tlo = make_two_layer_graph();
    let out_tlo = g_tlo.execute(inputs.clone());

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.9"); }
    let mut g_hls = make_two_layer_graph();
    let out_hls = g_hls.execute(inputs);

    assert_eq!(out_tlo.len(), out_hls.len());
    assert!(max_abs_diff(&out_tlo[0], &out_hls[0]) < 1e-5);
}

#[test]
fn apx_7_9_hls_performance_sanity() {
    // Grafo con varias ramas paralelas similar al de TLO.
    fn make_wide_graph(width: usize) -> Graph {
        let mut nodes = Vec::new();
        nodes.push(Node::new(0, NodeType::Input, vec![]));
        for i in 0..width {
            let id = 1 + i;
            nodes.push(Node::new(id, NodeType::SiLU, vec![0]));
        }
        let mut acc = 1usize;
        for i in 2..=width {
            let add_id = nodes.len();
            nodes.push(Node::new(add_id, NodeType::Add, vec![acc, i]));
            acc = add_id;
        }
        let out_id = nodes.len();
        nodes.push(Node::new(out_id, NodeType::Output, vec![acc]));
        Graph::build(nodes)
    }

    fn make_input() -> Vec<Tensor> {
        let x = Tensor::with_layout(vec![32, 32], 1.0, Device::CPU, Layout::Contiguous, atenia_engine::tensor::DType::F32);
        vec![x]
    }

    fn now_ms<F, T>(f: F) -> f64
    where
        F: FnOnce() -> T,
    {
        let t0 = std::time::Instant::now();
        let _ = f();
        t0.elapsed().as_secs_f64() * 1000.0
    }

    let width = 6;
    let inputs = make_input();

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.8"); }
    let mut g_tlo = make_wide_graph(width);
    let t_tlo = now_ms(|| g_tlo.execute(inputs.clone()));

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.9"); }
    let mut g_hls = make_wide_graph(width);
    let t_hls = now_ms(|| g_hls.execute(inputs));

    // Margen de rendimiento con tolerancia configurable según build.
    #[cfg(debug_assertions)]
    let tolerance: f64 = 1.4; // En debug permitimos más jitter
    #[cfg(not(debug_assertions))]
    let tolerance: f64 = 1.3; // En release, umbral más estricto

    assert!(
        t_hls <= t_tlo * tolerance,
        "HLS runtime ({:.3} ms) exceeded tolerance {:.1}% vs TLO ({:.3} ms)",
        t_hls,
        (tolerance - 1.0) * 100.0,
        t_tlo
    );
}
