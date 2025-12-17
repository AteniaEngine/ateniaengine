use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::{Tensor, Device, Layout};

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
    let x = Tensor::with_layout(vec![64, 64], 1.0, Device::CPU, Layout::Contiguous, atenia_engine::tensor::DType::F32);
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

#[test]
fn apx_7_9_hls_bench_print() {
    let width = 8;
    let inputs = make_input();

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.5"); }
    let mut g_hpge = make_wide_graph(width);
    let t_hpge = now_ms(|| g_hpge.execute(inputs.clone()));

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.7"); }
    let mut g_hpfa = make_wide_graph(width);
    let t_hpfa = now_ms(|| g_hpfa.execute(inputs.clone()));

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.8"); }
    let mut g_tlo = make_wide_graph(width);
    let t_tlo = now_ms(|| g_tlo.execute(inputs.clone()));

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.9"); }
    let mut g_hls = make_wide_graph(width);
    let t_hls = now_ms(|| g_hls.execute(inputs));

    eprintln!(
        "[APX 7.9 HLS] t_hpge={:.3} ms | t_hpfa={:.3} ms | t_tlo={:.3} ms | t_hls={:.3} ms",
        t_hpge, t_hpfa, t_tlo, t_hls
    );
}
