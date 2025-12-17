use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::{Tensor, Device, Layout};

fn make_deep_graph(width: usize, depth: usize) -> Graph {
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

fn make_inputs() -> Vec<Tensor> {
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
fn apx_7_11_pfls_bench_print() {
    let width = 4;
    let depth = 6;
    let inputs = make_inputs();

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.5"); }
    let mut g_hpge = make_deep_graph(width, depth);
    let t_hpge = now_ms(|| g_hpge.execute(inputs.clone()));

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.9"); }
    let mut g_hls = make_deep_graph(width, depth);
    let t_hls = now_ms(|| g_hls.execute(inputs.clone()));

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.10"); }
    let mut g_hlsd = make_deep_graph(width, depth);
    let t_hlsd = now_ms(|| g_hlsd.execute(inputs.clone()));

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.11"); }
    let mut g_pfls = make_deep_graph(width, depth);
    let t_pfls = now_ms(|| g_pfls.execute(inputs));

    eprintln!(
        "[APX 7.11 PFLS] t_hpge={:.3} ms | t_hls={:.3} ms | t_hlsd={:.3} ms | t_pfls={:.3} ms",
        t_hpge, t_hls, t_hlsd, t_pfls
    );
}
