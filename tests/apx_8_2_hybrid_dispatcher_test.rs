use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::{Tensor, Device, DType};
use atenia_engine::{HybridDispatcher, ExecDevice};

fn small_test_graph() -> Graph {
    let mut nodes = Vec::new();
    // 0: Input
    nodes.push(Node::new(0, NodeType::Input, vec![]));
    // 1: SiLU(0)
    nodes.push(Node::new(1, NodeType::SiLU, vec![0]));
    // 2: Output(1)
    nodes.push(Node::new(2, NodeType::Output, vec![1]));
    Graph::build(nodes)
}

fn max_abs_diff(a: &[Tensor], b: &[Tensor]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max_diff = 0.0f32;
    for (ta, tb) in a.iter().zip(b.iter()) {
        assert_eq!(ta.shape, tb.shape);
        assert_eq!(ta.dtype, tb.dtype);
        assert_eq!(ta.layout, tb.layout);
        assert_eq!(ta.data.len(), tb.data.len());
        for (va, vb) in ta.data.iter().zip(tb.data.iter()) {
            let d = (va - vb).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    max_diff
}

#[test]
fn apx_8_2_dispatcher_structure() {
    let mut g = small_test_graph();

    // Construir DualGraph directamente sin depender del modo APX global.
    let dg = atenia_engine::apx8::dualgraph::DualGraphBuilder::build(&mut g);
    assert_eq!(dg.cpu_nodes.len(), dg.gpu_nodes.len());
}

#[test]
fn apx_8_2_dispatcher_equivalence() {
    let mut g1 = small_test_graph();
    let mut g2 = small_test_graph();

    unsafe { std::env::set_var("ATENIA_APX_MODE", "7.12"); }
    let input1 = Tensor::zeros(vec![4], Device::CPU, DType::F32);
    let out_seq = g1.execute(vec![input1]);

    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.2"); }
    let input2 = Tensor::zeros(vec![4], Device::CPU, DType::F32);
    let out_hd = g2.execute(vec![input2]);

    assert!(max_abs_diff(&out_seq, &out_hd) < 1e-5);
}

#[test]
fn apx_8_2_dispatcher_selects_gpu_stub() {
    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.2"); }

    let dev = HybridDispatcher::select_device("MatMul");
    assert_eq!(dev, ExecDevice::GPU);
}
