use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::{Device};

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

#[test]
fn apx_8_1_dualgraph_basic_structure() {
    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.1"); }
    let mut g = small_test_graph();

    // Force an explicit build_plan for APX 8.1.
    g.build_plan();

    let dg = g.dual_graph.as_ref().expect("dual_graph must be built in 8.1 mode");

    assert_eq!(dg.cpu_nodes.len(), g.nodes.len());
    assert_eq!(dg.gpu_nodes.len(), g.nodes.len());

    for i in 0..g.nodes.len() {
        assert_eq!(dg.mapping_cpu_to_gpu[i], i);
        assert_eq!(dg.mapping_gpu_to_cpu[i], i);

        assert_eq!(dg.cpu_nodes[i].inputs, dg.gpu_nodes[i].inputs);

        let cpu_out = &dg.cpu_nodes[i].output;
        let gpu_out = &dg.gpu_nodes[i].output;
        match (cpu_out, gpu_out) {
            (Some(c), Some(gpu)) => {
                assert_eq!(c.shape, gpu.shape);
                assert_eq!(c.dtype, gpu.dtype);
                assert_eq!(c.layout, gpu.layout);
                // GPU mirror must be marked as Device::GPU.
                assert_eq!(gpu.device, Device::GPU);
            }
            (None, None) => {}
            _ => panic!("CPU/GPU outputs mismatch at node {i}"),
        }
    }
}
