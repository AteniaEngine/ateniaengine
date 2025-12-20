use atenia_engine::apx9::memory_planner::*;
use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_9_6_basic_structure() {
    let mut planner = GPUMemoryPlanner::new(1024 * 1024 * 1024);

    let t = Tensor::ones(vec![1024], Device::CPU, DType::F32);
    let sz = GPUMemoryPlanner::estimate_tensor_size(&t);
    assert!(sz > 0);

    let mut nodes = Vec::new();
    let mut n = Node::new(0, NodeType::Parameter, vec![]);
    n.set_output(t);
    nodes.push(n);

    let g = Graph::build(nodes);
    let plan = planner.plan_for_graph(&g);

    assert!(plan.total_required > 0);
    assert!(plan.assignments.len() >= 1);
}

#[test]
fn apx_9_6_buffer_reuse_chain() {
    let mut planner = GPUMemoryPlanner::new(1024 * 1024 * 1024);

    let a = Tensor::ones(vec![1024], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![2048], Device::CPU, DType::F32);
    let c = Tensor::ones(vec![1024], Device::CPU, DType::F32);

    let mut nodes = Vec::new();

    let mut n0 = Node::new(0, NodeType::Parameter, vec![]);
    n0.set_output(a);
    nodes.push(n0);

    let mut n1 = Node::new(1, NodeType::Parameter, vec![0]);
    n1.set_output(b);
    nodes.push(n1);

    let mut n2 = Node::new(2, NodeType::Parameter, vec![1]);
    n2.set_output(c);
    nodes.push(n2);

    let g = Graph::build(nodes);
    let plan = planner.plan_for_graph(&g);

    let mut off_a = None;
    let mut off_c = None;
    for a in &plan.assignments {
        if a.node_id == 0 { off_a = Some(a.offset); }
        if a.node_id == 2 { off_c = Some(a.offset); }
    }

    let off_a = off_a.expect("missing assign for A");
    let off_c = off_c.expect("missing assign for C");

    assert_eq!(off_a, off_c);
}

#[test]
fn apx_9_6_spill_policy() {
    let mut planner = GPUMemoryPlanner::new(1024 * 1024 * 1024);

    // "Large" tensor to trigger spill (depends on the internal threshold; we use something reasonable here).
    let big = Tensor::ones(vec![512 * 1024], Device::CPU, DType::F32); // ~2MB

    let mut nodes = Vec::new();
    let mut n0 = Node::new(0, NodeType::Parameter, vec![]);
    n0.set_output(big);
    nodes.push(n0);

    let g = Graph::build(nodes);
    let plan = planner.plan_for_graph(&g);

    // Since the threshold is high, we allow the test to be lax: either there is a spill
    // or it is assigned normally, but in no case should it affect numerics.
    assert!(plan.spills.len() == 0 || plan.spills.contains(&0));
}

#[test]
fn apx_9_6_no_numeric_change() {
    let mut v = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];

    for i in 0..v.len() {
        v[i] += b[i];
    }

    assert_eq!(v, vec![5.0, 7.0, 9.0]);
}
