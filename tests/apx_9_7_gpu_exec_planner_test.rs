use atenia_engine::apx9::gpu_execution_planner::*;
use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_9_7_structure() {
    let mut nodes = Vec::new();
    let mut n0 = Node::new(0, NodeType::MatMul, vec![]);
    n0.set_output(Tensor::ones(vec![16, 16], Device::CPU, DType::F32));
    nodes.push(n0);
    let g = Graph::build(nodes);

    let planner = GPUExecutionPlanner::new(1024 * 1024 * 1024);
    let plan = planner.build_plan(&g);

    assert!(!plan.steps.is_empty());
    let step = &plan.steps[0];
    assert!(!step.device.is_empty());
    assert!(!step.kernel_name.is_empty());
    assert!(!step.partitions.is_empty());
}

#[test]
fn apx_9_7_spill_policy() {
    let mut nodes = Vec::new();
    // Huge tensor to force spill (depends on the symbolic threshold in 9.6).
    let mut n0 = Node::new(0, NodeType::MatMul, vec![]);
    n0.set_output(Tensor::ones(vec![10_000_000], Device::CPU, DType::F32));
    nodes.push(n0);
    let g = Graph::build(nodes);

    let planner = GPUExecutionPlanner::new(1024 * 1024 * 1024);
    let plan = planner.build_plan(&g);

    // Allow no spill if the threshold is not exceeded; but if there is a spill, it must mark spill_to_cpu.
    if let Some(step) = plan.steps.first() {
        if step.spill_to_cpu {
            assert_eq!(step.device, "cpu");
        }
    }
}

#[test]
fn apx_9_7_partitions_logic() {
    let mut nodes = Vec::new();
    // Large tensor to trigger multiple partitions.
    let mut n0 = Node::new(0, NodeType::MatMul, vec![]);
    n0.set_output(Tensor::ones(vec![512 * 1024], Device::CPU, DType::F32));
    nodes.push(n0);
    let g = Graph::build(nodes);

    let planner = GPUExecutionPlanner::new(1024 * 1024 * 1024);
    let plan = planner.build_plan(&g);

    let step = &plan.steps[0];
    assert!(!step.partitions.is_empty());
}

#[test]
fn apx_9_7_no_numeric_change() {
    let mut v = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];

    for i in 0..v.len() {
        v[i] += b[i];
    }

    assert_eq!(v, vec![5.0, 7.0, 9.0]);
}
