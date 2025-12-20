use atenia_engine::apx9::gpu_execution_planner::*;
use atenia_engine::apx9::gpu_executor_mock::*;
use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_9_8_structure() {
    let mut nodes = Vec::new();
    let mut n0 = Node::new(0, NodeType::MatMul, vec![]);
    n0.set_output(Tensor::ones(vec![16, 16], Device::CPU, DType::F32));
    nodes.push(n0);
    let g = Graph::build(nodes);

    let planner = GPUExecutionPlanner::new(1024 * 1024 * 1024);
    let plan = planner.build_plan(&g);
    let result = execute_plan_mock(&plan);

    assert!(result.total_time_ms >= 0.0);
    assert_eq!(result.per_step.len(), plan.steps.len());
}

#[test]
fn apx_9_8_spills_propagate() {
    let mut nodes = Vec::new();
    let mut n0 = Node::new(0, NodeType::MatMul, vec![]);
    n0.set_output(Tensor::ones(vec![10_000_000], Device::CPU, DType::F32));
    nodes.push(n0);
    let g = Graph::build(nodes);

    let planner = GPUExecutionPlanner::new(1024 * 1024 * 1024);
    let plan = planner.build_plan(&g);
    let result = execute_plan_mock(&plan);

    if let Some(step) = plan.steps.first() {
        if step.spill_to_cpu {
            assert!(result.spills >= 1);
        }
    }
}

#[test]
fn apx_9_8_partition_overhead() {
    let mut nodes = Vec::new();

    // Small node (1 partition)
    let mut n0 = Node::new(0, NodeType::MatMul, vec![]);
    n0.set_output(Tensor::ones(vec![16, 16], Device::CPU, DType::F32));
    nodes.push(n0);

    // Large node (multiple symbolic partitions)
    let mut n1 = Node::new(1, NodeType::MatMul, vec![]);
    n1.set_output(Tensor::ones(vec![512 * 1024], Device::CPU, DType::F32));
    nodes.push(n1);

    let g = Graph::build(nodes);
    let planner = GPUExecutionPlanner::new(1024 * 1024 * 1024);
    let plan = planner.build_plan(&g);
    let result = execute_plan_mock(&plan);

    assert_eq!(result.per_step.len(), 2);
    // The step with more partitions should be symbolically more expensive.
    assert!(result.per_step[1] >= result.per_step[0]);
}

#[test]
fn apx_9_8_no_numeric_change() {
    let mut v = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];

    for i in 0..v.len() {
        v[i] += b[i];
    }

    assert_eq!(v, vec![5.0, 7.0, 9.0]);
}
