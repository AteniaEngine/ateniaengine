use atenia_engine::amg::graph::Graph;
use atenia_engine::amg::nodes::{Node, NodeType};
use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::init_apx;

#[test]
fn test_apx4_11_matmul_hook_runs_and_matches_cpu() {
    // Ensure APX facilities (including GPU memory pool) are initialized.
    init_apx();

    let m = 64;
    let k = 64;
    let n = 64;

    let a = Tensor::randn(&[m, k], Device::CPU);
    let b = Tensor::randn(&[k, n], Device::CPU);

    // Minimal graph: Input A, Input B, MatMul, Output.
    let mut nodes = Vec::new();
    let a_id = 0usize;
    nodes.push(Node::new(a_id, NodeType::Input, vec![]));
    let b_id = 1usize;
    nodes.push(Node::new(b_id, NodeType::Input, vec![]));
    let mm_id = 2usize;
    nodes.push(Node::new(mm_id, NodeType::MatMul, vec![a_id, b_id]));
    let out_id = 3usize;
    nodes.push(Node::new(out_id, NodeType::Output, vec![mm_id]));

    let mut g = Graph::new(nodes);

    // Execute once with GPU hooks active (APX_TRACE=1 is already set).
    let outputs_gpu = g.execute(vec![a.clone(), b.clone()]);
    assert_eq!(outputs_gpu.len(), 1);
    let y_gpu = &outputs_gpu[0];

    // Pure CPU reference: naive matmul.
    let mut y_cpu = Tensor::zeros_new(&[m, n], Device::CPU);
    {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a.data[i * k + p] * b.data[p * n + j];
                }
                y_cpu.data[i * n + j] = acc;
            }
        }
    }

    assert_eq!(y_gpu.shape, y_cpu.shape);
    let mut max_diff = 0.0f32;
    for (yg, yc) in y_gpu.data.iter().zip(y_cpu.data.iter()) {
        let d = (yg - yc).abs();
        if d > max_diff {
            max_diff = d;
        }
    }

    println!("[APX 4.11 TEST] MatMul hook max_diff = {}", max_diff);
    assert!(max_diff < 1e-3);
}

#[test]
fn test_apx4_11_linear_hook_runs_and_matches_cpu() {
    // Ensure APX facilities (including GPU memory pool) are initialized.
    init_apx();

    let m = 32;
    let k = 128;
    let n = 64;

    let x = Tensor::randn(&[m, k], Device::CPU);
    let w = Tensor::randn(&[k, n], Device::CPU);
    let b = Tensor::randn(&[n], Device::CPU);

    // Graph: x, w, b, Linear, Output.
    let mut nodes = Vec::new();
    let x_id = 0usize;
    nodes.push(Node::new(x_id, NodeType::Input, vec![]));
    let w_id = 1usize;
    nodes.push(Node::new(w_id, NodeType::Input, vec![]));
    let b_id = 2usize;
    nodes.push(Node::new(b_id, NodeType::Input, vec![]));
    let lin_id = 3usize;
    nodes.push(Node::new(lin_id, NodeType::Linear, vec![x_id, w_id, b_id]));
    let out_id = 4usize;
    nodes.push(Node::new(out_id, NodeType::Output, vec![lin_id]));

    let mut g = Graph::new(nodes);

    let outputs_gpu = g.execute(vec![x.clone(), w.clone(), b.clone()]);
    assert_eq!(outputs_gpu.len(), 1);
    let y_gpu = &outputs_gpu[0];

    // Simple CPU reference: y = x·w + b
    let mut y_cpu = Tensor::zeros_new(&[m, n], Device::CPU);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += x.data[i * k + p] * w.data[p * n + j];
            }
            acc += b.data[j];
            y_cpu.data[i * n + j] = acc;
        }
    }

    assert_eq!(y_gpu.shape, y_cpu.shape);
    let mut max_diff = 0.0f32;
    for (yg, yc) in y_gpu.data.iter().zip(y_cpu.data.iter()) {
        let d = (yg - yc).abs();
        if d > max_diff {
            max_diff = d;
        }
    }

    println!("[APX 4.11 TEST] Linear hook max_diff = {}", max_diff);
    assert!(max_diff < 1e-3);
}
