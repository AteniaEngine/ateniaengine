//! APX v20 M3-e.1 — unit tests for `Graph::migrate_all_cuda_to_cpu`,
//! the primitive that implements the Degrade reaction strategy by
//! moving every Cuda-resident `node.output` back to CPU.
//!
//! The guard wiring itself (`check_guard_before_node` reacting to
//! `GuardAction::Degrade`) is exercised end-to-end by integration
//! tests landing in M3-e.3; this file covers the migration primitive
//! in isolation: empty graph, all-CPU graph, mixed storage, all-Cuda.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::gpu::gpu_engine;
use atenia_engine::tensor::{DType, Device, Layout, Tensor, TensorStorage};

fn require_gpu(test_name: &str) -> bool {
    if gpu_engine().is_some() {
        true
    } else {
        println!(
            "[TEST:{}] no GPU available (gpu_engine() = None) -> graceful skip",
            test_name
        );
        false
    }
}

fn tensor_from(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    let mut t = Tensor::with_layout(
        shape,
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    t.as_cpu_slice_mut().copy_from_slice(&data);
    t
}

#[test]
fn test_migrate_empty_graph() {
    // A freshly built graph with no nodes should produce a zero-count
    // report regardless of GPU availability — the method walks
    // `self.nodes`, which is empty.
    let gb = GraphBuilder::new();
    let mut graph = gb.build();

    let report = graph
        .migrate_all_cuda_to_cpu()
        .expect("migrate must not fail on empty graph");

    assert_eq!(report.tensors_migrated, 0);
    assert_eq!(report.bytes_freed_estimate, 0);
}

#[test]
fn test_migrate_no_cuda_tensors() {
    // Graph built, inputs set, all outputs remain on CPU. Migration
    // should observe no Cuda storage and report zero migrations.
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w_id = gb.input();
    let lin_id = gb.linear(x_id, w_id, None);
    let _out_id = gb.output(lin_id);

    let mut graph = gb.build();

    let x = tensor_from(vec![1, 2], vec![1.0, 2.0]);
    let w = tensor_from(vec![2, 1], vec![3.0, 4.0]);
    let _ = graph.execute(vec![x, w]);

    // Sanity: every materialized output is Cpu.
    for node in &graph.nodes {
        if let Some(ref out) = node.output {
            assert!(
                matches!(out.storage, TensorStorage::Cpu(_)),
                "pre-condition: outputs should be Cpu before migration"
            );
        }
    }

    let report = graph
        .migrate_all_cuda_to_cpu()
        .expect("migrate on all-Cpu graph must succeed");
    assert_eq!(report.tensors_migrated, 0);
    assert_eq!(report.bytes_freed_estimate, 0);

    // Post-condition: storage is still Cpu (no-op path preserved data).
    for node in &graph.nodes {
        if let Some(ref out) = node.output {
            assert!(matches!(out.storage, TensorStorage::Cpu(_)));
        }
    }
}

#[test]
fn test_migrate_mixed_storage() {
    if !require_gpu("test_migrate_mixed_storage") {
        return;
    }

    // Build a small graph, execute forward so every node has an output,
    // then migrate a strict subset of outputs to GPU. The migration
    // primitive must move exactly that subset back to CPU and preserve
    // every value.
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w_id = gb.input();
    let lin_id = gb.linear(x_id, w_id, None);
    let _out_id = gb.output(lin_id);

    let mut graph = gb.build();

    let x = tensor_from(vec![1, 2], vec![1.0, 2.0]);
    let w = tensor_from(vec![2, 1], vec![3.0, 4.0]);
    let _ = graph.execute(vec![x, w]);

    // Snapshot expected data per node (before migration).
    let expected: Vec<Option<Vec<f32>>> = graph
        .nodes
        .iter()
        .map(|n| n.output.as_ref().map(|t| t.as_cpu_slice().to_vec()))
        .collect();

    // Migrate only node 0 and node 2 to GPU; leave the others on CPU.
    let indices_to_migrate = [0usize, 2usize];
    for &i in &indices_to_migrate {
        if let Some(ref mut out) = graph.nodes[i].output {
            out.ensure_gpu().expect("ensure_gpu must succeed");
            assert!(
                matches!(out.storage, TensorStorage::Cuda(_)),
                "post-ensure_gpu: node {} should be Cuda",
                i
            );
        }
    }

    let report = graph
        .migrate_all_cuda_to_cpu()
        .expect("migrate_all_cuda_to_cpu must succeed");
    assert_eq!(
        report.tensors_migrated, 2,
        "exactly the two nodes migrated above should be reported"
    );
    assert!(
        report.bytes_freed_estimate > 0,
        "bytes_freed_estimate should be > 0 when at least one tensor moved"
    );

    // Post-condition: every output is Cpu again and values match the
    // snapshot taken before any migration.
    for (i, node) in graph.nodes.iter().enumerate() {
        match (&node.output, &expected[i]) {
            (Some(out), Some(want)) => {
                assert!(
                    matches!(out.storage, TensorStorage::Cpu(_)),
                    "node {} should be Cpu after migrate_all_cuda_to_cpu",
                    i
                );
                assert_eq!(
                    out.as_cpu_slice(),
                    want.as_slice(),
                    "node {} data must be preserved bit-for-bit through Cuda roundtrip",
                    i
                );
            }
            (None, None) => {}
            (a, b) => panic!(
                "node {} output presence mismatch: before={:?}, after_some={:?}",
                i,
                b.is_some(),
                a.is_some()
            ),
        }
    }
}

#[test]
fn test_migrate_all_cuda() {
    if !require_gpu("test_migrate_all_cuda") {
        return;
    }

    // Same shape as the mixed test but migrates every materialized
    // output. The migration call must report exactly that many and
    // leave the graph in a fully CPU-resident state.
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();
    let w_id = gb.input();
    let lin_id = gb.linear(x_id, w_id, None);
    let _out_id = gb.output(lin_id);

    let mut graph = gb.build();

    let x = tensor_from(vec![1, 2], vec![1.0, 2.0]);
    let w = tensor_from(vec![2, 1], vec![3.0, 4.0]);
    let _ = graph.execute(vec![x, w]);

    let expected: Vec<Option<Vec<f32>>> = graph
        .nodes
        .iter()
        .map(|n| n.output.as_ref().map(|t| t.as_cpu_slice().to_vec()))
        .collect();

    // Push every materialized output to GPU.
    let mut ensured = 0usize;
    for node in graph.nodes.iter_mut() {
        if let Some(ref mut out) = node.output {
            out.ensure_gpu().expect("ensure_gpu must succeed");
            assert!(matches!(out.storage, TensorStorage::Cuda(_)));
            ensured += 1;
        }
    }
    assert!(
        ensured > 0,
        "test setup: expected at least one materialized output to migrate"
    );

    let report = graph
        .migrate_all_cuda_to_cpu()
        .expect("migrate_all_cuda_to_cpu must succeed");
    assert_eq!(report.tensors_migrated, ensured);
    assert!(report.bytes_freed_estimate > 0);

    for (i, node) in graph.nodes.iter().enumerate() {
        if let (Some(out), Some(want)) = (&node.output, &expected[i]) {
            assert!(matches!(out.storage, TensorStorage::Cpu(_)));
            assert_eq!(out.as_cpu_slice(), want.as_slice());
        }
    }
}
