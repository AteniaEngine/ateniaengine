//! APX v20 M3-e.1 — unit tests for `Graph::migrate_all_cuda_to_cpu`,
//! the primitive that implements the Degrade reaction strategy by
//! moving every Cuda-resident `node.output` back to CPU.
//!
//! The guard wiring itself (`check_guard_before_node` reacting to
//! `GuardAction::Degrade`) is exercised end-to-end by integration
//! tests landing in M3-e.3; this file covers the migration primitive
//! in isolation: empty graph, all-CPU graph, mixed storage, all-Cuda.

use std::sync::Arc;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::reactive::ReactiveExecutionContext;
use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::gpu::gpu_engine;
use atenia_engine::tensor::{DType, Device, Layout, Tensor, TensorStorage};
use atenia_engine::v15::policy::types::DecisionBias;
use atenia_engine::v16::contract::constraints::{Constraints, RuntimeState};
use atenia_engine::v16::contract::execution_contract::{
    ExecutionBackend, ExecutionContract,
};
use atenia_engine::v16::guards::execution_guard::ExecutionGuard;
use atenia_engine::v16::guards::guard_action::GuardAction;
use atenia_engine::v16::guards::guard_conditions::GuardConditions;
use atenia_engine::v16::guards::guard_manager::GuardManager;

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

// =====================================================================
// Integration tests (M3-e.3): end-to-end wiring through
// ReactiveExecutionContext → GuardManager → GuardAction::Degrade →
// Graph::migrate_all_cuda_to_cpu → execution continues on CPU.
// =====================================================================

/// Test fixture: emits `Degrade` when the SignalBus reports any
/// failures recorded during the test. The signal used (`recent_failures`)
/// is one of the two that can be driven deterministically from the test
/// without depending on real hardware probes — the same approach taken
/// by `amg_signalbus_abort_test.rs` for the `Abort` path. See the
/// handoff note on `SignalBus` (no injection API for `memory_pressure`).
struct DegradeIfFailuresGuard;

impl ExecutionGuard for DegradeIfFailuresGuard {
    fn name(&self) -> &'static str {
        "degrade_if_failures_guard_m3_e_3_fixture"
    }
    fn evaluate(
        &self,
        _contract: &ExecutionContract,
        conditions: &GuardConditions,
    ) -> GuardAction {
        if conditions.recent_failures > 0 {
            GuardAction::Degrade
        } else {
            GuardAction::Continue
        }
    }
}

/// Permissive contract: `require_stability = false` so the
/// `GuardManager` legality check never vetos `Continue` under
/// pre-OOM conditions (the test does not touch pre_oom).
fn permissive_contract() -> ExecutionContract {
    ExecutionContract {
        bias: DecisionBias {
            risk_weight: 0.3,
            latency_weight: 0.4,
            stability_weight: 0.5,
            memory_pressure_weight: 0.5,
            offload_cost_weight: 0.4,
        },
        runtime_snapshot: RuntimeState {
            memory_headroom: 0.8,
            is_stable: true,
            recent_recovery: false,
            offload_supported: true,
        },
        allowed_backends: vec![ExecutionBackend::Local],
        forbidden_backends: vec![],
        max_aggressiveness: 0.5,
        require_fallback: false,
        require_stability: false,
        constraints: Constraints { items: vec![] },
    }
}

/// Small deterministic graph: `out = linear(input, weight)` with
/// `weight` stored as a `Parameter`. We use `Parameter` (not another
/// `Input`) because parameter outputs persist across `execute_checked`
/// calls, so a Cuda migration applied to them survives into the
/// guarded execution that we want to test. Inputs are overwritten by
/// `set_input_outputs` at the start of each execute, which would
/// destroy a pre-execute migration.
fn build_linear_param_graph() -> (atenia_engine::amg::graph::Graph, usize, usize) {
    let mut gb = GraphBuilder::new();
    let x_id = gb.input();

    let mut w = Tensor::with_layout(
        vec![2, 1],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    w.set_cpu_data(vec![3.0, 4.0]);
    let w_id = gb.parameter(w);

    let lin_id = gb.linear(x_id, w_id, None);
    let _out_id = gb.output(lin_id);

    (gb.build(), x_id, w_id)
}

fn make_x_input() -> Tensor {
    tensor_from(vec![1, 2], vec![1.0, 2.0])
}

/// Expected output of `linear(x=[[1,2]], w=[[3],[4]])` = [[1*3+2*4]] = [[11]].
const EXPECTED_OUT_LINEAR: f32 = 11.0;

#[test]
fn test_degrade_triggers_migration_and_continues() {
    if !require_gpu("test_degrade_triggers_migration_and_continues") {
        return;
    }

    // 1. Build the graph and materialize outputs once with a plain
    //    `execute` (no reactive context): this gives us baseline
    //    Cpu-resident outputs we can snapshot.
    let (mut graph, _x_id, w_id) = build_linear_param_graph();
    let baseline = graph.execute(vec![make_x_input()]);
    assert_eq!(baseline.len(), 1);
    assert!((baseline[0].as_cpu_slice()[0] - EXPECTED_OUT_LINEAR).abs() < 1e-5);

    // Snapshot the Parameter's values before we push it to VRAM.
    let w_snapshot: Vec<f32> = graph.nodes[w_id]
        .output
        .as_ref()
        .expect("parameter output must exist after first execute")
        .as_cpu_slice()
        .to_vec();

    // 2. Migrate the Parameter's output to VRAM. Parameters persist
    //    across execute calls, so after this step the next
    //    `execute_checked` enters the graph with `w` on Cuda storage.
    {
        let w_tensor = graph.nodes[w_id]
            .output
            .as_mut()
            .expect("parameter output present");
        w_tensor.ensure_gpu().expect("ensure_gpu on parameter must succeed");
        assert!(
            matches!(w_tensor.storage, TensorStorage::Cuda(_)),
            "setup: parameter should be Cuda after ensure_gpu"
        );
    }

    // 3. Attach a ReactiveExecutionContext that emits `Degrade` on
    //    `recent_failures > 0`, and record one failure on the SignalBus
    //    so the guard fires on the very first `check_guard_before_node`
    //    call inside the next `execute_checked`.
    let bus = Arc::new(SignalBus::new());
    bus.failure_counter().record_failure();

    let guards: Vec<Box<dyn ExecutionGuard>> = vec![Box::new(DegradeIfFailuresGuard)];
    let guard_manager = GuardManager::new(guards);
    let ctx = ReactiveExecutionContext::new(bus, permissive_contract(), guard_manager);
    graph.set_reactive_context(ctx);

    // 4. Execute under the reactive context. The guard must emit
    //    Degrade, which runs `migrate_all_cuda_to_cpu`, moving the
    //    Parameter (and any other Cuda output) back to Cpu before the
    //    Linear node runs. If migration did not happen, the Linear's
    //    host-path dispatch would panic on `as_cpu_slice` of the Cuda
    //    parameter.
    let result = graph.execute_checked(vec![make_x_input()]);
    assert!(
        result.is_ok(),
        "execute_checked under Degrade must succeed, got {:?}",
        result.err()
    );
    assert!(
        graph.last_abort().is_none(),
        "no abort expected when guard emits Degrade, got {:?}",
        graph.last_abort()
    );

    // 5. Post-conditions: every output is back on Cpu (migration
    //    happened AND the forward ran) and values are preserved
    //    bit-for-bit against the baseline (roundtrip through Cuda
    //    must be exact for f32).
    let w_after = graph.nodes[w_id]
        .output
        .as_ref()
        .expect("parameter output still present");
    assert!(
        matches!(w_after.storage, TensorStorage::Cpu(_)),
        "parameter must be Cpu after Degrade-driven execute_checked"
    );
    assert_eq!(
        w_after.as_cpu_slice(),
        w_snapshot.as_slice(),
        "parameter values must survive the Cuda→Cpu migration bit-exactly"
    );

    let outputs = result.expect("checked above that it is Ok");
    assert_eq!(outputs.len(), 1);
    assert!(
        (outputs[0].as_cpu_slice()[0] - EXPECTED_OUT_LINEAR).abs() < 1e-5,
        "final output value should match the reference computation"
    );
}

#[test]
fn test_no_degrade_without_failure_signal_preserves_cuda_parameter() {
    if !require_gpu("test_no_degrade_without_failure_signal_preserves_cuda_parameter") {
        return;
    }

    // Negative counterpart to the positive test above: the same setup
    // (parameter on VRAM, reactive context attached) but *without*
    // `record_failure()`. With no failures recorded, the guard emits
    // `Continue`, the graph never calls `migrate_all_cuda_to_cpu`, and
    // the parameter must still be on Cuda after execute_checked
    // completes.
    //
    // Numerical correctness is unaffected: `nn::linear::linear` uses
    // `weight.copy_to_cpu_vec()` to read the parameter, which handles
    // Cuda storage transparently via D→H copy (M3-d.2 contract). This
    // test therefore also confirms that the CPU Linear path tolerates
    // a Cuda parameter when the reaction strategy chooses not to
    // migrate — it does not *require* migration for correctness, only
    // for VRAM pressure relief.
    let (mut graph, _x_id, w_id) = build_linear_param_graph();
    let _baseline = graph.execute(vec![make_x_input()]);

    {
        let w_tensor = graph.nodes[w_id]
            .output
            .as_mut()
            .expect("parameter output present");
        w_tensor.ensure_gpu().expect("ensure_gpu on parameter must succeed");
        assert!(
            matches!(w_tensor.storage, TensorStorage::Cuda(_)),
            "setup: parameter should be Cuda after ensure_gpu"
        );
    }

    let bus = Arc::new(SignalBus::new());
    // Deliberately no `record_failure()`.
    let guards: Vec<Box<dyn ExecutionGuard>> = vec![Box::new(DegradeIfFailuresGuard)];
    let guard_manager = GuardManager::new(guards);
    let ctx = ReactiveExecutionContext::new(bus, permissive_contract(), guard_manager);
    graph.set_reactive_context(ctx);

    let result = graph.execute_checked(vec![make_x_input()]);
    assert!(
        result.is_ok(),
        "execute_checked with Continue guard should succeed, got {:?}",
        result.err()
    );
    assert!(
        graph.last_abort().is_none(),
        "no abort expected when guard emits Continue"
    );

    // Output is numerically correct even without migration, because
    // `nn::linear` reads the weight via `copy_to_cpu_vec` (D→H on the
    // fly).
    let outputs = result.expect("checked above");
    assert_eq!(outputs.len(), 1);
    assert!(
        (outputs[0].as_cpu_slice()[0] - EXPECTED_OUT_LINEAR).abs() < 1e-5,
        "numerical output must be correct even when the parameter stays on Cuda"
    );

    // Key negative assertion: the parameter was NOT migrated. The
    // reaction strategy did nothing, as expected.
    let w_after = graph.nodes[w_id]
        .output
        .as_ref()
        .expect("parameter output still present");
    assert!(
        matches!(w_after.storage, TensorStorage::Cuda(_)),
        "parameter must remain Cuda when guard emits Continue (no migration)"
    );
}

