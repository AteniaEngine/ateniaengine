//! M4.7.5.a — probe-cache amortisation audit.
//!
//! The historical TODO at `Graph::check_guard_before_node` flagged
//! `collect_guard_conditions` as ~40 ms per call (two `nvidia-smi`
//! subprocess reads + sysinfo refresh per call). On a 1850-node
//! Llama 2 13B forward that would be 60+ seconds of pure probing
//! overhead — bigger than the entire compute budget on the dev
//! hardware.
//!
//! The `SignalBus::probe_cache` (introduced in M3-e) caches the
//! full `ProbeValues` bundle for `SIGNAL_BUS_CACHE_TTL = 100 ms`,
//! so the actual cost is bounded by `wall_clock / TTL × probe_cost`,
//! independent of node count. This test stands as the regression
//! gate for any future cache-bypass change: if a node count
//! ramps up without changing wall time, the probe count must
//! stay bounded.
//!
//! What the test does:
//!
//!   - Build a deterministic 200-node graph (chain of `Add` nodes).
//!   - Attach a `ReactiveExecutionContext` with mock probes that
//!     return zero pressure (so no migration fires; we are isolating
//!     the probe-cache hit rate, not the reaction path).
//!   - Run the forward.
//!   - Assert `probe_calls_count` is at most
//!     `ceil(elapsed_ms / 90)` — slight slack vs the 100 ms TTL
//!     to absorb scheduler jitter and the initial cache miss.
//!     (90 ms instead of 100 ms gives `ceil` headroom of ~10 %
//!     over the cache TTL.)
//!
//! No probe call must be charged per-node. The lower bound is 1
//! (the first call is always a cache miss); the practical upper
//! bound on a sub-second forward is ~1–2.
//!
//! Why this test deserves a separate file rather than living inside
//! `signal_bus`'s own unit tests: those test the cache primitive
//! in isolation; this file proves the cache survives integration
//! through the full executor + guard hook chain. Moving the
//! amortisation contract to graph-level lets a future regression
//! in `check_guard_before_node` (e.g. someone adds a second
//! `collect_guard_conditions` call inside an arm) be caught here.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::reactive::ReactiveExecutionContext;
use atenia_engine::amm::ram_probe::{RamProbeApi, RamProbeError, RamSnapshot};
use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::amm::vram_probe::{VramProbeApi, VramProbeError, VramSnapshot};
use atenia_engine::tensor::Tensor;
use atenia_engine::v15::policy::types::DecisionBias;
use atenia_engine::v16::contract::constraints::{Constraints, RuntimeState};
use atenia_engine::v16::contract::execution_contract::{
    ExecutionBackend, ExecutionContract,
};
use atenia_engine::v16::guards::execution_guard::ExecutionGuard;
use atenia_engine::v16::guards::guard_manager::GuardManager;
use atenia_engine::v16::guards::simple_memory_pressure_guard::SimpleMemoryPressureGuard;

// ----- Mock probes that always report low pressure ------------------

struct LowPressureVramProbe(AtomicU64);
impl VramProbeApi for LowPressureVramProbe {
    fn snapshot(&self) -> Result<VramSnapshot, VramProbeError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(VramSnapshot {
            total_bytes: 1000,
            free_bytes: 900,
            used_bytes: 100,
        })
    }
}

struct LowPressureRamProbe(AtomicU64);
impl RamProbeApi for LowPressureRamProbe {
    fn snapshot(&self) -> Result<RamSnapshot, RamProbeError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(RamSnapshot {
            total_bytes: 1000,
            available_bytes: 900,
            used_bytes: 100,
        })
    }
}

fn permissive_contract() -> ExecutionContract {
    // Mirrors `tests/m3_e_11_5_promotion_test.rs::permissive_contract`.
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

fn cache_dir() -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("atenia_m4_7_5_a_{}", Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("test cache dir");
    dir
}

#[test]
fn probe_cache_amortises_under_multi_node_forward() {
    // Build a chain of 200 Add nodes. Each Add reads its two inputs,
    // produces an output. The graph executor walks the plan top-down
    // and fires `check_guard_before_node` on every node — this is the
    // hot path the cache must amortise.
    let mut gb = GraphBuilder::new();
    let in_a = gb.input();
    let in_b = gb.input();
    let mut tail = gb.add(in_a, in_b);
    let n = 200usize;
    for _ in 0..(n - 1) {
        tail = gb.add(tail, in_b);
    }
    let _ = gb.output(tail);
    let mut graph = gb.build();

    // Reactive context with the canonical SimpleMemoryPressureGuard
    // (Degrade above 0.65). The mock probes report 10 % usage so the
    // guard never fires — this isolates the cache amortisation from
    // the migration arms.
    let vram_probe = Arc::new(LowPressureVramProbe(AtomicU64::new(0)));
    let ram_probe = Arc::new(LowPressureRamProbe(AtomicU64::new(0)));
    let bus = Arc::new(SignalBus::with_probes(
        None,
        None,
        None,
        None,
        Some(vram_probe.clone()),
        Some(ram_probe.clone()),
    ));
    let guards: Vec<Box<dyn ExecutionGuard>> =
        vec![Box::new(SimpleMemoryPressureGuard::new())];
    let gm = GuardManager::new(guards);
    let dir = cache_dir();
    graph.set_reactive_context(
        ReactiveExecutionContext::new_without_gc(bus.clone(), permissive_contract(), gm)
            .with_cache_dir(dir.clone()),
    );

    // Snapshot the probe counter before, run the forward, snapshot
    // after. The two `Tensor::new_cpu` arguments are tiny (1×1
    // scalars) so the forward itself is sub-second; this puts the
    // amortisation under maximum stress (every `check_guard_before_node`
    // call is racing the previous one's cache entry).
    let before = bus.probe_calls_count();
    let start = Instant::now();
    let _outputs = graph.execute(vec![
        Tensor::new_cpu(vec![1, 1], vec![1.0]),
        Tensor::new_cpu(vec![1, 1], vec![2.0]),
    ]);
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let after = bus.probe_calls_count();

    let probe_delta = after - before;

    // Sanity checks.
    assert!(
        probe_delta >= 1,
        "expected at least one probe call (initial cache miss), got {}",
        probe_delta
    );

    // Amortisation contract: at SIGNAL_BUS_CACHE_TTL = 100 ms, the
    // upper bound on cache misses for an `elapsed_ms` walk is
    // `ceil(elapsed_ms / 100) + 1` (the +1 is the unconditional
    // initial miss; we use 90 ms as the divisor to absorb up to
    // ~10 % scheduler jitter that could shift cache-entry boundaries
    // earlier than nominal).
    let expected_max = (elapsed_ms / 90) + 2;

    println!(
        "probe_cache amortisation: {} nodes, {} ms wall, {} probe calls, expected_max {}",
        n, elapsed_ms, probe_delta, expected_max
    );

    assert!(
        probe_delta <= expected_max,
        "probe count {} exceeds amortisation budget {} for elapsed {} ms ({} nodes). \
         Either SIGNAL_BUS_CACHE_TTL is no longer being honoured, or someone added a \
         second collect_guard_conditions() call inside check_guard_before_node — see \
         M4.7.5.a TODO closure on `Graph::check_guard_before_node`.",
        probe_delta,
        expected_max,
        elapsed_ms,
        n,
    );

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn probe_cache_does_not_charge_per_node_when_ttl_dominates() {
    // Stronger version of the contract: deliberately stress with
    // 500 nodes and verify that the probe count stays in the
    // single digits at ms-class wall time. If the cache were
    // bypassed, a 500-node forward would charge 500 × ~150 ms =
    // 75 seconds and 500 probe calls; under the cache it should
    // be at most ~10.
    let mut gb = GraphBuilder::new();
    let a = gb.input();
    let b = gb.input();
    let mut tail = gb.add(a, b);
    for _ in 0..499 {
        tail = gb.add(tail, b);
    }
    let _ = gb.output(tail);
    let mut graph = gb.build();

    let vram_probe = Arc::new(LowPressureVramProbe(AtomicU64::new(0)));
    let ram_probe = Arc::new(LowPressureRamProbe(AtomicU64::new(0)));
    let bus = Arc::new(SignalBus::with_probes(
        None,
        None,
        None,
        None,
        Some(vram_probe),
        Some(ram_probe),
    ));
    let guards: Vec<Box<dyn ExecutionGuard>> =
        vec![Box::new(SimpleMemoryPressureGuard::new())];
    let gm = GuardManager::new(guards);
    let dir = cache_dir();
    graph.set_reactive_context(
        ReactiveExecutionContext::new_without_gc(bus.clone(), permissive_contract(), gm)
            .with_cache_dir(dir.clone()),
    );

    let before = bus.probe_calls_count();
    let start = Instant::now();
    let _outputs = graph.execute(vec![
        Tensor::new_cpu(vec![1, 1], vec![0.5]),
        Tensor::new_cpu(vec![1, 1], vec![0.25]),
    ]);
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let after = bus.probe_calls_count();
    let probe_delta = after - before;

    println!(
        "stress: 500 nodes, {} ms wall, {} probe calls",
        elapsed_ms, probe_delta
    );

    // Hard ceiling: even if the forward took 200 ms, that allows
    // at most 4 probe cycles. We assert <= 10 to absorb extreme
    // jitter (concurrent build noise, GC pauses, etc.) while
    // staying well below "one per node".
    assert!(
        probe_delta <= 10,
        "probe count {} exceeded hard ceiling 10 over a 500-node forward; \
         indicates the SignalBus cache was bypassed.",
        probe_delta
    );

    let _ = std::fs::remove_dir_all(&dir);
}
