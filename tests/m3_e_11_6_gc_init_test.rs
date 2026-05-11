//! APX v20 M3-e.11.6 — tests for the automatic orphan-file GC
//! that runs during `ReactiveExecutionContext::new()`, and the
//! `new_without_gc` escape hatch that tests use to avoid
//! touching the user's default cache dir.
//!
//! Test setup uses `std::fs::FileTimes` + `File::set_times`
//! (stable since Rust 1.75) to age files artificially without
//! sleeping for minutes or pulling in the `filetime` crate.

use std::fs::{self, File, FileTimes};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

use atenia_engine::amg::reactive::ReactiveExecutionContext;
use atenia_engine::amm::signal_bus::SignalBus;
use atenia_engine::v15::policy::types::DecisionBias;
use atenia_engine::v16::contract::constraints::{Constraints, RuntimeState};
use atenia_engine::v16::contract::execution_contract::{ExecutionBackend, ExecutionContract};
use atenia_engine::v16::guards::execution_guard::ExecutionGuard;
use atenia_engine::v16::guards::guard_manager::GuardManager;
use atenia_engine::v16::guards::simple_memory_pressure_guard::SimpleMemoryPressureGuard;

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn test_cache_dir(label: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("atenia_m3_e_11_6_gc_{}_{}", label, Uuid::new_v4()));
    fs::create_dir_all(&dir).expect("create test cache dir");
    dir
}

fn cleanup(dir: &PathBuf) {
    let _ = fs::remove_dir_all(dir);
}

/// Write a bytes-only placeholder at the named path and adjust
/// its modification time to `age_ago` in the past. Panics on IO
/// error — callers have a simple, clean test context and should
/// never see one.
fn create_aged_file(path: &PathBuf, age_ago: Duration) {
    fs::write(path, b"placeholder").expect("write placeholder");
    let old = SystemTime::now() - age_ago;
    let f = File::options()
        .write(true)
        .open(path)
        .expect("open for set_times");
    let times = FileTimes::new().set_modified(old).set_accessed(old);
    f.set_times(times).expect("set_times");
}

/// Build a minimal SignalBus + Contract + GuardManager triplet.
/// Tests here do not care about the probes — they only exercise
/// ReactiveExecutionContext's construction-time GC behavior.
fn trivial_bus_contract_guards() -> (Arc<SignalBus>, ExecutionContract, GuardManager) {
    let bus = Arc::new(SignalBus::with_probes(None, None, None, None, None, None));
    let contract = ExecutionContract {
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
    };
    let guards: Vec<Box<dyn ExecutionGuard>> = vec![Box::new(SimpleMemoryPressureGuard::new())];
    let gm = GuardManager::new(guards);
    (bus, contract, gm)
}

// ---------------------------------------------------------------------
// Test 1: GC sweeps aged orphans at init
// ---------------------------------------------------------------------

#[test]
fn test_gc_automatic_on_context_init() {
    let dir = test_cache_dir("auto_gc");

    // Seed: two aged orphans + one fresh "live" file. The fresh
    // one should survive, the aged ones should not.
    let orphan_a = dir.join("tensor_aaaa.bin");
    let orphan_b = dir.join("tensor_bbbb.bin");
    let fresh = dir.join("tensor_cccc.bin");
    create_aged_file(&orphan_a, Duration::from_secs(15 * 60)); // 15 min ago
    create_aged_file(&orphan_b, Duration::from_secs(11 * 60)); // 11 min ago
    fs::write(&fresh, b"fresh").expect("write fresh");

    assert!(orphan_a.exists());
    assert!(orphan_b.exists());
    assert!(fresh.exists());

    // Build a context with GC enabled, pointed at our test dir.
    // `new_without_gc` + `with_cache_dir` + `run_startup_gc`
    // reproduces exactly what `new()` would do on the default
    // dir, without touching the default dir.
    let (bus, contract, gm) = trivial_bus_contract_guards();
    let _ctx = ReactiveExecutionContext::new_without_gc(bus, contract, gm)
        .with_cache_dir(dir.clone())
        .run_startup_gc();

    // Aged files gone, fresh file alive.
    assert!(
        !orphan_a.exists(),
        "orphan_a (15 min old) must have been swept"
    );
    assert!(
        !orphan_b.exists(),
        "orphan_b (11 min old) must have been swept"
    );
    assert!(
        fresh.exists(),
        "fresh file (well under 10 min) must have survived"
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------
// Test 2: GC can be disabled via new_without_gc
// ---------------------------------------------------------------------

#[test]
fn test_gc_disabled_via_new_without_gc() {
    let dir = test_cache_dir("disabled");

    // Seed an aged orphan.
    let orphan = dir.join("tensor_old.bin");
    create_aged_file(&orphan, Duration::from_secs(15 * 60));
    assert!(orphan.exists());

    // Build the context WITHOUT GC. The file should still be
    // there afterwards — this is the escape hatch tests need
    // when they want to inspect a cache dir without the GC
    // racing them.
    let (bus, contract, gm) = trivial_bus_contract_guards();
    let _ctx =
        ReactiveExecutionContext::new_without_gc(bus, contract, gm).with_cache_dir(dir.clone());

    assert!(
        orphan.exists(),
        "aged file must survive when GC is disabled via new_without_gc"
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------
// Test 3: GC handles missing cache dir gracefully
// ---------------------------------------------------------------------

#[test]
fn test_gc_handles_missing_cache_dir_gracefully() {
    // Point at a path that definitely does not exist — GC at init
    // must not panic or error out, because that would break
    // startup on any host without a pre-existing cache dir
    // (which is the usual state on a first run).
    let missing =
        std::env::temp_dir().join(format!("atenia_m3_e_11_6_gc_missing_{}", Uuid::new_v4()));
    assert!(!missing.exists(), "pre-condition: dir must not exist");

    let (bus, contract, gm) = trivial_bus_contract_guards();
    let _ctx = ReactiveExecutionContext::new_without_gc(bus, contract, gm)
        .with_cache_dir(missing.clone())
        .run_startup_gc();

    // Nothing should have panicked. The directory is still
    // absent (GC does not create the dir — only the migration
    // primitives do).
    assert!(
        !missing.exists(),
        "GC must not create the dir just by running on it"
    );
    // No cleanup needed — we never created it.
}
