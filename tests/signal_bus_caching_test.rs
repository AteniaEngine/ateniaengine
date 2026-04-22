//! APX v20 M3-e.4 — tests for memory-pressure probe caching in
//! `SignalBus`.
//!
//! The cache amortizes the ~40ms cost of a real probe (two
//! subprocesses: `nvidia-smi` + a RAM snapshot) over calls that
//! arrive within `SIGNAL_BUS_CACHE_TTL` of a prior probe. Coverage:
//!
//! 1. First call hits the probe → `probe_calls_count` becomes `> 0`.
//! 2. A second call issued immediately after the first uses the
//!    cache → counter stays put.
//! 3. A call issued after the TTL expires re-probes → counter
//!    increments.
//! 4. Signals that are *not* cached (`recent_failures` in this test)
//!    reflect injected state on the very next `collect_guard_conditions`,
//!    without triggering a new probe call.
//!
//! Tests that depend on the first probe actually succeeding are
//! wrapped in a graceful check: if the host has no working memory
//! probes (e.g. CI boxes without `nvidia-smi`), the probe returns
//! `None` and there is no cached value to observe. Those tests skip
//! with a `println!` instead of failing.

use std::thread;
use std::time::Duration;

use atenia_engine::amm::signal_bus::{SignalBus, SIGNAL_BUS_CACHE_TTL};

/// Runs the initial probe and reports whether the host supports it.
/// Returns `true` when `collect_guard_conditions` returned `Some` —
/// i.e. the probe subsystem is functional and subsequent cache-hit
/// assertions are meaningful. Returns `false` (with a skip message)
/// when the probe returned `None`, which happens on hosts without
/// `nvidia-smi` in `PATH`.
fn require_working_probe(bus: &SignalBus, test_name: &str) -> bool {
    if bus.collect_guard_conditions().is_some() {
        true
    } else {
        println!(
            "[TEST:{}] memory probe returned None (host lacks nvidia-smi or similar) -> graceful skip",
            test_name
        );
        false
    }
}

#[test]
fn test_first_call_increments_probe_count() {
    let bus = SignalBus::new();
    assert_eq!(
        bus.probe_calls_count(),
        0,
        "a fresh bus must start with zero probe calls"
    );

    // Single call: if the host supports the probe, the counter
    // should have advanced by exactly 1 regardless of cache state
    // (this is the very first call).
    let conditions = bus.collect_guard_conditions();

    if conditions.is_none() {
        println!(
            "[TEST:test_first_call_increments_probe_count] probe returned None \
             (host lacks nvidia-smi or similar) -> skipping positive assertion"
        );
        // Still assert the counter tracked the *attempt*: the
        // implementation increments before inspecting probe results,
        // so an attempted probe counts even on failure.
        assert_eq!(bus.probe_calls_count(), 1);
        return;
    }

    assert_eq!(
        bus.probe_calls_count(),
        1,
        "exactly one probe should have been invoked after the first collect"
    );
}

#[test]
fn test_second_call_within_ttl_uses_cache() {
    let bus = SignalBus::new();
    if !require_working_probe(&bus, "test_second_call_within_ttl_uses_cache") {
        return;
    }

    // After require_working_probe, exactly one probe has run and the
    // cache holds a fresh value.
    let count_after_first = bus.probe_calls_count();
    assert_eq!(count_after_first, 1);

    // Immediate second call: must hit the cache, so the counter is
    // unchanged. No sleep here — staying well inside the TTL.
    let _ = bus.collect_guard_conditions();
    assert_eq!(
        bus.probe_calls_count(),
        count_after_first,
        "second call within the TTL must hit the cache and not run a new probe"
    );

    // A few more back-to-back calls for good measure.
    for _ in 0..5 {
        let _ = bus.collect_guard_conditions();
    }
    assert_eq!(
        bus.probe_calls_count(),
        count_after_first,
        "rapid-fire calls within the TTL must all hit the cache"
    );
}

#[test]
fn test_call_after_ttl_refreshes_cache() {
    let bus = SignalBus::new();
    if !require_working_probe(&bus, "test_call_after_ttl_refreshes_cache") {
        return;
    }

    let count_after_first = bus.probe_calls_count();
    assert_eq!(count_after_first, 1);

    // Sleep long enough that the cache entry goes stale. Add a 50ms
    // margin on top of the TTL to absorb scheduler jitter — 150ms
    // for a 100ms TTL.
    thread::sleep(SIGNAL_BUS_CACHE_TTL + Duration::from_millis(50));

    let _ = bus.collect_guard_conditions();
    assert_eq!(
        bus.probe_calls_count(),
        count_after_first + 1,
        "a call after the TTL has elapsed must re-probe"
    );
}

#[test]
fn test_injected_failures_visible_through_cached_memory_pressure() {
    let bus = SignalBus::new();
    if !require_working_probe(
        &bus,
        "test_injected_failures_visible_through_cached_memory_pressure",
    ) {
        return;
    }

    // The first collect populated the cache and (if the host has no
    // failures on record) set `recent_failures = 0`.
    let baseline = bus.collect_guard_conditions().expect("probe succeeded");
    let count_after_baseline = bus.probe_calls_count();

    // Record a failure after the cache was populated. The cached
    // value covers only `memory_pressure`; `recent_failures` must
    // reflect the injection on the very next call.
    bus.failure_counter().record_failure();

    let after_injection = bus
        .collect_guard_conditions()
        .expect("probe still succeeds; cache is valid so no new probe needed");

    assert_eq!(
        after_injection.recent_failures,
        baseline.recent_failures + 1,
        "recent_failures must reflect the injected failure on the next call (not cached)"
    );
    assert_eq!(
        bus.probe_calls_count(),
        count_after_baseline,
        "injecting a failure must not trigger a probe: recent_failures is read \
         fresh from the FailureCounter while memory_pressure stays on cache"
    );
}
