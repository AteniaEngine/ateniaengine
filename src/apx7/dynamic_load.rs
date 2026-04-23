//! APX 7.4 dynamic load sampling.
//!
//! Produces a `LoadSnapshot` consumed by `matmul_dispatcher` when
//! `ATENIA_APX_MODE >= 7.4` to pick between `seq` / `pex` / `ws` /
//! `pgl` matmul strategies.
//!
//! # History: the debt #7 fix
//!
//! Before M3-e.11 debt #7, `sample_system_load` constructed a fresh
//! `sysinfo::System` on every call and read `cpu_usage()` from it.
//! `sysinfo` computes CPU usage as a delta between two successive
//! refreshes, so a freshly-constructed `System` always reports
//! `0.0%` for every core — the module silently returned
//! `cpu_load = 0.0` in production since its introduction, and the
//! `seq` branch of `choose_strategy` (which triggers only for
//! `cpu_load > 85.0`) had never been exercised. The bug was
//! discovered during M3-e.6 investigation, which established the
//! same stateful-probe pattern for the new `CpuProbe` in
//! `src/amm/cpu_probe.rs`.
//!
//! This module keeps a single `sysinfo::System` alive for the
//! lifetime of the process, guarded by `Mutex` for thread-safe
//! access from matmul dispatchers running on any worker thread. A
//! one-time warmup on the first `sample_system_load` call pays
//! the mandatory `sysinfo::MINIMUM_CPU_UPDATE_INTERVAL` (~200 ms
//! on all supported platforms) to establish the baseline, after
//! which every subsequent call does a single fast `refresh_cpu`
//! + `cpus()` read.
//!
//! The fix is **local-stateful** — it deliberately does NOT route
//! through `CpuProbe` from M3-e.6. Consolidating both paths onto a
//! single probe is valid future work but it is a larger refactor
//! (dynamic_load's consumer is `matmul_dispatcher`, which has no
//! direct access to a `SignalBus`), out of scope for this fix.

use std::sync::{Mutex, OnceLock, RwLock};
use sysinfo::{System, MINIMUM_CPU_UPDATE_INTERVAL};

#[derive(Debug, Clone, Copy)]
pub struct LoadSnapshot {
    pub cpu_load: f32,       // percentage (0.0 .. 100.0 per-core-averaged)
    pub threads_available: usize,
}

pub static LAST_SNAPSHOT: RwLock<LoadSnapshot> = RwLock::new(LoadSnapshot {
    cpu_load: 0.0,
    threads_available: 1,
});

/// Process-wide `sysinfo::System` that persists across calls.
///
/// Lazily initialized on the first `sample_system_load` call. The
/// `OnceLock` holds the `Mutex<System>` and also a flag tracking
/// whether the warmup pair of refreshes has completed; the flag
/// lives inside the mutex to avoid extra synchronization overhead.
static SYS_STATE: OnceLock<Mutex<SysState>> = OnceLock::new();

struct SysState {
    sys: System,
    /// `true` after the two-refresh warmup has completed. The
    /// first `sample_system_load` call observes this as `false`,
    /// performs the warmup (refresh + sleep + refresh), then sets
    /// it to `true`. Subsequent calls do a single `refresh_cpu`.
    warmed_up: bool,
}

fn sys_state() -> &'static Mutex<SysState> {
    SYS_STATE.get_or_init(|| {
        Mutex::new(SysState {
            sys: System::new(),
            warmed_up: false,
        })
    })
}

/// Sample current CPU load and derive a [`LoadSnapshot`].
///
/// **First-call cost**: ~200 ms of warmup (two `sysinfo` refreshes
/// separated by `MINIMUM_CPU_UPDATE_INTERVAL`) to establish a
/// baseline. Subsequent calls are fast — a single `refresh_cpu`
/// + `cpus()` read, typically tens of microseconds.
///
/// Thread-safe: the internal `System` is behind a `Mutex`. Concurrent
/// callers serialize on that mutex; with the fast per-call cost
/// that serialization is negligible relative to the matmul work
/// the caller is dispatching.
pub fn sample_system_load() -> LoadSnapshot {
    let state = sys_state();
    let mut guard = state.lock().unwrap_or_else(|p| p.into_inner());

    // Warmup on first call. `sysinfo` reports CPU usage as a delta
    // between two refreshes, so a freshly-constructed System reads
    // 0.0 after only one refresh. Pay the ~200 ms warmup cost
    // once per process so subsequent calls produce real data.
    if !guard.warmed_up {
        guard.sys.refresh_cpu();
        // Drop the mutex during sleep so concurrent waiters don't
        // block on us for 200 ms. They'll re-acquire afterwards
        // and find `warmed_up = true`.
        drop(guard);
        std::thread::sleep(MINIMUM_CPU_UPDATE_INTERVAL);
        let mut guard = state.lock().unwrap_or_else(|p| p.into_inner());
        // A concurrent caller might have warmed up already while
        // we were sleeping. Only do the second refresh if we are
        // still the warmup candidate.
        if !guard.warmed_up {
            guard.sys.refresh_cpu();
            guard.warmed_up = true;
        }
        // Fall through to the common sample-read path below.
        return read_and_record(&mut guard);
    }

    guard.sys.refresh_cpu();
    read_and_record(&mut guard)
}

/// Shared tail between the warmup-path and the steady-state
/// path: read `cpu_usage()` from the already-refreshed `System`,
/// compute `LoadSnapshot`, persist in `LAST_SNAPSHOT`, return.
fn read_and_record(state: &mut SysState) -> LoadSnapshot {
    let mut load = 0.0f32;
    let cpus = state.sys.cpus();
    if !cpus.is_empty() {
        for cpu in cpus {
            load += cpu.cpu_usage() as f32;
        }
        load /= cpus.len() as f32;
    }

    let total_threads = num_cpus::get_physical().max(1);
    let mut threads_available = ((1.0 - (load / 100.0)) * total_threads as f32).round() as usize;
    if threads_available < 1 {
        threads_available = 1;
    }

    let snap = LoadSnapshot {
        cpu_load: load,
        threads_available,
    };

    *LAST_SNAPSHOT.write().unwrap() = snap;

    snap
}

pub fn get_last_snapshot() -> LoadSnapshot {
    *LAST_SNAPSHOT.read().unwrap()
}

/// Suggested scheduling strategy according to the current snapshot.
/// Returns "seq", "pex", "ws" or "pgl" (heuristic fallback).
pub fn choose_strategy(snap: &LoadSnapshot) -> &'static str {
    if snap.cpu_load > 85.0 {
        return "seq";
    }

    if snap.threads_available <= 4 {
        return "pex";
    }

    if snap.threads_available >= 12 {
        return "ws";
    }

    "pgl"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;

    /// Test 1: deterministic no-panic / well-formed output.
    ///
    /// Calls `sample_system_load` twice with a sleep between
    /// them, and asserts that both samples produce well-formed
    /// values: no panic, no NaN, no infinity, `threads_available >
    /// 0`. Does NOT assert a specific `cpu_load` — that depends on
    /// the host's actual load at test time.
    #[test]
    fn test_sample_system_load_returns_well_formed_values() {
        let first = sample_system_load();
        assert!(
            first.cpu_load.is_finite(),
            "first cpu_load must be finite, got {}",
            first.cpu_load
        );
        assert!(
            !first.cpu_load.is_nan(),
            "first cpu_load must not be NaN"
        );
        assert!(
            first.threads_available >= 1,
            "threads_available must be >= 1, got {}",
            first.threads_available
        );

        // Sleep long enough that sysinfo's minimum-interval
        // constraint is satisfied and the second read computes a
        // fresh delta.
        thread::sleep(MINIMUM_CPU_UPDATE_INTERVAL + Duration::from_millis(50));

        let second = sample_system_load();
        assert!(second.cpu_load.is_finite());
        assert!(!second.cpu_load.is_nan());
        assert!(second.threads_available >= 1);
    }

    /// Test 2: smoke test that sampling can produce non-zero
    /// readings under real CPU load.
    ///
    /// Spawns a busy thread that burns CPU for ~500 ms, then
    /// samples three times spaced apart. Asserts that AT LEAST
    /// ONE of the three samples reports `cpu_load > 0.0`. Three
    /// samples with an "at least one > 0" criterion tolerates CI
    /// schedulers that might time-slice the busy thread unevenly
    /// — the pre-fix behavior (every sample stuck at 0.0) would
    /// fail all three, and after the fix we need any one to
    /// succeed.
    #[test]
    fn test_sample_detects_real_cpu_load() {
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let busy = thread::spawn(move || {
            // Tight compute loop. `black_box` keeps the
            // optimizer from eliminating the work.
            let mut acc: f64 = 1.0;
            while !stop_clone.load(Ordering::Relaxed) {
                for _ in 0..10_000 {
                    acc = std::hint::black_box(acc * 1.0000001 + 0.0000001);
                }
            }
            acc
        });

        // Warmup + first sample (discarded — warmup establishes
        // baseline, and we want the subsequent samples to
        // reflect the busy thread we've just kicked off).
        let _ = sample_system_load();

        let mut samples: Vec<f32> = Vec::with_capacity(3);
        for _ in 0..3 {
            thread::sleep(Duration::from_millis(150));
            samples.push(sample_system_load().cpu_load);
        }

        stop.store(true, Ordering::Relaxed);
        let _ = busy.join();

        assert!(
            samples.iter().any(|&v| v > 0.0),
            "at least one sample must detect the busy thread: {:?}",
            samples
        );
    }
}
