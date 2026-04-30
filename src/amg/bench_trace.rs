//! M6.a — per-NodeType wall-clock instrumentation for
//! `Graph::execute_single`.
//!
//! Off by default. `cargo build --features bench-trace`
//! turns the instrumentation on. Adds ~10 ns per executed
//! node (a single `Instant::now()` pair + atomic adds)
//! when active, zero overhead when off (no symbols emitted).
//!
//! The accumulators are global `OnceLock<Mutex<HashMap<...>>>`
//! keyed by a `&'static str` op-name produced by
//! `node_type_name`. Two counters per key: total wall-clock
//! and call count. The bench example calls
//! [`reset`] before the warmup, runs, and reads the snapshot
//! via [`snapshot`].
//!
//! Thread-safety: the executor is single-threaded for
//! Llama inference today, but rayon-parallel matmul kernels
//! are off the hot path of `execute_single` itself, so a
//! `Mutex` lock per node is fine. If a future M6+ surfaces
//! a multi-threaded executor, swap the Mutex for a sharded
//! per-thread accumulator merged on snapshot.
//!
//! ## Why a global table and not a stack-allocated trace
//!
//! The decode loop runs ~360 nodes per step on 13B and the
//! bench wants per-NodeType aggregates, not a chronologically
//! ordered dump. A flat `HashMap<op, (calls, ns)>` is the
//! simplest shape that delivers what the report needs.
//!
//! ## Public API surface
//!
//! ```ignore
//! use atenia_engine::amg::bench_trace::{reset, snapshot, BenchEntry};
//! reset();
//! graph.execute(...);
//! let snap = snapshot();
//! for (op, entry) in &snap {
//!     println!("{op:>16}: {} calls, {:.3} ms total",
//!         entry.calls, entry.total_ns as f64 / 1e6);
//! }
//! ```

#[cfg(feature = "bench-trace")]
use std::collections::HashMap;
#[cfg(feature = "bench-trace")]
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "bench-trace")]
use std::time::Duration;

/// One per-NodeType accumulator entry.
#[derive(Debug, Clone, Default)]
pub struct BenchEntry {
    pub calls: u64,
    pub total_ns: u64,
}

#[cfg(feature = "bench-trace")]
fn table() -> &'static Mutex<HashMap<&'static str, BenchEntry>> {
    static T: OnceLock<Mutex<HashMap<&'static str, BenchEntry>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Reset every accumulator to zero. Call before a measured run.
#[cfg(feature = "bench-trace")]
pub fn reset() {
    table().lock().unwrap().clear();
}
#[cfg(not(feature = "bench-trace"))]
pub fn reset() {}

/// Snapshot of the current accumulators, sorted by total_ns
/// descending so the heaviest ops surface first.
#[cfg(feature = "bench-trace")]
pub fn snapshot() -> Vec<(&'static str, BenchEntry)> {
    let map = table().lock().unwrap();
    let mut v: Vec<_> = map.iter().map(|(k, v)| (*k, v.clone())).collect();
    v.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));
    v
}
#[cfg(not(feature = "bench-trace"))]
pub fn snapshot() -> Vec<(&'static str, BenchEntry)> { Vec::new() }

/// Record one node execution. Called from `Graph::execute_single`
/// via the `record!` macro below. Inlined to keep the cost a
/// single atomic add when there's no contention.
#[cfg(feature = "bench-trace")]
#[inline]
pub fn record(op: &'static str, elapsed: Duration) {
    let mut map = table().lock().unwrap();
    let entry = map.entry(op).or_default();
    entry.calls += 1;
    entry.total_ns += elapsed.as_nanos() as u64;
}

/// True iff the bench-trace feature is compiled in. Used by
/// the bench example to print a hint when run against a
/// non-instrumented build.
pub fn is_enabled() -> bool {
    cfg!(feature = "bench-trace")
}
