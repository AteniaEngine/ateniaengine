//! Reactive execution layer for AMG graphs (APX v20 M2).
//!
//! Groups the three pieces a graph needs to gate execution on live
//! telemetry: a signal bus that produces `GuardConditions`, a
//! contract that declares what is legally allowed, and a guard
//! manager that combines multiple guards into a single verdict.
//!
//! A `Graph` carries an `Option<ReactiveExecutionContext>`. When set,
//! schedulers consult `Graph::check_guard_before_node` before each
//! node and abort cleanly if a guard triggers. When unset (the
//! default), execution behavior is identical to pre-M2.
//!
//! This module also defines the abort reason enum surfaced by the
//! checked execution path.

use std::collections::VecDeque;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::amm::signal_bus::SignalBus;
use crate::tensor::disk_tier;
use crate::v16::contract::execution_contract::ExecutionContract;
use crate::v16::guards::guard_conditions::GuardConditions;
use crate::v16::guards::guard_manager::GuardManager;

/// M4.7.5.b — touch-ordered LRU of graph node IDs, used by the
/// per-tensor selective-spill policy that lands in M4.7.5.c+.d.
///
/// The structure is a single `Mutex<VecDeque<usize>>`. On a touch:
/// the existing entry (if any) is removed via `retain` (O(N)) and
/// the id is appended at the back, so the front of the deque is
/// the *least recently used* node and the back is the *most
/// recently used*. The `migrate_all_cpu_to_disk` selective sibling
/// (M4.7.5.c) reads the front to pick eviction candidates.
///
/// **Cost analysis** (Risk #1 of the M4.7.5 investigation): O(N)
/// per touch on a deque of N nodes. For TinyLlama (N≈200,
/// 200 ops/forward) the touch cost is ~40k pointer comparisons per
/// forward; at ~1 ns each, ~40 µs total — well under 0.01% of a
/// 30 s forward. For Llama 2 13B (N≈1850, ~1850 ops/forward) the
/// touch cost is ~3.4 M comparisons ≈ 3.4 ms, still <0.02% of a
/// 30 s budget. A doubly-linked-list LRU would be O(1) per touch
/// at the cost of unsafe-adjacent code; the simpler structure is
/// chosen for M4.7.5 and re-evaluated in M5 if the
/// micro-benchmark surfaces a regression.
///
/// The LRU is touched from `NodeTimingRecorder::drop` so it
/// reflects the **completion order** of nodes, not their
/// scheduling order — which is what selective-spill policy needs
/// (a node that just finished is the most recently used; one
/// that finished 30 layers ago is a strong eviction candidate).
///
/// Lock contention is non-existent in the executor's
/// single-thread hot path. The mutex is taken explicitly so a
/// future multi-thread executor (M5+) keeps correctness without
/// changes to this API.
#[derive(Debug)]
pub struct TouchOrder {
    inner: Mutex<VecDeque<usize>>,
}

impl TouchOrder {
    /// Construct an empty touch order.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(VecDeque::new()),
        }
    }

    /// Record `node_id` as the most recently used. If `node_id` was
    /// already present, it is moved to the back; otherwise it is
    /// appended. Idempotent.
    ///
    /// A poisoned mutex is silently recovered (`into_inner`); the
    /// touch is a best-effort signal — losing one update is far
    /// less harmful than poisoning the executor on a benign
    /// concurrent panic in tests.
    pub fn touch(&self, node_id: usize) {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        // Remove any prior entry, then append. `retain` is O(N) on
        // the deque; see the type's docstring for the cost analysis.
        guard.retain(|&id| id != node_id);
        guard.push_back(node_id);
    }

    /// Snapshot the touch order as a `Vec<usize>`, oldest first
    /// (front of the deque) to most recently used (back). Used by
    /// the selective-spill policy to pick eviction candidates and
    /// by tests to assert the order.
    pub fn snapshot(&self) -> Vec<usize> {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.iter().copied().collect()
    }

    /// Number of distinct node ids currently tracked.
    pub fn len(&self) -> usize {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.len()
    }

    /// Drop every entry. Used by tests that want a deterministic
    /// starting state across multiple `execute` calls on the same
    /// graph.
    pub fn clear(&self) {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.clear();
    }
}

impl Default for TouchOrder {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime reactive execution layer attached to a `Graph`.
///
/// Cheap to attach; `signal_bus` is an `Arc` and the other two are
/// small owned structs. Future milestones (policy evaluation in M3+,
/// strategy selection in M4+) will add fields here rather than on
/// `Graph` itself.
pub struct ReactiveExecutionContext {
    pub signal_bus: Arc<SignalBus>,
    pub contract: ExecutionContract,
    pub guard_manager: GuardManager,
    /// M3-e.5: per-context counter of processed `GuardAction::Degrade`
    /// verdicts. Incremented each time the guard-handling site observes
    /// a `Degrade` action and initiates a migration, whether the
    /// migration itself succeeds or fails. Readable via
    /// [`degrade_events_count`](Self::degrade_events_count).
    pub(crate) degrade_events_count: AtomicU64,
    /// M3-e.6: per-context counter of `Degrade` verdicts that were
    /// vetoed by the CPU-availability precondition. Incremented when
    /// the guard said "migrate" but the CPU signals say another
    /// process is saturating CPU and Atenia's contribution is small,
    /// so migration would make the external pressure worse. Not a
    /// subset of `degrade_events_count` — vetoed verdicts never
    /// reached the migration site. Readable via
    /// [`degrade_vetoed_by_cpu_count`](Self::degrade_vetoed_by_cpu_count).
    pub(crate) degrade_vetoed_by_cpu_count: AtomicU64,
    /// M3-e.11.5: per-context counter of processed
    /// `GuardAction::DeepDegrade` verdicts — whether the verdict
    /// originated directly from a guard or was promoted from
    /// `Degrade` by the `dual_memory_pressure` check at the
    /// reaction site. Disjoint from `degrade_events_count`:
    /// promoted verdicts count as DeepDegrade, not Degrade. The
    /// counter is incremented even when `migrate_all_to_disk`
    /// fails, matching the "count the attempt" policy established
    /// in M3-e.5.
    pub(crate) deep_degrade_events_count: AtomicU64,
    /// M3-e.11.5: directory under which `DeepDegrade` migrations
    /// write their spillover files. Populated with
    /// [`disk_tier::default_cache_dir`] by [`Self::new`]; tests
    /// and advanced consumers override via
    /// [`Self::with_cache_dir`] to place the files in a
    /// deterministic location (typically a unique temp dir).
    pub cache_dir: PathBuf,
    /// M4.7.5.b: touch-ordered LRU of graph node ids. Updated
    /// from `NodeTimingRecorder::drop` so each node lands at the
    /// MRU end at completion. Read by the M4.7.5.c selective-spill
    /// policy to pick eviction candidates from the LRU end.
    /// Carried as `Arc` so the `NodeTimingRecorder` can hold a
    /// cheap clone alongside its `Arc<SignalBus>` without keeping
    /// the whole context alive for the duration of one node.
    pub(crate) lru_touch_order: Arc<TouchOrder>,
}

/// M3-e.6: system-wide CPU utilization threshold above which the
/// reaction path considers "the CPU is under pressure". Paired with
/// [`CPU_SELF_CONTRIBUTION_MIN`] to decide whether that pressure is
/// caused by Atenia or by something else.
pub const CPU_PRESSURE_TOTAL_THRESHOLD: f32 = 0.80;

/// M3-e.6: minimum share `self / total` that attributes CPU pressure
/// to Atenia. Below this, Atenia is considered **not responsible** —
/// some other process is loading the CPU and migrating more work to
/// the CPU would worsen the external pressure, so the Degrade arm
/// skips the migration.
pub const CPU_SELF_CONTRIBUTION_MIN: f32 = 0.50;

/// M3-e.7: format a compact fragment describing the GPU compute
/// utilization for inclusion in `[AMG Guard]` log lines. Returns
/// `" gpu_util_total=0.XX, gpu_util_self=0.XX,"` when both fields
/// are populated, or `" gpu_util=n/a,"` when either is absent.
///
/// The leading space and trailing comma let the caller inline the
/// fragment into an existing log format string without rebuilding
/// the whole message. Observability-only: the fragment never
/// changes behavior, it only adds diagnostic context to the log.
pub fn format_gpu_util_fragment(conditions: &GuardConditions) -> String {
    match (conditions.gpu_util_total, conditions.gpu_util_self) {
        (Some(total), Some(self_)) => format!(
            " gpu_util_total={:.2}, gpu_util_self={:.2},",
            total, self_
        ),
        _ => " gpu_util=n/a,".to_string(),
    }
}

/// M3-e.8: format a compact fragment describing the foreground-app
/// indicator for inclusion in `[AMG Guard]` log lines. Returns one
/// of three strings:
/// - `" foreground=atenia,"`  — the OS foreground is this process.
/// - `" foreground=other,"`   — the OS foreground is a different process.
/// - `" foreground=n/a,"`     — the probe could not determine (unsupported
///   platform, screen locked, etc.).
///
/// Same leading-space / trailing-comma conventions as the other
/// fragments. Observability-only; no behavior changes.
pub fn format_foreground_fragment(conditions: &GuardConditions) -> String {
    match conditions.foreground_is_atenia {
        Some(true) => " foreground=atenia,".to_string(),
        Some(false) => " foreground=other,".to_string(),
        None => " foreground=n/a,".to_string(),
    }
}

/// M3-e.10: format a compact fragment describing self-latency
/// state for inclusion in `[AMG Guard]` log lines.
///
/// Returns one of:
/// - `" latency=5.2ms→7.8ms (1.50x),"` — baseline and EWMA both
///   known; ratio above 1 means Atenia is running slower than its
///   own recent baseline.
/// - `" latency=n/a,"` — monitor has fewer than `min_samples`
///   measurements (cold baseline), or either field is missing.
///
/// Same leading-space / trailing-comma convention as the other
/// fragments. Observability-only.
pub fn format_latency_fragment(conditions: &GuardConditions) -> String {
    match (
        conditions.latency_baseline_ms,
        conditions.latency_current_ms,
        conditions.latency_ratio,
    ) {
        (Some(b), Some(c), Some(r)) => {
            format!(" latency={:.2}ms→{:.2}ms ({:.2}x),", b, c, r)
        }
        _ => " latency=n/a,".to_string(),
    }
}

/// M3-e.9: format a compact fragment describing battery state for
/// inclusion in `[AMG Guard]` log lines. The two `GuardConditions`
/// fields (`on_battery` and `battery_level`) can vary independently,
/// so the fragment is built from both:
/// - `" battery=plugged_0.85,"` — plugged in at 85% charge.
/// - `" battery=plugged,"`      — plugged in, level unknown.
/// - `" battery=on_0.15,"`      — on battery at 15% charge.
/// - `" battery=on,"`           — on battery, level unknown.
/// - `" battery=0.50,"`         — level known but AC state unknown (rare driver state).
/// - `" battery=n/a,"`          — neither field present (desktop or stub platform).
///
/// Same leading-space / trailing-comma convention as the other
/// fragments. Observability-only.
pub fn format_battery_fragment(conditions: &GuardConditions) -> String {
    match (conditions.on_battery, conditions.battery_level) {
        (Some(true), Some(level)) => format!(" battery=on_{:.2},", level),
        (Some(true), None) => " battery=on,".to_string(),
        (Some(false), Some(level)) => format!(" battery=plugged_{:.2},", level),
        (Some(false), None) => " battery=plugged,".to_string(),
        (None, Some(level)) => format!(" battery={:.2},", level),
        (None, None) => " battery=n/a,".to_string(),
    }
}

/// Decide whether a `Degrade` verdict should be vetoed because the
/// CPU is saturated by *external* processes rather than by Atenia.
///
/// Returns `true` (veto, skip migration) only when **all** of the
/// following hold:
/// - Both CPU fields are populated on `conditions`. If either is
///   `None` (probe absent or probe failure), the function returns
///   `false` — fail-open: an unknown CPU state must not block a
///   reaction the memory guard has already requested.
/// - `cpu_pressure_total > `[`CPU_PRESSURE_TOTAL_THRESHOLD`]: the
///   system is genuinely under CPU pressure.
/// - `self / total < `[`CPU_SELF_CONTRIBUTION_MIN`]: Atenia's share
///   of that pressure is small, so Atenia is not the cause.
///
/// See the M3-e.6 handoff scenarios table for the decision matrix
/// this implements.
pub fn cpu_saturated_externally(conditions: &GuardConditions) -> bool {
    let (total, self_) = match (conditions.cpu_pressure_total, conditions.cpu_pressure_self) {
        (Some(t), Some(s)) => (t, s),
        _ => return false,
    };
    if total <= CPU_PRESSURE_TOTAL_THRESHOLD {
        return false;
    }
    // `total` > 0.80 here, so it cannot be zero; division is safe.
    let share = self_ / total;
    share < CPU_SELF_CONTRIBUTION_MIN
}

/// M3-e.11.5: VRAM pressure threshold above which the reaction
/// site considers the GPU memory tier "saturated enough to
/// warrant disk spillover". Strictly above this value (`>`) is
/// required to match the semantics of `memory_pressure` in the
/// existing `SimpleMemoryPressureGuard`. Paired with
/// [`DEEP_DEGRADE_RAM_THRESHOLD`] in [`dual_memory_pressure`];
/// the conjunction ("both tiers saturated") is what distinguishes
/// `DeepDegrade` from plain `Degrade`.
///
/// The value 0.85 is **deliberately more conservative** than the
/// 0.65 threshold of `SimpleMemoryPressureGuard`: reaching this
/// level means regular `Degrade` migration is insufficient
/// (because RAM is also pressured), and the cost of disk I/O
/// justifies a stricter trigger.
pub const DEEP_DEGRADE_VRAM_THRESHOLD: f32 = 0.85;

/// M3-e.11.5: RAM pressure threshold, sibling of
/// [`DEEP_DEGRADE_VRAM_THRESHOLD`]. Same value (0.85) because
/// the signal we care about is symmetric: either tier alone
/// above 0.85 does **not** trigger promotion (plain `Degrade`
/// can free VRAM to RAM or just live with RAM pressure), only
/// both simultaneously. The stricter-than-Degrade rationale
/// above applies to both.
pub const DEEP_DEGRADE_RAM_THRESHOLD: f32 = 0.85;

/// M3-e.11.5: decide whether a `Degrade` verdict should be
/// **promoted** to `DeepDegrade` because both VRAM and RAM are
/// saturated simultaneously. Called by the reaction site BEFORE
/// the CPU-veto check — promotion to DeepDegrade bypasses the
/// CPU veto entirely because disk spillover does not add CPU
/// load the way Cpu-tier migration does.
///
/// Returns `true` (promote) only when **all** of the following
/// hold:
/// - `vram_pressure` is `Some(_)`.
/// - `ram_pressure` is `Some(_)`.
/// - `vram_pressure > DEEP_DEGRADE_VRAM_THRESHOLD`.
/// - `ram_pressure > DEEP_DEGRADE_RAM_THRESHOLD`.
///
/// Returns `false` in every other case — fail-open when either
/// signal is missing (the reaction site falls back to plain
/// `Degrade` behavior, which is the pre-M3-e.11.5 path and
/// always safe).
pub fn dual_memory_pressure(conditions: &GuardConditions) -> bool {
    match (conditions.vram_pressure, conditions.ram_pressure) {
        (Some(v), Some(r)) => {
            v > DEEP_DEGRADE_VRAM_THRESHOLD && r > DEEP_DEGRADE_RAM_THRESHOLD
        }
        _ => false,
    }
}

/// M3-e.11.5: format a compact fragment describing the memory
/// tiers for inclusion in `[AMG Guard]` log lines. Combines the
/// aggregate `memory_pressure` with the per-tier breakdown
/// introduced in M3-e.11.3:
///
/// - `" memory=vram=0.92,ram=0.87,total=0.92,"` when both tiers
///   and the aggregate are populated.
/// - `" memory=vram=0.72,total=0.72,"` when only VRAM probed.
/// - `" memory=ram=0.50,total=0.50,"` when only RAM probed.
/// - `" memory=n/a,"` when nothing is available.
///
/// Leading space and trailing comma match the convention of
/// the other log fragments (GPU util, foreground, battery,
/// latency). Observability-only.
pub fn format_memory_fragment(conditions: &GuardConditions) -> String {
    let total = conditions.memory_pressure;
    match (conditions.vram_pressure, conditions.ram_pressure) {
        (Some(v), Some(r)) => format!(
            " memory=vram={:.2},ram={:.2},total={:.2},",
            v, r, total
        ),
        (Some(v), None) => format!(" memory=vram={:.2},total={:.2},", v, total),
        (None, Some(r)) => format!(" memory=ram={:.2},total={:.2},", r, total),
        (None, None) => " memory=n/a,".to_string(),
    }
}

impl ReactiveExecutionContext {
    /// Construct a reactive context with GC-on-init enabled.
    ///
    /// On creation, [`disk_tier::gc_orphan_disk_tensors`] is
    /// invoked on the default cache dir with a 10-minute
    /// threshold. This sweeps files left behind by a previous
    /// process that crashed before its `Arc<InnerDiskFile>` drops
    /// could remove them (e.g. SIGKILL, power loss, panic=abort).
    /// The GC is **best-effort** — any error from sweep is
    /// silently ignored so startup never blocks on a GC hiccup.
    ///
    /// # Known limitation of the 10-minute threshold
    ///
    /// If a workload takes longer than 10 minutes between
    /// successive migrations of the same tensor, an otherwise-
    /// valid disk file older than 10 minutes could be swept by
    /// a sibling process's GC at startup. This is unlikely in
    /// practice — ML workloads touch tensors far more frequently
    /// than that — but the risk is real. Future improvements
    /// (lockfile pattern, mtime-touching by the live process)
    /// are tracked as follow-up work.
    ///
    /// Tests that want a deterministic cache dir should combine
    /// [`Self::new_without_gc`] with [`Self::with_cache_dir`] —
    /// that pair lets the test set the dir AND skip GC on a
    /// location it doesn't control. Use
    /// [`Self::run_startup_gc`] when the test **does** want to
    /// exercise the GC on its own dir.
    pub fn new(
        signal_bus: Arc<SignalBus>,
        contract: ExecutionContract,
        guard_manager: GuardManager,
    ) -> Self {
        let ctx = Self::new_without_gc(signal_bus, contract, guard_manager);
        let _ = disk_tier::gc_orphan_disk_tensors(&ctx.cache_dir, 10);
        ctx
    }

    /// M3-e.11.6: construct the context **without** running GC
    /// at init. The primary consumer is tests — the GC in
    /// [`Self::new`] touches whatever `default_cache_dir`
    /// resolves to on the host, which may perturb concurrent
    /// test runs or a real Atenia process's cache. Tests that
    /// pin `cache_dir` via [`Self::with_cache_dir`] and do not
    /// care about exercising GC should always prefer this
    /// constructor.
    pub fn new_without_gc(
        signal_bus: Arc<SignalBus>,
        contract: ExecutionContract,
        guard_manager: GuardManager,
    ) -> Self {
        Self {
            signal_bus,
            contract,
            guard_manager,
            degrade_events_count: AtomicU64::new(0),
            degrade_vetoed_by_cpu_count: AtomicU64::new(0),
            deep_degrade_events_count: AtomicU64::new(0),
            cache_dir: disk_tier::default_cache_dir(),
            lru_touch_order: Arc::new(TouchOrder::new()),
        }
    }

    /// M4.7.5.b — shared handle to the touch-ordered LRU. Cheap
    /// clone (atomic refcount); the underlying `TouchOrder`
    /// lives on the context for as long as the context does.
    /// Used by the selective-spill policy and by tests that
    /// need to inspect node completion order.
    pub fn lru_touch_order(&self) -> Arc<TouchOrder> {
        Arc::clone(&self.lru_touch_order)
    }

    /// M3-e.11.6: manually run the orphan-file GC on
    /// `self.cache_dir`. Returns `self` for builder chaining.
    /// Typical test usage:
    ///
    /// ```ignore
    /// let ctx = ReactiveExecutionContext::new_without_gc(...)
    ///     .with_cache_dir(test_dir)
    ///     .run_startup_gc();
    /// ```
    ///
    /// Equivalent to `ReactiveExecutionContext::new(...)` but
    /// lets the caller control both `cache_dir` and whether GC
    /// fires at all.
    pub fn run_startup_gc(self) -> Self {
        let _ = disk_tier::gc_orphan_disk_tensors(&self.cache_dir, 10);
        self
    }

    /// M3-e.11.5: builder-style override for the disk-spill cache
    /// directory. Intended primarily for tests that want a
    /// deterministic, unique location (usually a per-test temp
    /// dir generated via `uuid::Uuid`). Production code typically
    /// accepts the default from [`disk_tier::default_cache_dir`].
    ///
    /// Note: this does NOT re-run the startup GC on the new
    /// directory. To invoke GC on a custom `cache_dir`, chain
    /// `with_cache_dir(_).run_startup_gc()`.
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    /// Number of times `GuardAction::Degrade` was processed by this
    /// context, whether the resulting migration succeeded or failed.
    /// Useful for monitoring how often reaction is triggered in a
    /// given execution run.
    pub fn degrade_events_count(&self) -> u64 {
        self.degrade_events_count.load(Ordering::Relaxed)
    }

    /// M3-e.6: number of times a `Degrade` verdict was vetoed by the
    /// CPU-availability precondition because the CPU was saturated by
    /// external processes. These verdicts never reached the migration
    /// site; they are disjoint from `degrade_events_count`.
    pub fn degrade_vetoed_by_cpu_count(&self) -> u64 {
        self.degrade_vetoed_by_cpu_count.load(Ordering::Relaxed)
    }

    /// M3-e.11.5: number of times `GuardAction::DeepDegrade` was
    /// processed by this context — whether the verdict came
    /// directly from a guard or via promotion from `Degrade`,
    /// and whether the resulting `migrate_all_to_disk` succeeded
    /// or failed. Disjoint from `degrade_events_count`.
    pub fn deep_degrade_events_count(&self) -> u64 {
        self.deep_degrade_events_count.load(Ordering::Relaxed)
    }

    /// Record that a Degrade verdict was processed. Called from the
    /// graph's guard-handling site; not part of the public API because
    /// external code has no business incrementing the counter.
    pub(crate) fn record_degrade_event(&self) {
        self.degrade_events_count.fetch_add(1, Ordering::Relaxed);
    }

    /// M3-e.6: record that a Degrade verdict was vetoed by the CPU
    /// precondition. Called from the guard-handling site before the
    /// `Degrade` arm's migration body runs.
    pub(crate) fn record_degrade_veto_by_cpu(&self) {
        self.degrade_vetoed_by_cpu_count
            .fetch_add(1, Ordering::Relaxed);
    }

    /// M3-e.11.5: record that a DeepDegrade verdict was processed.
    /// Same "count the attempt" semantics as `record_degrade_event`.
    pub(crate) fn record_deep_degrade_event(&self) {
        self.deep_degrade_events_count
            .fetch_add(1, Ordering::Relaxed);
    }
}

/// Report produced by a successful `Graph::migrate_all_cuda_to_cpu`
/// call. Returned to `check_guard_before_node` so the guard-handling
/// site can log what the Degrade action actually did. Bytes freed are
/// an estimate based on `numel * size_of::<f32>()`; the actual VRAM
/// release depends on the refcount of the underlying `Arc<InnerGpuPtr>`
/// at drop time, which is not observable from the migration site.
#[derive(Debug, Clone)]
pub struct DegradeReport {
    pub tensors_migrated: usize,
    pub bytes_freed_estimate: usize,
}

impl fmt::Display for DegradeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mib = self.bytes_freed_estimate as f64 / (1024.0 * 1024.0);
        write!(
            f,
            "Degrade: migrated {} tensors, freed ~{:.2} MiB (estimate)",
            self.tensors_migrated, mib
        )
    }
}

/// Report produced by a migration primitive introduced in M3-e.11.4.
///
/// Unlike [`DegradeReport`], which models a successful shallow
/// Degrade at the reaction site, `MigrationReport` captures the
/// richer outcome space of the new cpu→disk and composite
/// cuda→cpu→disk primitives:
///
/// - `tensors_migrated`: tensors that changed storage variant on
///   this call (Cuda → Cpu or Cpu → Disk, depending on the method).
/// - `tensors_skipped`: tensors that were already in the target
///   tier (or in a tier the method does not touch, e.g. `Disk`
///   when calling `migrate_all_cpu_to_disk`).
/// - `failure`: `Some((index, err))` when a per-tensor migration
///   failed mid-iteration. The index refers to `self.nodes[idx]`.
///
/// Return policy in the owning methods:
/// - `failure.is_none()` → complete success. Every eligible
///   tensor reached the target tier.
/// - `tensors_migrated > 0 && failure.is_some()` → partial
///   progress. Some tensors moved, one failed; the report is
///   returned via `Ok(_)` so the caller can log the partial
///   completion and decide whether to retry or continue.
/// - `tensors_migrated == 0 && failure.is_some()` → total failure.
///   The method returns `Err` in this case rather than `Ok`
///   with an empty report, matching the contract consumers of
///   `migrate_all_cuda_to_cpu` already expect.
///
/// The split between `Ok(partial)` and `Err(total)` lets callers
/// use `?` confidently: a bubbled error guarantees no side
/// effects landed; a successful result guarantees at least one
/// tensor moved.
#[derive(Debug, Clone)]
pub struct MigrationReport {
    pub tensors_migrated: usize,
    pub tensors_skipped: usize,
    pub failure: Option<(usize, crate::tensor::StorageTransferError)>,
}

impl MigrationReport {
    pub fn new() -> Self {
        Self {
            tensors_migrated: 0,
            tensors_skipped: 0,
            failure: None,
        }
    }

    /// Every eligible tensor reached the target tier. Equivalent
    /// to `self.failure.is_none()`.
    pub fn is_complete(&self) -> bool {
        self.failure.is_none()
    }

    /// At least one tensor moved AND the run did not reach every
    /// eligible tensor — useful for logs that distinguish "I did
    /// some work and hit a snag" from "I did nothing and returned
    /// Ok because there was nothing to do".
    pub fn is_partial(&self) -> bool {
        self.tensors_migrated > 0 && self.failure.is_some()
    }
}

impl Default for MigrationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MigrationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.failure {
            None => write!(
                f,
                "Migration complete: {} migrated, {} skipped",
                self.tensors_migrated, self.tensors_skipped
            ),
            Some((idx, err)) => write!(
                f,
                "Migration partial: {} migrated, {} skipped, failed at node {} ({:?})",
                self.tensors_migrated, self.tensors_skipped, idx, err
            ),
        }
    }
}

/// M4.7.5.c — accumulating sibling of [`MigrationReport`] for the
/// per-tensor `migrate_selected_cpu_to_disk` primitive.
///
/// The legacy [`MigrationReport`] carries a single
/// `Option<(usize, StorageTransferError)>` because
/// `migrate_all_cpu_to_disk` stops at the first failure. The
/// selective primitive **continues past failures** so a single
/// transient I/O hiccup on one tensor does not prevent the
/// remaining N-1 tensors from being spilled — the M3-e reaction
/// loop fires under memory pressure, and giving up on the rest of
/// the eviction set when only one tensor failed defeats the
/// pressure-relief goal. Each per-tensor failure lands in the
/// `failures` vector with its node id; `tensors_migrated` counts
/// the successes.
///
/// Risk #5 of the M4.7.5 investigation: a partial-progress
/// contract built around a single `failure` field interacts badly
/// with a per-tensor caller. This report shape is the explicit
/// resolution.
#[derive(Debug, Clone)]
pub struct SelectiveMigrationReport {
    /// Number of tensors that successfully reached
    /// `TensorStorage::Disk` during this call.
    pub tensors_migrated: usize,
    /// Number of requested ids that were not migration candidates
    /// (already on Disk, on Cuda, or `node.output == None`).
    pub tensors_skipped: usize,
    /// Per-tensor failures encountered during the walk. Empty on a
    /// fully-successful run. Each entry is `(node_id, error)` and
    /// the order is the order in which the failures were observed
    /// (which is the order ids appear in the input slice).
    pub failures: Vec<(usize, crate::tensor::StorageTransferError)>,
}

impl SelectiveMigrationReport {
    pub fn new() -> Self {
        Self {
            tensors_migrated: 0,
            tensors_skipped: 0,
            failures: Vec::new(),
        }
    }

    /// Every requested tensor that was a migration candidate
    /// successfully reached disk. Skipped tensors do not count
    /// against this — a request for an already-Disk tensor is a
    /// "complete" outcome.
    pub fn is_complete(&self) -> bool {
        self.failures.is_empty()
    }

    /// At least one tensor moved AND at least one failed.
    pub fn is_partial(&self) -> bool {
        self.tensors_migrated > 0 && !self.failures.is_empty()
    }

    /// Total number of input ids that were processed
    /// (successes + skipped + failures).
    pub fn total_processed(&self) -> usize {
        self.tensors_migrated + self.tensors_skipped + self.failures.len()
    }
}

impl Default for SelectiveMigrationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SelectiveMigrationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.failures.is_empty() {
            write!(
                f,
                "Selective migration: {} migrated, {} skipped, 0 failures",
                self.tensors_migrated, self.tensors_skipped
            )
        } else {
            write!(
                f,
                "Selective migration: {} migrated, {} skipped, {} failures \
                 (first at node {})",
                self.tensors_migrated,
                self.tensors_skipped,
                self.failures.len(),
                self.failures[0].0,
            )
        }
    }
}

/// Reasons why a checked execution may abort before a given node runs.
///
/// Produced only when the graph has a `reactive_context` set. When the
/// context is absent, checked execution never returns these errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionAbortReason {
    /// The combined guard verdict for the current runtime state was
    /// `Abort`. The full `GuardConditions` snapshot is included for
    /// diagnostics.
    GuardAborted {
        at_node: usize,
        conditions: GuardConditions,
    },
    /// The `GuardManager` rejected the combined guard verdict as
    /// illegal given the current `ExecutionContract` (e.g. continuing
    /// under a pre-OOM signal while stability is required). The
    /// message comes from `GuardError::IllegalAction`.
    GuardEvaluationFailed {
        at_node: usize,
        message: String,
    },
}
