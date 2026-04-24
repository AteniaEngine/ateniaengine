# Handoff — APX v20 M3 (at M3 close)

**Status at handoff**: M3 is cleanly closed. M3-a, M3-c, M3-d, and
M3-e.1–e.11 all complete. M3-e.12 (behavior modes) remains
explicitly deferred pending real-workload measurements. All nine
M3 technical debts have been resolved or formally deferred:

- **Correctness debts (closed)**: 8 — #1, #2, #3, #4, #6, #7, #8, #9.
- **Performance optimization (deferred to post-M4 with concrete
  re-evaluation criteria)**: 1 — #5 (`FusedSelfAttention` fused
  backward). See the "Deferred performance optimizations" block
  below.
- **Open correctness debts**: 0.

Next: M4 (real `ModelLoader` — safetensors support, weight loading,
graph population from checkpoints).

This handoff documents the architectural state at M3-e.5 close so the
next person (or the same person after a long break) can resume on
M3-e.6 without re-deriving context from scratch. For the narrative of
how M3-a landed first, see the historical snapshot in
[HANDOFF_APX_V20_M3_a.md](./HANDOFF_APX_V20_M3_a.md); the decisions
and invariants it records still hold.

---

## What is ready

| Sub-milestone | State | Summary |
|---------------|-------|---------|
| M3-a | ✅ | `TensorStorage` enum; `Tensor.data: Vec<f32>` → `Tensor.storage: TensorStorage`; canonical accessor API; 132-file migration. |
| M3-c | ✅ | Deprecated shims (`data()`, `data_mut()`, `num_elements()`) removed; the two tests that exercised them retired. |
| M3-d.1 | ✅ | `Arc`-refcounted VRAM ownership via `InnerGpuPtr`; thread-safe `gpu_engine()` singleton with primary-context retain and per-op `cuCtxSetCurrent`. |
| M3-d.2 | ✅ | `TensorStorage::Cuda(TensorGPU)` variant; `ensure_gpu` / `ensure_cpu` with real H↔D transfers; `GpuTransferError` enum for structured error reporting. |
| M3-d.3 | ✅ | `backward_checked` / `backward_sequential_checked` with `ensure_cpu` pre-pass on every `node.output`; 15 `ensure_cpu().expect(...)` guards inside 8 backward closures whose body produces intermediate tensors that are then consumed as host slices. |
| M3-d.4 | ✅ | GPU-plan tape-registration gap closed (intercept skips when `record_tape` is active); 3 CUDA ops (`cuda_linear`, `cuda_batch_matmul`, `cuda_fused_linear_silu`) migrated to `&Tensor` signatures with storage-based dispatch; new `launch_*_f32_device_ptrs` C variants that skip the H↔D roundtrip when every operand is already VRAM-resident. |
| M3-e.1 | ✅ | `Graph::migrate_all_cuda_to_cpu` primitive + `DegradeReport`; Degrade arm in `check_guard_before_node` wired to call it. **First-pass reaction: memory pressure with implicit CPU availability assumption. Expansion pending in sub-milestones e.6–e.11.** |
| M3-e.2 | ✅ | `SimpleMemoryPressureGuard` with configurable threshold (default 0.65, strict `>`). Emits `Degrade` above threshold, `Continue` otherwise. |
| M3-e.3 | ✅ | End-to-end integration tests wiring `ReactiveExecutionContext` → `GuardManager` → `GuardAction::Degrade` → migration → continued execution. Positive/negative pair using a `DegradeIfFailuresGuard` fixture. |
| M3-e.4 | ✅ | `SignalBus` memory-pressure probe caching with `SIGNAL_BUS_CACHE_TTL = 100ms`; `probe_calls_count()` accessor for telemetry. Preserves freshness of injectable signals (`recent_failures`, `latency_spike`). |
| M3-e.5 | ✅ | Observability: enriched `[AMG Guard]` logs (timestamp + node_id + memory_pressure + probes_so_far) and `ReactiveExecutionContext::degrade_events_count()` counter. |
| M3-e.6 | ✅ | CPU pressure probe with process attribution (`CpuProbe` stateful `System`; self vs. system-wide fields on `GuardConditions`). |
| M3-e.7 | ✅ | GPU compute utilization probe with process attribution. |
| M3-e.8 | ✅ | Foreground application detection. |
| M3-e.9 | ✅ | Battery state monitoring. |
| M3-e.10 | ✅ | Self-latency as primary decision signal (`NodeTimingRecorder`). |
| M3-e.11 | ✅ | `TensorStorage::Disk` tier + cascade migration (`Cuda → Cpu → Disk`), `GuardAction::DeepDegrade` with rank-based dominance, dual-pressure promotion at reaction site, automatic GC of orphan disk files at `ReactiveExecutionContext::new`. Eliminates `Abort` as the only fallback in dual-pressure scenarios. |
| M3-e.12 | 🕓 | Deferred — behavior modes, evaluated post-e.11 once real measurements inform mode boundaries. |

---

## Architectural decisions that now govern the code

These were locked as M3-d progressed. Treat them as invariants; future
milestones extend them rather than re-litigate.

1. **Single GPU-engine singleton** — `crate::gpu::gpu_engine()` returns
   `Option<&'static GpuMemoryEngine>`. The engine retains the device's
   **primary CUDA context** (`cuDevicePrimaryCtxRetain`) and every
   public operation calls `cuCtxSetCurrent(ctx)` on the current thread
   before issuing driver calls. This is what makes the singleton safe
   across Rust's test-thread workers.

2. **VRAM ownership by `Arc<InnerGpuPtr>`** — `TensorGPU` holds
   `inner: Arc<InnerGpuPtr>`. `InnerGpuPtr` has a `Drop` that calls
   `engine.free` when the last clone drops. `TensorGPU::clone` is an
   `Arc::clone` (cheap, shares VRAM); dropping one clone never frees
   the region as long as another clone exists.

3. **Storage enum, not trait object** — `TensorStorage` is a concrete
   enum with `Cpu(Vec<f32>)` and `Cuda(TensorGPU)`. Dispatch happens
   through exhaustive `match`, not through `dyn Storage`. The compiler
   forces every consumer to handle both variants explicitly.

4. **Explicit transfers, no implicit ones** — `Tensor::as_cpu_slice`
   (and its mutable twin) panic on `Cuda` storage with a message
   pointing the caller to `ensure_cpu()`. Every op in the project that
   needs host-side access obeys this rule; no op silently D→H copies
   a Cuda tensor as a convenience. `ensure_gpu` / `ensure_cpu` /
   `copy_to_cpu_vec` are the only surfaces where transfers happen, and
   they are explicit in the call site.

5. **Backward always on CPU, guaranteed by pre-pass** —
   `Graph::backward_checked` walks every `node.output` before any
   closure runs and calls `ensure_cpu()?`; it returns
   `Err(GpuTransferError)` on transfer failure. The unchecked
   `Graph::backward` wraps this and panics on `Err`. The 8 backward
   closures that build intermediate tensors inside their body (MatMul,
   FusedQKV, Linear full/raw, Transpose2D, Reshape, TransposeLastTwo,
   Softmax AVX2) also call `ensure_cpu().expect(...)` on each
   intermediate before consuming it as a slice. This is a defensive
   layer: today the ops those closures call are all CPU-resident, so
   the `ensure_cpu` is a no-op, but if an op is later rewired to
   produce `Cuda` storage the guard already catches it.

6. **`cuda_*_raw` private + `cuda_*` public dispatch** — The 3 CUDA
   ops expose a `pub fn cuda_*(..., &Tensor, ...)` that inspects the
   storage of every operand. If all operands are `Cuda`, the wrapper
   extracts device pointers via `cuda_device_ptr` /
   `cuda_device_ptr_mut` (helpers in `src/cuda/mod.rs`) and calls the
   C variant `launch_*_f32_device_ptrs` which skips alloc / memcpy /
   free entirely. Otherwise it falls through to a private
   `cuda_*_raw(&[f32], ..., &mut [f32], ...)` that contains the
   original host-path body. Mixed storage falls into the host path,
   which panics through `as_cpu_slice`, surfacing the inconsistency
   as a setup bug.

7. **GPU-plan intercept is forward-only** — The `exec_gpu_segment`
   intercept in `execute_single_inner` runs only when `record_tape` is
   false. In training mode, every node falls through to its regular
   `NodeType` handler, which registers the backward tape entry. The
   tape-registration gap that caused MatMul grads to silently be zero
   is closed this way; no new GPU-side backward was introduced.

8. **Pool and engine coexist with documented boundaries** — The
   legacy `apx4_12` 64 MB block pool remains the allocator for the
   temp buffers inside the classic `launch_*_f32` wrappers (host-path
   CUDA ops that alloc / copy / launch / copy / free in a single
   call). The engine singleton owns VRAM for long-lived
   `TensorStorage::Cuda` regions. Cross-free was not observed and is
   prevented by boundary: the `_device_ptrs` variants never alloc or
   free — VRAM ownership stays entirely on the Rust side via `Arc`.

9. **Vendor-neutrality at engine layer** — Code that lives on the
   engine layer (`src/amg/`, `src/apx*_*/`) must not import
   `crate::cuda::*` directly. The vendor-specific detail is contained
   inside `src/gpu/` and `src/cuda/`; APX modules reach GPU
   functionality through the abstractions in `src/gpu/` (wrappers,
   dispatch hooks, utilities). Three categories are explicit
   exceptions to the rule, because their reason to exist *is* to
   exercise vendor-specific code:
   - **Benchmark binaries** under `src/bin/*_bench.rs` that measure a
     specific kernel (e.g., `apx_4_10_fused_linear_silu_bench.rs`,
     `apx_4_11_hooks_bench.rs`, `apx_4_12_pool_bench.rs`) — they
     import the kernel directly by design.
   - **Kernel-level tests** under `tests/apx_4_*_*.rs` that validate
     a specific `cuda_*` function numerically against a CPU
     reference.
   - **Internal developer tooling** (if any lands in the future).
   Forcing abstraction on these would be cosmetic — the rule targets
   production code, not the tools that measure or validate it.

10. **Pass-through re-export as refactor path** — When a file that
    lives on the engine layer accumulates direct `crate::cuda::*`
    imports, the refactor pattern is: move the body to
    `src/gpu/...`, leave the original file as a one-line
    `pub use crate::gpu::...::*;` re-export. External callers keep
    working unchanged; the `crate::cuda::*` dependency is absorbed
    into the `src/gpu/` layer. This pattern is used by the five
    files that moved cuda-leaked code into `src/gpu/` — four in the
    M3-d.4 follow-up refactor and `src/apx4_3/gpu_executor.rs` in
    commit 80417af, which closed the vendor-neutrality invariant end
    to end (see the "Known debts" section for the history).

---

## How M3-e.1–e.5 landed (first-pass reaction)

The original v20 M3 idea — "when a guard signals `Degrade` due to VRAM
pressure, move Cuda-resident tensors back to RAM and keep executing" —
was implemented across five sub-fases. This section documents the
decisions taken during that first pass so future maintainers do not
have to re-derive them. **All decisions here scope to memory pressure
only**; the CPU-availability assumption they make implicit is the
reason the plan was expanded to e.6–e.12 (see the next section).

- **Which tensors move?** Decision: every `node.output` whose storage
  is `TensorStorage::Cuda` at the moment the guard fires, mirroring
  the backward pre-pass. Grads (`output.grad`) are left untouched —
  they are CPU-resident by construction under invariant #5. Parameters
  and activations are treated identically; no policy-driven subset
  selection in the first pass. Implemented in
  `Graph::migrate_all_cuda_to_cpu` (M3-e.1).

- **Which guard and policy drive the decision?** A concrete
  `SimpleMemoryPressureGuard` (M3-e.2) emits `Degrade` when
  `conditions.memory_pressure > DEGRADE_MEMORY_PRESSURE_THRESHOLD`
  (default 0.65, strict `>`). `GuardManager` priority resolves
  `Abort > Degrade > Continue`, so a higher-severity verdict from any
  other guard still wins. The Degrade arm in
  `Graph::check_guard_before_node` handles the verdict by calling
  `migrate_all_cuda_to_cpu` and continuing execution.

- **Observability**: the migration site logs
  `[AMG Guard][t_ms=...] Degrade triggered at node N: memory_pressure=X.XX, probes_so_far=K. Migrated M tensors, freed ~Y MiB.`
  on success (and the equivalent "FAILED" line on `Err`).
  `ReactiveExecutionContext::degrade_events_count()` exposes an
  atomic counter of processed Degrade verdicts (success or failure),
  readable from tests and monitoring code (M3-e.5).

- **Probe cost**: `SignalBus::collect_memory_pressure` costs ~40 ms
  per call (two subprocesses on Windows/Linux). M3-e.4 added a
  100 ms TTL cache keyed on `Instant`; fresh calls within the TTL
  hit cache and do not re-probe. `probe_calls_count()` exposes the
  cache-miss counter for regression tests.

- **What M3-e.1–e.5 did not do**: did not add GPU backward, did not
  unify the pool and the engine, did not refactor the CUDA-leaked
  files. Those stay on the debt list.

---

## Remaining M3-e sub-milestones (e.6–e.12)

The first-pass reaction (e.1–e.5) assumes CPU is always available to
absorb a migration and does not distinguish "Atenia is saturating my
GPU" from "another process is saturating my GPU". For Atenia's "good
citizen" design goal — coexisting with normal user activity — both
gaps matter. The plan below closes them before M3 is declared
complete.

### M3-e.6 — CPU pressure with process attribution

**Status**: PENDING (next sub-milestone).

**Scope**: Add a `CpuPressureProbe` that reports both system-wide CPU
utilization and Atenia's own process contribution. Extend
`GuardConditions` with `cpu_pressure_total` and `cpu_pressure_self`
fields. Modify the Degrade arm in `check_guard_before_node` to skip
migration when system CPU is high but Atenia's contribution is low
(other processes are saturating the CPU; migrating would make things
worse). Platform-specific implementation required: `getrusage` on
Unix, Windows Process API on Windows.

**Rationale**: Fixes the most dangerous blind spot of the M3-e current
design — preventing `Degrade` from worsening situations where the CPU
is saturated by external processes.

### M3-e.7 — GPU compute utilization with process attribution

**Status**: PENDING.

**Scope**: Extend the memory-pressure probe to also query compute
utilization via `nvidia-smi --query-compute-apps`. Distinguish
Atenia's GPU usage from other processes'. Add a new `GuardAction`
variant (tentatively `Throttle` or `DeferWork`) for the case where the
GPU is saturated by an external process while VRAM still has capacity:
`Degrade` alone is not sufficient, because the problem is compute
time, not memory.

**Rationale**: Covers the scenario where a user is gaming or running
graphics apps while Atenia trains in background. Atenia should yield
GPU time, not try to migrate tensors.

### M3-e.8 — Foreground application detection

**Status**: PENDING.

**Scope**: Use OS APIs to detect whether a non-Atenia application is
in the foreground (the user is actively interacting with something
else). Binary signal: `foreground_is_atenia: bool`. When false, Atenia
operates in conservative mode — reduced threads, lower probe
frequency, smaller batches. Platform APIs: `GetForegroundWindow` on
Windows, X11/Wayland on Linux, `NSWorkspace` on macOS.

**Rationale**: Simple and effective "good citizen" heuristic — when
the user is working with something else, Atenia should step back
noticeably.

### M3-e.9 — Battery state monitoring

**Status**: PENDING.

**Scope**: Detect battery level and plugged status via OS API. Add a
conservation mode triggered when battery is below threshold (e.g.,
20%) and not plugged in. In conservation mode: pause non-critical
work, reduce processing frequency, extend caching TTL. No-op on
desktop systems without a battery.

**Rationale**: A laptop user on battery does not want Atenia draining
power to finish a training run faster. They want the laptop to last
through their work.

### M3-e.10 — Self-latency as primary decision signal

**Status**: PENDING.

**Scope**: Leverage the existing `latency_monitor` (currently only
used for spike detection in `GuardConditions`) as a primary reaction
signal. If Atenia's own processing latency degrades more than X% from
baseline, something external is applying pressure regardless of what
other probes report. Add percentage-threshold logic and wire it into
the decision path.

**Rationale**: Zero additional probe cost (already measuring). The
most reliable signal because it captures "am I actually running slow?"
directly, bypassing any probe misreadings. Arguably the single most
honest signal in the set because it measures impact, not cause.

### M3-e.11 — `TensorStorage::Disk` tier + cascade migration

**Status**: PENDING (planned after e.6–e.10 complete). **CRITICAL**:
without this, the only safe outcome in dual-pressure scenarios (VRAM
and RAM both saturated) is `Abort`.

**Scope**: Add `TensorStorage::Disk(DiskTensorHandle)` variant to the
`TensorStorage` enum. Implement serialize / deserialize for RAM → Disk
migration. Extend the migration primitive from
`migrate_all_cuda_to_cpu` to `migrate_all_to_next_tier` with cascade
logic (`Cuda → Cpu`, `Cpu → Disk`). Add a guard variant (tentatively
`DeepDegrade`) that triggers when dual-pressure is detected — both
VRAM and RAM above threshold. Mechanism to bring tensors back from
disk when resources free up. Cleanup of temporary disk files on
`Drop`.

**Rationale**: The current 2-tier design (VRAM + RAM) has no safe
migration target in dual-pressure scenarios, forcing `Abort` as the
only outcome. Adding disk as a third tier eliminates this case:
workloads can continue with reduced throughput rather than fail.
Aligns with Atenia's "good citizen" design principle — slow but
functional is infinitely better than crashed.

**Technical notes**: First implementation uses a simple
serialize-to-file approach (Option 1 from the design discussion)
rather than `mmap` or chunked paging. Simpler semantics, predictable
behavior; can be refined later with measurement once real workloads
exercise the cascade.

### M3-e.12 — Behavior modes (evaluate post-e.11)

**Status**: DEFERRED — evaluate after e.6–e.11 are complete with real
measurements.

**Tentative scope**: Four discrete operating modes —
- `SoloMachine`: all resources available.
- `SharedMachine`: other apps active but not foreground.
- `UserActive`: foreground is a non-Atenia app.
- `Conservation`: low battery or thermal pressure (thermal pressure
  is handled by the OS — see "Metrics explicitly out of scope" — so
  this mode's trigger is likely battery-only).

Transitions between modes would be driven by the signals from
e.6–e.11. Discrete modes are easier to understand and debug than
continuous optimization.

**Rationale for deferral**: Designing the mode structure before
observing how the signals behave in practice would be premature
optimization. Real workload measurements should inform the mode
boundaries.

---

## Metrics explicitly out of scope (rationale documented)

The following signals were considered and explicitly not planned.
Recording the rationale here prevents the same debate recurring when
the next maintainer looks at the gap.

- **Thermal monitoring**: The OS already handles thermal throttling
  at the hardware level. Atenia adding its own thermal response would
  duplicate what the OS does and risk conflicting with it.

- **PCIe bandwidth**: Rarely the real bottleneck in typical workloads.
  Adding instrumentation without evidence of impact is premature
  optimization.

- **Predictive models (ML-based reaction)**: Simple reactive systems
  with good metrics outperform complex predictors in most cases.
  Adding ML to the reaction strategy is over-engineering before the
  reactive baseline has been validated in practice.

- **Remote telemetry**: Violates Atenia's "total transparency"
  principle. User privacy takes precedence over operational
  observability gains.

---

## Known debts carried forward

These are documented so they are not confused with regressions.

1. **CUDA-leaked files (follow-up refactor after M3-d.4)** — The
   original list flagged five engine-layer files that imported
   `crate::cuda::*` directly. Post-M3-d close, a dedicated refactor
   reclassified and resolved them:
   - **Resolved via pass-through re-export** (four files; body moved
     to `src/gpu/`, original kept as `pub use`):
     `src/apx4/gpu_kernels.rs` → `src/gpu/ops/matmul_wrapper.rs`;
     `src/apx4_3/gpu_utils.rs` → `src/gpu/utils.rs`;
     `src/apx4_5/batch_matmul_cuda.rs` →
     `src/gpu/ops/batch_matmul_dispatch.rs`;
     `src/apx4_11/gpu_hooks.rs` → `src/gpu/dispatch/hooks.rs`.
   - **Reclassified as legitimate exception** (one file):
     `src/bin/apx_4_10_fused_linear_silu_bench.rs` is a benchmark
     binary; its purpose is to measure a specific CUDA kernel and
     the direct import is allowed under invariant #9.
   - **Separate, larger debt still open**: `src/apx4_3/gpu_executor.rs`
     keeps three `crate::cuda::*` imports (`cuda_matmul`,
     `cuda_linear`, `cuda_fused_linear_silu`). It was touched
     partially during M3-d.4 (the tape-registration gap was fixed at
     the intercept level in `execute_single_inner` instead of
     inside the segment executor itself). Refactoring it requires
     rethinking `exec_gpu_segment` end to end and is deferred to
     its own milestone.

2. **APX 8.4 `GPUMirror`** (`Tensor.gpu: Option<GPUMirror>`) was a
   metadata-only mirror introduced long before `TensorStorage::Cuda`.
   It was still wired through `sync_cpu` / `sync_gpu` on `Tensor`
   but its arms for the `Cuda` and `Disk` storage variants were
   no-ops. Reconciling the two paths was pending.

   **Resolved in commit b95e8bc**: `GPUMirror` and `MirrorState`
   (APX 8.4) plus the unused `GPUPersistenceInfo` eviction heuristic
   (APX 8.5) were removed entirely. `Tensor` no longer carries
   `gpu: Option<GPUMirror>` or `persistence: Option<GPUPersistenceInfo>`
   fields; the 8 mirror/persistence methods (`ensure_gpu_mirror`,
   `mark_gpu_dirty`, `mark_cpu_dirty`, `sync_cpu`, `sync_gpu`,
   `enable_gpu_persistence`, `note_gpu_use`, `maybe_drop_gpu`) are
   gone. `src/apx8/mirror.rs` and `src/apx8/persistent.rs` were
   deleted; `gpu_vec_add` in `src/apx8/gpu_kernels.rs` no longer
   touches mirror metadata. Tests `apx_8_4_mirror_test.rs` and
   `apx_8_5_persistent_cache_test.rs` were removed; `apx_8_6` lost
   the single `a.sync_cpu()` line that depended on the removed
   method. `TensorStorage::Cuda` (M3-d) covers the real VRAM path.
   Real eviction over `TensorStorage::Cuda` is deferred to post-M4
   pending real-workload measurement.

3. **The C side of the 3 ops has minor quality debts**: silent
   allocation failures (`atenia_pool_alloc` returning null is only
   `printf`-reported, not propagated to Rust), ~210 lines of
   copy-paste across the 3 wrappers, sparse inline docs. The new
   `_device_ptrs` variants follow a stricter pattern (return `int`
   error codes, checked on the Rust side), but the legacy wrappers
   were not refactored.

4. **`FusedLinearActivationChain` duplicate handler** — The op has
   two copies of its forward handler in `src/amg/graph.rs`: a
   pre-match shortcut (accepts 3/4/5 inputs) and a match-arm
   fallback (accepts only 4/5). Neither registers a backward
   BackOp, so the forward-only fusion is effectively shared between
   the two. The duplication is a code smell inherited from earlier
   milestones; consolidating the handlers (and deciding whether to
   add a real fused BackOp) is open work.

5. **`FusedSelfAttention` fused backward** (optional optimization) —
   The op runs fused on forward but falls back to unfused backward.
   A fused backward would reduce the number of backward closures
   that need the `ensure_cpu` pre-pass guard. Low priority until a
   real model shows it in the profile.

6. **Incomplete storage tier hierarchy (dual-pressure blind spot)** —
   The current storage tier hierarchy (VRAM + RAM) is incomplete.
   Dual-pressure scenarios where both tiers are saturated currently
   resolve to `Abort` via `GuardAction::Abort`. This is a known
   limitation, not a final design. The M3-e.11 sub-milestone
   introduces a third tier (Disk) and cascade migration logic that
   eliminates `Abort` as the only option in these cases. Disk tier
   represents continuity at reduced throughput — aligned with
   Atenia's good-citizen design principle that favors slow correct
   execution over fast failure.

7. **`src/apx7/dynamic_load.rs` — silently broken CPU load sampling** —
   The `sample_system_load` function creates a fresh `sysinfo::System`
   instance per call and reads `cpu_usage()` from it. Because sysinfo
   calculates CPU usage as a delta between two refresh calls, a fresh
   `System` always reports 0.0% CPU usage on its first read. The
   function has therefore been returning a constant near-zero CPU
   load since its introduction. The downstream strategy selector in
   `choose_strategy` uses this value as a heuristic input, meaning
   its "high CPU" branch has never been exercised in practice. This
   was discovered during M3-e.6 investigation when the same pattern
   was identified as incorrect for the new `CpuProbe`, which
   explicitly maintains a stateful `System` instance across calls to
   avoid this bug.

   Resolution requires either (a) making `sample_system_load`
   stateful like the new `CpuProbe`, or (b) refactoring
   `choose_strategy` to use the new `CpuProbe` directly instead of
   maintaining a separate code path. Option (b) is architecturally
   cleaner but requires understanding the strategy selector's
   dependencies. Tracked as separate cleanup; does not block other
   work.

8. **`FusedLinearActivationChain` — backward pass does not register
   a `BackOp`** — The helper `exec_fused_linear_activation_chain`
   (introduced in commit 8c328bd as part of debt #4 cleanup)
   computes forward correctly but does not register a `BackOp`
   for gradient tracking. Consequences: backward pass through a
   `FusedLinearActivationChain` node produces zero or incorrect
   gradients for the fused inputs.

   Existing tests that exercise backward on this node
   (`apx_2_5_fused_kernels_test`) pass because they validate
   seq-vs-par equivalence (both use the same broken fused
   backward), not correctness against a non-fused reference.
   Any production use of this fused node for training would
   produce silently wrong gradients.

   This was known conceptually as a "fused backward optimization"
   debt, but the actual impact — gradients are wrong, not just
   non-optimized — became clear during investigation of debt #5.
   Tracked separately from debt #5 (`FusedSelfAttention` fused
   backward) because the contexts differ: #5 is optimization-only,
   #8 is correctness.

   Resolution requires implementing the analytical backward for
   the chain: grad for `w2`, `b2` (if present), `w1`, `b1` (if
   present), and `x`, given grad flowing in. Complex but bounded
   scope.

   **Resolved in commit b623ca2**: `exec_fused_linear_activation_chain`
   now accepts `record_tape: bool` and registers an analytical
   `BackOp` that captures the required intermediates (`y1`, per-
   activation inputs, `a_last`) by move into the closure.
   Activation derivatives for ReLU / SiLU / GELU are inlined; the
   reverse-iteration loop handles `Vec<ActType>` of any length.
   `ensure_cpu()` is called before each capture so the closure-
   owned tensors stay on CPU regardless of subsequent
   Degrade/DeepDegrade migrations. Correctness locked by
   `tests/m3_debt_fused_chain_backward_test.rs` (8 cases: 3/4-b1/
   4-b2/5-input × SiLU vs non-fused reference, Vec<ActType>=2,
   ReLU/GELU non-zero finite smoke tests, seq-vs-par parity on
   the fused chain).

9. **`exec_gpu_add` / `exec_gpu_mul` misleading naming** —
   **Resolved via option (c) in commit c72783d**: both
   methods eliminated along with the `NodeType::Add` and
   `NodeType::Mul` match arms in `exec_gpu_segment`.
   Investigation confirmed the arms were statically
   unreachable: `GpuPlan::build` consults
   `is_cuda_available_for`, which only accepts
   `NodeType::MatMul`, so the planner never produces a segment
   containing `Add` or `Mul`. The stubs executed `a.add(b)` /
   `a.mul(b)` on CPU, misrepresenting their behavior through
   the `exec_gpu_` prefix. Option (a) rename was rejected as
   cosmetic (moves the debt without resolving it); option (b)
   real GPU dispatch was out of scope (requires CUDA kernels,
   `launch_*_f32_device_ptrs` variants, storage-aware
   dispatch, tests — a dedicated sub-milestone). Removal is
   behavior-preserving: the `_ => {}` arm in
   `exec_gpu_segment` already covered every `NodeType` the
   planner actually produces in these code paths.

---

## Files actually touched in M3-e.1–e.5

Recorded here as the factual blast radius of the first pass, for
reference:

- `src/amg/graph.rs` — `Graph::migrate_all_cuda_to_cpu` primitive,
  Degrade arm in `check_guard_before_node`, `reactive_context()`
  accessor, enriched observability logs.
- `src/amg/reactive.rs` — `DegradeReport` struct, atomic
  `degrade_events_count` counter on `ReactiveExecutionContext`.
- `src/amm/signal_bus.rs` — `SIGNAL_BUS_CACHE_TTL` constant,
  `memory_pressure_cache` with `Mutex<Option<(f32, Instant)>>`,
  `probe_calls_count()` accessor.
- `src/v16/guards/simple_memory_pressure_guard.rs` (new) —
  `SimpleMemoryPressureGuard` with `DEGRADE_MEMORY_PRESSURE_THRESHOLD`
  constant.
- `src/v16/guards/mod.rs` — `pub mod simple_memory_pressure_guard`.
- New test files under `tests/`:
  - `m3_e_degrade_test.rs` (migration primitive + e.3 integration).
  - `m3_e_simple_memory_pressure_guard_test.rs` (guard unit tests).
  - `signal_bus_caching_test.rs` (TTL cache behavior).
  - `m3_e_observability_test.rs` (degrade-events counter).

---

## Next steps

**M3-e is complete**. Sub-milestones e.1–e.11 all closed. e.12
(behavior modes) remains explicitly deferred, pending real workload
measurements to inform mode boundaries.

**Remaining M3 correctness debts**: none. M3 is cleanly closed
from a correctness perspective. The one remaining item (Debt #5)
is a pure performance optimization with no correctness gap; it is
documented separately below with re-evaluation criteria tied to
M4 measurements.

**Closed during M3-e and follow-up cleanup** (for reference):

- **Debt #1** — Vendor-neutrality invariant fully closed: last
  cuda-leaked file (`src/apx4_3/gpu_executor.rs`) moved to
  `src/gpu/dispatch/executor.rs` in commit 80417af.
- **Debt #4** — `FusedLinearActivationChain` duplicate handlers
  consolidated into `exec_fused_linear_activation_chain`; validator
  relaxed to accept 3/4/5 inputs (commit 8c328bd).
- **Debt #6** — Storage tier hierarchy completed by M3-e.11 (Disk
  tier + cascade + dual-pressure handling).
- **Debt #7** — `src/apx7/dynamic_load.rs` stateful CPU sampling
  (commit 2a83ee8).
- **Debt #8** — `FusedLinearActivationChain` backward BackOp
  registered with analytical gradients for x/W1/b1/W2/b2 across
  all input layouts and any `Vec<ActType>` length (commit b623ca2).
  Closes the highest-impact correctness gap of the M3 debt list —
  APX 4.9 fused patterns can now be trained correctly.
- **Debt #2** — APX 8.4 `GPUMirror` and APX 8.5 `GPUPersistenceInfo`
  removed entirely (commit b95e8bc). Metadata-only stubs that
  pre-dated `TensorStorage::Cuda` (M3-d) and became redundant;
  eliminated cleanly because they had no `Drop`, no device pointer,
  and zero production consumers outside `tensor.rs` itself. Real
  eviction over `TensorStorage::Cuda` deferred to post-M4.
- **Debt #9** — `exec_gpu_add` / `exec_gpu_mul` CPU-fallback stubs
  removed entirely (commit c72783d). Dead code by construction:
  `GpuPlan::build`'s filter `is_cuda_available_for` only accepts
  `NodeType::MatMul`, so the corresponding match arms in
  `exec_gpu_segment` were statically unreachable. Option (c)
  (removal) was selected over (a) rename or (b) real GPU dispatch;
  zero observable behavior change since the `_ => {}` arm already
  covered every `NodeType` the planner actually emits.
- **Debt #3** — C-side quality on the 3 CUDA ops. Resolved across
  two commits:
  - **Fase 3.1** (commit b6d507f): eliminated the dead
    `launch_matmul_f32` legacy host wrapper (~70 LOC of C,
    declared in `matmul_kernel.h` and defined in
    `matmul_kernel.cu` but never linked from Rust — `matmul.rs`
    uses `matmul_f32_launch_device` directly). Deleted the
    orphan header. Added a null-check to the 3 `pool_alloc`
    calls in `src/cuda/matmul.rs` that previously let a null
    device pointer flow into `cudaMemcpy` and surface as a
    cryptic `cudaErrorInvalidDevicePointer`.
  - **Fase 3.2** (commit 0313b73): introduced
    `src/cuda/pool_helpers.rs` with
    `with_pooled_device_buffers`, a single Rust helper that
    centralizes alloc / H↔D memcpy / kernel launch / free for
    the CPU-path of all four CUDA ops. Eliminated the three
    remaining legacy C host wrappers (`launch_linear_f32`,
    `launch_batch_matmul_f32`, `launch_fused_linear_silu_f32`
    — ~210 LOC of C removed combined). The Rust-side `cuda_*_raw`
    functions and `cuda_matmul` now all invoke the same
    `_device_ptrs` launcher through the helper, unifying the
    CPU-path with the all-Cuda path on the same kernel entry.
    Added `StorageTransferError::PoolExhausted { size_bytes }`
    to propagate alloc failures with the actual root cause
    instead of a masked CUDA-driver error. Net: -166 LOC across
    10 files, single well-documented helper in place of four
    partial copies of the same pattern.
  - **Fase 3.3** (dedicated pool-exhaustion integration test)
    deferred as follow-up. The current APX 4.12 pool
    auto-grows by calling `cuda_malloc` directly when its block
    list is empty (`GpuMemoryPool::alloc` in
    `src/apx4_12/gpu_memory_pool.rs:19-29`), so a deterministic
    test for `PoolExhausted` requires one of: a CUDA-absent
    machine (inverts the skip pattern other tests use), real
    multi-GB VRAM exhaustion (non-portable, flaky), or a pool
    refactor with a hard ceiling plus dependency injection for
    the alloc. None fit a debt cleanup scope. The variant is
    validated by the type system plus the four existing tests
    that exercise the helper's happy path end-to-end
    (apx_4_2_matmul_test, apx_4_4_linear_test,
    apx_4_5_batch_matmul_test, apx_4_10_fused_linear_silu_gpu_test).

## Deferred performance optimizations (not M3-scope)

### #5 — `FusedSelfAttention` fused backward

**Status**: Deferred. Not a correctness issue.

**Current state**: `FusedSelfAttention` runs fused in forward (and
is active by default under `ATENIA_APX_MODE=4.19`) but uses the
naive, non-fused backward through the individual BackOps of the
underlying subgraph (3 Linears, Transpose, MatMul, Softmax,
MatMul). Numerically correct — validated bit-exact against the
non-fused reference by `tests/apx_4_18_self_attention_backward_test.rs`
with `1e-5` tolerance across `dX`, `dWq`, `dWk`, `dWv`. The
intermediates needed for a fused backward (`q`, `k`, `v`, `att`,
`out`) are already cached during forward in
`self.fused_outputs[id] = FusedOutput::SelfAttention { .. }` —
so the memory cost of caching is already paid.

**Potential benefit**: ~1-4% end-to-end speedup in training,
estimated without measurement. Reduces `ensure_cpu` guard
invocations in backward closures from ~7 to ~1-2 and removes
several intermediate-gradient allocations that exist only to
bridge adjacent BackOps.

**Implementation risk**: Requires either (a) skipping BackOp
registration on individual nodes of the attention pattern —
exactly the mechanism that caused a historic silent-zero-grad
bug on this op (see the history section in the docstring of
`tests/apx_4_18_self_attention_backward_test.rs`), or (b) fused
and individual BackOps coexisting, which is incorrect because it
double-counts gradients. The current equivalence test catches
numerical drift but would NOT detect "which BackOp ran" — a
silent fallback to the naive path would pass the test, so option
(a) without additional observability is brittle.

**Re-evaluation criteria**: Re-examine when M4 brings a
transformer (≥100M params) executing forward+backward
end-to-end AND the profile shows attention backward as >5% of
total training time. Until **both** conditions are met,
implementation is speculative optimization.

**Pre-requisites if resumed**:
- Deterministic benchmark of standalone attention block backward
  (measurable baseline → evaluable delta).
- Observability test verifying which BackOp actually runs
  (counter, mock, or instrumentation flag) — not just numerical
  equivalence.
- Reference transformer model (2-4 layers) with full
  forward+backward pass executing under the M4 runtime.

With these pre-requisites in place, estimated work drops from
**5-8h (structural risk)** to **3-5h (safe regression prevention)**.

**Rationale for deferral over implementation**:
1. Not a correctness gap — unlike the 8 closed debts, the forward
   fusion preserves the original subgraph structure and backward
   traverses real, already-registered BackOps. The existing
   `apx_4_18` test proves bit-exact equivalence.
2. Without a real transformer running end-to-end (M4+), there is
   no profile to target. Any estimate of the 1-4% speedup is
   speculation.
3. The option-(a) implementation path re-enters exactly the
   territory that produced the historic silent-zero-grad bug.
   That bug was caught by the equivalence test only because an
   alternative path (naive mode) was available for comparison;
   a subtler regression in the fused backward itself would not
   be caught by the current suite.
4. M3 was correctness foundations. Optimizations without
   measurement do not fit that theme and spend complexity budget
   in `src/amg/graph.rs` (already 4k+ LOC) for uncertain gain.

**Admin**:

- Rename this document to `at M3-e close`.
- Resync `README.md` with the full M3 plan (intentionally left stale
  for e.6–e.11 during this sprint).

**Then**:

- **M4**: Real `ModelLoader` — `safetensors` support, weight
  loading, graph population from checkpoints.

---

## How to resume

1. Read this file and the **Architectural decisions that now govern
   the code** section.
2. Skim [HANDOFF_APX_V20_M3_a.md](./HANDOFF_APX_V20_M3_a.md) only if
   you need historical context (e.g., why `TensorStorage` was chosen
   as an enum instead of a trait).
3. Check `cargo build --lib` and `cargo build --tests` compile clean;
   on a GPU-equipped machine, confirm the M3-e regression suite
   passes before starting M3-e.6:
   - `tests/m3_e_degrade_test`
   - `tests/m3_e_simple_memory_pressure_guard_test`
   - `tests/signal_bus_caching_test`
   - `tests/m3_e_observability_test`
   - `tests/cuda_ops_device_ptrs_dispatch_test`
   - `tests/backward_after_ensure_gpu_test`
4. Start M3-e.6 with design questions, not code — the CPU probe's
   process-attribution API differs between Windows and Unix; the
   signal must be defined (both total and self fields on
   `GuardConditions`) before the Degrade arm's new skip logic can
   be written.
