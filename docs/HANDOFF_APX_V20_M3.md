# Handoff — APX v20 M3 (at M3-e.5 close)

**Status at handoff**: M3-a, M3-c, M3-d, and M3-e.1–e.5 complete. M3-e
was expanded from an initial 5 sub-fases to a planned 11 after the
discovery that memory-only reaction is insufficient for Atenia's
"good citizen" design goal. M3-e.6 (CPU pressure with process
attribution) is the next sub-milestone.

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
| M3-e.1 | ✅ | `Graph::migrate_all_cuda_to_cpu` primitive + `DegradeReport`; Degrade arm in `check_guard_before_node` wired to call it. **First-pass reaction: memory pressure with implicit CPU availability assumption. Expansion pending in sub-milestones e.6–e.10.** |
| M3-e.2 | ✅ | `SimpleMemoryPressureGuard` with configurable threshold (default 0.65, strict `>`). Emits `Degrade` above threshold, `Continue` otherwise. |
| M3-e.3 | ✅ | End-to-end integration tests wiring `ReactiveExecutionContext` → `GuardManager` → `GuardAction::Degrade` → migration → continued execution. Positive/negative pair using a `DegradeIfFailuresGuard` fixture. |
| M3-e.4 | ✅ | `SignalBus` memory-pressure probe caching with `SIGNAL_BUS_CACHE_TTL = 100ms`; `probe_calls_count()` accessor for telemetry. Preserves freshness of injectable signals (`recent_failures`, `latency_spike`). |
| M3-e.5 | ✅ | Observability: enriched `[AMG Guard]` logs (timestamp + node_id + memory_pressure + probes_so_far) and `ReactiveExecutionContext::degrade_events_count()` counter. |
| M3-e.6 | ⏳ | **NEXT.** CPU pressure probe with process attribution (see "Remaining M3-e sub-milestones"). |
| M3-e.7 | ⏳ | GPU compute utilization with process attribution. |
| M3-e.8 | ⏳ | Foreground application detection. |
| M3-e.9 | ⏳ | Battery state monitoring. |
| M3-e.10 | ⏳ | Self-latency as primary decision signal. |
| M3-e.11 | 🕓 | Deferred — behavior modes, evaluated post-e.10 once real measurements inform mode boundaries. |

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
    into the `src/gpu/` layer. This pattern is used by the four
    files resolved in the M3-d.4 follow-up refactor (see the "Known
    debts" section for the history).

---

## How M3-e.1–e.5 landed (first-pass reaction)

The original v20 M3 idea — "when a guard signals `Degrade` due to VRAM
pressure, move Cuda-resident tensors back to RAM and keep executing" —
was implemented across five sub-fases. This section documents the
decisions taken during that first pass so future maintainers do not
have to re-derive them. **All decisions here scope to memory pressure
only**; the CPU-availability assumption they make implicit is the
reason the plan was expanded to e.6–e.11 (see the next section).

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

## Remaining M3-e sub-milestones (e.6–e.11)

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

### M3-e.11 — Behavior modes (evaluate post-e.10)

**Status**: DEFERRED — evaluate after e.6–e.10 are complete with real
measurements.

**Tentative scope**: Four discrete operating modes —
- `SoloMachine`: all resources available.
- `SharedMachine`: other apps active but not foreground.
- `UserActive`: foreground is a non-Atenia app.
- `Conservation`: low battery or thermal pressure (thermal pressure
  is handled by the OS — see "Metrics explicitly out of scope" — so
  this mode's trigger is likely battery-only).

Transitions between modes would be driven by the signals from
e.6–e.10. Discrete modes are easier to understand and debug than
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

2. **APX 8.4 `GPUMirror`** (`Tensor.gpu: Option<GPUMirror>`) is a
   metadata-only mirror introduced long before `TensorStorage::Cuda`.
   It is still wired through `sync_cpu` / `sync_gpu` on `Tensor` but
   its arms for the new `Cuda` storage variant are no-ops.
   Reconciling the two paths is pending.

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

**Immediate (next session)**:

- **M3-e.6**: CPU pressure with process attribution (PRIORITY —
  fixes the most dangerous blind spot of the first-pass reaction).

**Then in order**:

- M3-e.7: GPU compute utilization with process attribution.
- M3-e.8: Foreground application detection.
- M3-e.9: Battery state monitoring.
- M3-e.10: Self-latency as primary decision signal.
- M3-e.11: Behavior modes (only if e.6–e.10 measurements justify it).

**After M3-e is truly complete (post e.10)**:

- Pay M3 technical debts:
  - `src/apx4_3/gpu_executor.rs` refactor (debt #1, remaining item).
  - `FusedLinearActivationChain` duplicate handler consolidation
    (debt #4).
  - `FusedSelfAttention` fused backward (debt #5, optional
    optimization).
- Rename this document to `at M3-e close`.
- Resync `README.md` with the full M3 plan (intentionally left
  stale for e.6–e.10 during this sprint).

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
