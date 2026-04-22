# Handoff — APX v20 M3 (at M3-d close)

**Status at handoff**: M3-a, M3-c, and M3-d complete. M3-e (reaction
strategy that moves real VRAM back to RAM under guard `Degrade`) is
the next sub-milestone.

This handoff documents the architectural state at M3-d close so the
next person (or the same person after a long break) can resume on
M3-e without re-deriving context from scratch. For the narrative of
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
| M3-e | 🟡 | Not started. This handoff's main purpose is to brief it. |

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

---

## How M3-e fits

The original v20 M3 idea — "when a guard signals `Degrade` due to VRAM
pressure, move Cuda-resident tensors back to RAM and keep executing" —
is now straightforward to implement:

- `ensure_cpu()` on a Cuda tensor already performs a real D→H copy and
  swaps the storage variant. It returns `Result`, so callers can act
  on transfer failure.
- Dropping the original `Arc<InnerGpuPtr>` releases VRAM automatically
  when the last clone goes out of scope.
- Guard signals arrive through `ReactiveExecutionContext` (M2); the
  reaction strategy needs a place to hook in, inspect tensors in the
  graph, and call `ensure_cpu()` on the subset chosen by the policy.

What M3-e must decide before code:

- **Which tensors move?** Parameters, activations, or both? A simple
  first cut might migrate everything in `self.nodes[_].output` whose
  storage is `Cuda` when the guard fires, mirroring the backward
  pre-pass.
- **Which guard and policy drive the decision?** The `Degrade` path
  is agreed; the specific `GuardCondition` and `PolicySignal` that
  trigger the migration need to be wired.
- **Observability**: the migration should emit something the test
  harness and benchmarks can see (bytes moved, tensors touched,
  latency of the migration) so its effect is measurable.

M3-e does **not** need to add GPU backward, does **not** need to
unify the pool and the engine, and does **not** need to refactor the
CUDA-leaked files. Those stay on the debt list.

---

## Known debts carried into M3-e

These are documented so they are not confused with regressions.

1. **Five CUDA-leaked files** still import `crate::cuda::*` and live
   under `apx*_*` module paths: `src/apx4/gpu_kernels.rs`,
   `src/apx4_11/gpu_hooks.rs`, `src/apx4_3/gpu_utils.rs`,
   `src/apx4_5/batch_matmul_cuda.rs`,
   `src/bin/apx_4_10_fused_linear_silu_bench.rs`.
   `src/apx4_3/gpu_executor.rs` was touched partially during M3-d.4
   (segment execution still does not register tape; the fix was done
   at the intercept level in `execute_single_inner` instead).
   Refactoring these into `src/gpu/` is deferred.

2. **`tests/apx_4_18_self_attention_backward_test.rs::self_attention_backward_4_18_matches_naive`** is marked
   `#[ignore]`. The test predated the M3-d.4 tape-gap fix and
   tolerated missing MatMul grads via a zero-vector fallback, so its
   hardcoded expected values implicitly compared against zeros. With
   the fix, real grads are produced but the expectations were never
   analytically derived. Re-enabling the test requires recomputing
   dQ / dK / dV / dWq / dWk / dWv / dX against a reference framework.

3. **APX 8.4 `GPUMirror`** (`Tensor.gpu: Option<GPUMirror>`) is a
   metadata-only mirror introduced long before `TensorStorage::Cuda`.
   It is still wired through `sync_cpu` / `sync_gpu` on `Tensor` but
   its arms for the new `Cuda` storage variant are no-ops.
   Reconciling the two paths is pending.

4. **The C side of the 3 ops has minor quality debts**: silent
   allocation failures (`atenia_pool_alloc` returning null is only
   `printf`-reported, not propagated to Rust), ~210 lines of
   copy-paste across the 3 wrappers, sparse inline docs. The new
   `_device_ptrs` variants follow a stricter pattern (return `int`
   error codes, checked on the Rust side), but the legacy wrappers
   were not refactored.

---

## Files that M3-e will likely touch

A rough expected blast radius, for planning:

- `src/amm/` — Reaction strategy code, integration with existing
  `AmmForecaster` / `SignalBus`.
- `src/amg/graph.rs` — Hook for the migration to run in response to
  a guard signal during execution.
- `src/tensor/tensor.rs` — Possibly a helper to "migrate the
  reachable subgraph to CPU" (analogous to the backward pre-pass).
- New tests under `tests/` — end-to-end: build a graph with VRAM
  pressure injected, fire `Degrade`, assert tensors migrated and
  execution continues with correct numerics.

---

## How to resume

1. Read this file and the **Architectural decisions that now govern
   the code** section.
2. Skim [HANDOFF_APX_V20_M3_a.md](./HANDOFF_APX_V20_M3_a.md) only if
   you need historical context (e.g., why `TensorStorage` was chosen
   as an enum instead of a trait).
3. Check `cargo build --lib` and `cargo build --tests` compile clean;
   confirm `tests/cuda_ops_device_ptrs_dispatch_test` and
   `tests/backward_after_ensure_gpu_test` pass on a GPU-equipped
   machine before starting M3-e.
4. Start M3-e with design questions, not code — the three decisions
   listed under **How M3-e fits** above should be answered before
   implementation.
