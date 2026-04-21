# Handoff — APX v20 M3 (Real GPU Allocation for Tensor)

**Status at handoff**: M3-a completed and tested. M3-b eliminated
(absorbed into M3-a). M3-c, M3-d, M3-e pending. No code for later
sub-milestones has been written.

This document is a snapshot of where M3 stands today — not a full
history of how we got here. For the investigation trail see git
log and the older version of this file.

---

## Where we are in the project

| Milestone | Status | Summary |
|---|---|---|
| v20 M1 | ✅ Completed and committed | Conv2D and MaxPool2D ported to AMG with forward, backward, autograd tape integration. 16 tests passing. |
| v20 M2 | ✅ Completed and committed | SignalBus wired to AMG executor via `ReactiveExecutionContext`. `execute_checked` returns `Result<_, ExecutionAbortReason>`. 4 tests passing. |
| v20 M3-a | ✅ **Completed, tests passing, ready to commit** | `TensorStorage` enum + `Tensor` migrated to storage-based representation + 132 files migrated to the new API + 8 storage tests passing. |
| v20 M3-c | 🟡 Pending | Remove the `#[deprecated]` compatibility shims (`data`, `data_mut`, `num_elements`). Small, mechanical. |
| v20 M3-d | 🟡 Pending | Implement `TensorStorage::Cuda` with real VRAM allocation, `Drop`, host↔device transfers; wire `ensure_gpu`/`ensure_cpu` to do real work. |
| v20 M3-e | 🟡 Pending | Reaction strategy that moves real VRAM to RAM on guard `Degrade` — the original v20 M3 idea, now unblocked. |
| v20 M4–M8 | Not started | Real model loading, v17 retirement. |

---

## What M3-a delivered

### `TensorStorage` enum

Defined in `src/tensor/tensor.rs`, re-exported from `crate::tensor`:

```rust
pub enum TensorStorage {
    Cpu(Vec<f32>),
    // Cuda(TensorGPU) — added in M3-d.
}
```

Enum, not trait, for three reasons documented in the doc comment:
no `dyn` in the hot path, exhaustive `match` forces backend
handling at every consumer, and CUDA / ROCm / Metal APIs diverge
enough that a shared trait surface would be a useless lowest
common denominator.

### `Tensor` struct

The `pub data: Vec<f32>` field is gone. Replaced by:

```rust
pub storage: TensorStorage,
```

All other fields (`shape`, `device`, `dtype`, `layout`, `strides`,
`grad`, `gpu`, `persistence`, `op`) are unchanged.

`#[derive(Clone)]` is preserved. When M3-d lands and
`TensorStorage::Cuda` exists, `Clone` there will imply a VRAM→VRAM
transfer — we will document the cost but not change the trait.

### New accessor API

All `pub`, all on `impl Tensor`, all with doc comments:

| Method | Signature | Use |
|---|---|---|
| `new_cpu` | `fn new_cpu(shape: Vec<usize>, data: Vec<f32>) -> Tensor` | Canonical constructor for CPU tensors with known contents |
| `new_cpu_with_layout` | `fn new_cpu_with_layout(shape, data, device, dtype, layout) -> Tensor` | Constructor preserving non-default device/dtype/layout |
| `set_cpu_data` | `fn set_cpu_data(&mut self, Vec<f32>)` | Replace the storage in place; panics on length mismatch |
| `as_cpu_slice` | `fn as_cpu_slice(&self) -> &[f32]` | Immutable view; panics if storage is not CPU |
| `as_cpu_slice_mut` | `fn as_cpu_slice_mut(&mut self) -> &mut [f32]` | Mutable view; panics if storage is not CPU |
| `copy_to_cpu_vec` | `fn copy_to_cpu_vec(&self) -> Vec<f32>` | Owned copy; triggers D→H transfer in M3-d for Cuda |
| `ensure_cpu` | `fn ensure_cpu(&mut self) -> &mut Self` | No-op on CPU in M3-a; transfer in M3-d; returns `&mut Self` for chaining |
| `numel` | `fn numel(&self) -> usize` | Element count from `shape`; independent of backend |
| `storage` | `fn storage(&self) -> &TensorStorage` | Explicit match for backend-dispatch helpers |

### Compatibility shims (kept for M3-c)

Three `#[deprecated]` methods kept for parties that still rely on
the pre-0.20 field-access surface:

- `fn data(&self) -> &[f32]` — delegates to `as_cpu_slice`
- `fn data_mut(&mut self) -> &mut [f32]` — delegates to `as_cpu_slice_mut`
- `fn num_elements(&self) -> usize` — delegates to `numel`

No call site inside this repo uses them (verified by grep and by
zero `use of deprecated` warnings in both `cargo build --lib` and
`cargo build --tests`). They exist solely as stability guarantees
for downstream consumers and will be removed in M3-c.

### Migration scale

- **132 files touched**: 1 core rewrite (`src/tensor/tensor.rs`) +
  24 source files + 1 binary + 88 test files + cleanup commits.
- 5 sub-lotes for the migration, each with its own cargo build
  verification:
  - Lote 1: struct literals in `src/` (7 files, 10 sites).
  - Lote 2: field assignments in `src/` + tests (14 files).
  - Lote 3: all `src/amg/` reads via indexing/slice (4 files, ~153
    sites — `graph.rs` alone held ~140).
  - Lote 4: rest of `src/` (17 files + 1 bin, ~84 sites).
  - Lote 5: rest of `tests/` (88 files, ~310 sites).
- Final result: `cargo build --lib` and `cargo build --tests`
  both clean, 0 errors, 0 warnings.

### Tests

`tests/tensor_storage_test.rs`:

- `test_cpu_storage_roundtrip`
- `test_numel`
- `test_storage_accessor`
- `test_set_cpu_data`
- `test_ensure_cpu_noop`
- `test_deprecated_data_still_works` (with `#[allow(deprecated)]`)
- `test_deprecated_data_mut_still_works` (with `#[allow(deprecated)]`)
- `test_clone_preserves_storage`

All 8 pass. Run with `cargo test --test tensor_storage_test`.

### What did not change in M3-a

- `GPUMirror` and `GPUPersistenceInfo` fields on `Tensor` — still
  there. Their elimination is a separate cleanup; removing them
  touches `src/apx8/**` and associated tests.
- `v15`, `v16`, `v17`, `v19` and the SignalBus module — not
  modified (per standing milestone restrictions).
- `src/cuda/matmul.rs` still takes CPU-resident tensors through
  `as_cpu_slice()` / `as_cpu_slice_mut()`. A comment flags this as
  the point to revisit in M3-d (either change the signature to
  `&mut Tensor` and call `ensure_cpu` inside, or accept
  `TensorStorage::Cuda` directly and skip H→D).

---

## Decisions locked during M3 (do NOT re-ask)

1. **Enum, not trait, for storage**. Reasons in the doc comment
   of `TensorStorage` in `src/tensor/tensor.rs`. When ROCm / Metal
   land they become additional enum variants.

2. **Vendor-neutral from day 1**. The engine consumes
   `TensorStorage` and the accessor methods. No file outside
   `src/cuda/` or `src/gpu/` may `use crate::cuda::…` directly.
   Specific kernels (`cuda_matmul` et al.) stay CUDA-only; the
   vendor-neutrality is enforced at the Tensor API layer, not at
   kernel granularity (kernels remain vendor-specific; v22+ will
   introduce a higher-level dispatch if needed).

3. **Backward always on CPU in M3-d**. `TensorStorage::Cuda`
   appears in M3-d but backward closures will still `ensure_cpu()`
   on their inputs. Running backward on GPU when inputs are
   already there is a v21+ optimization.

4. **API breaking at the Tensor layer**. `tensor.data` (field) is
   gone. Call sites migrate to `as_cpu_slice()`, `copy_to_cpu_vec()`,
   `new_cpu()`, `set_cpu_data()`, etc. This was done in M3-a across
   132 files.

5. **`GPUMirror` and `GPUPersistenceInfo` stay for now**. They are
   APX 8.4/8.5 stubs and their elimination touches ~15 files. It
   is a separate, later commit — not part of M3.

6. **`new_cpu_with_layout` helper**. Introduced during Lote 1
   after seeing ≥4 sites with the "new_cpu + 4 overrides" pattern.
   Signature is fixed (shape, data, device, dtype, layout);
   callers that need non-default strides assign `t.strides = …`
   after construction.

7. **`TensorStorage` is re-exported** from `crate::tensor` (in
   `src/tensor/mod.rs`) — needed for the split-borrow trick in
   `src/optim/adamw.rs`, where reading `param.grad` and writing
   `param.storage` disjointly requires direct field pattern
   matching rather than a method call.

---

## M3-c (next sub-milestone) — what to do

**Scope**: remove the three `#[deprecated]` compatibility shims
(`fn data`, `fn data_mut`, `fn num_elements`) from
`src/tensor/tensor.rs`.

**Expected blast radius**: zero. At M3-a close, grep across the
full repo found zero callers of these methods (the migration was
exhaustive). The removal should be a 3-method deletion + one
doc-comment update + `cargo build --lib && cargo build --tests`
green.

**Risks**: downstream consumers outside this repo that were
relying on the shims. Since this is a pre-release engine
(APX v20 in a tagged `0.1.0` crate), we are free to break them
now. The stubs exist only to keep this repo's own migration
traceable across the intermediate commits.

**Deliverable**: single small commit removing the shims + updating
this handoff to drop M3-c from the pending list.

---

## M3-d (later sub-milestone) — what to design first

Before writing code for M3-d:

1. **Decide `cuda_matmul` and friends' signatures**. The comment
   at the top of `src/cuda/matmul.rs` lists three options:
   (a) change to `&mut Tensor` and call `ensure_cpu` internally;
   (b) accept `TensorStorage::Cuda` directly and skip H→D;
   (c) keep the current signature and require callers to
   `ensure_cpu` upstream. This is the first real design choice for
   M3-d.

2. **`TensorGPU` replacement**. The existing struct in
   `src/gpu/tensor/tensor_gpu.rs` is 2D-only, F32-only, no `Drop`,
   tightly coupled to `GpuMemoryEngine`. M3-d will either rebuild
   it or define a new type that `TensorStorage::Cuda` wraps.

3. **Error handling contract**. `ensure_gpu()` must decide: panic
   on VRAM allocation failure, or `Result<(), StorageError>`?
   Current `GpuMemoryEngine` returns `Result`; propagating it
   would force many call sites to handle errors they currently
   assume away.

4. **The 6 CUDA-leaked files outside `src/cuda/` and `src/gpu/`**
   (`apx4/gpu_kernels.rs`, `apx4_11/gpu_hooks.rs`,
   `apx4_3/gpu_executor.rs`, `apx4_3/gpu_utils.rs`,
   `apx4_5/batch_matmul_cuda.rs`,
   `src/bin/apx_4_10_fused_linear_silu_bench.rs`) all compile
   clean against the new Tensor API but still `use crate::cuda::…`
   directly. Whether to refactor them under an abstraction is
   **M3-d decision**, not M3-a.

---

## M3-e (the original v20 M3) — what it becomes

Original M3 was "reaction strategy: when guard returns `Degrade`,
move dormant tensors from GPU to CPU". It was redefined after the
M3.1 investigation because `Tensor.device` was a logical label.

After M3-d, that capability is finally achievable: a strategy can
`ensure_cpu()` a `Tensor` and the engine will free VRAM for real.

M3-e therefore becomes:

1. A new `StrategyAction::MoveToCpu(Vec<TensorRef>)` (or similar
   name) handled by the graph executor.
2. A reaction strategy that returns this action when a guard
   fires.
3. Integration tests that validate the VRAM residency change.

Nothing to do here until M3-d is in.

---

## Files and modules that would be heavily touched by M3-d

If helpful for planning M3-d:

- `src/tensor/tensor.rs` — add `Cuda(TensorGPU)` variant and
  implement `ensure_cpu` / `ensure_gpu` with real transfers. Plus
  `copy_to_cpu_vec` must grow a D→H path.
- `src/gpu/tensor/tensor_gpu.rs` — either replaced or extended
  (support >2D, non-F32 dtypes, `Drop` that frees VRAM).
- `src/cuda/{matmul, linear, batch_matmul, fused_linear_silu}.rs`
  — signatures revisited per the decision above.
- The 6 CUDA-leaked files listed above.

## Environment state at time of handoff

- Host: Windows 11, Intel i7-14650HX
- GPU: NVIDIA RTX 4070 Laptop + Intel UHD Graphics (integrated)
- CUDA Toolkit: v13.2
- MSVC BuildTools: 14.50.35717
- `build.rs` auto-detects CUDA and MSVC paths; respects
  `CUDA_PATH` and `MSVC_TOOLS_PATH` overrides.

---

## How to resume after compact

1. `git status` — confirm whether M3-a is committed or still staged.
2. Read this file (`docs/HANDOFF_APX_V20_M3.md`).
3. Read `ROADMAP.md` for the broader plan.
4. If the user wants to continue M3, the natural next step is
   **M3-c** (remove the deprecated shims). It is small, mechanical,
   and unblocks M3-d's clean-surface state.
5. If the user pivots: M3-a is self-contained and shippable; the
   rest of v20 can be paused here without losing anything. The
   deprecated shims keep the surface stable for any downstream
   consumer that was mid-migration.

---

## What NOT to do

- Do not start M3-d implementation before at least M3-c is
  closed. The deprecated shims complicate any future change to
  `Tensor`'s method surface.
- Do not modify `src/v15/`, `src/v16/`, `src/v17/`, `src/v19/`
  (the SignalBus module, `src/amm/signal_bus.rs`), or
  `src/apx4_3/`, `src/apx4_7/` outside the migrations already
  landed in M3-a. Their semantics are frozen for later milestones.
- Do not extend scope into real migration strategies (original M3
  idea). That is M3-e and depends on M3-d.
- Do not add new fields to `Tensor` unless strictly required.
- Do not panic on probe failures, mutex poisoning, or missing
  VRAM. Follow the fail-open / recover-defaults convention
  established in v18 and v19.
