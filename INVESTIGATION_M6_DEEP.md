# M6 Investigation — Read-Only Deep Dive

**Status**: read-only analysis over `origin/main = 105f966` (≡ M5.f.a tree
at commit `8b8253d` plus `INVESTIGATION_M6.md`). No code written, no fixes
proposed — surfaces and gates identified for the next M6 attempt.

This document is the technical companion to `INVESTIGATION_M6.md`. The
former records what was tried and reverted; this one records what the
code actually does today and what the surgical surfaces look like.

---

## 1. Current GPU Dispatch Path (M5.f.a)

### Two parallel "GPU surfaces" coexist on main today

**Surface A — `try_gpu_matmul` (M4.7.3 residency-aware)**

Entry: `src/amg/graph.rs:3282-3284` inside the `NodeType::MatMul` arm:

```
if gpu_hooks::gpu_can_run_matmul(m, k, n)
   && gpu_hooks::try_gpu_matmul(&a, &b, &mut out) { ... }
```

- `gpu_can_run_matmul` lives at `src/gpu/dispatch/hooks.rs:107-132`.
  Gates: `ops > 256`, `cuda_available()` true, **and
  `max_per_alloc <= DEFAULT_BLOCK_SIZE = 64 MiB`**
  (`src/apx4_12/mod.rs:16`).
- `try_gpu_matmul` at `src/gpu/dispatch/hooks.rs:151-238`: shape/dtype
  checks, then dispatches three sub-paths:
  - `all_cuda` → `cuda_matmul_inplace` (residency,
    `src/cuda/matmul.rs:82-139`) → device-pointer launcher
    `matmul_f32_launch_device` directly on VRAM.
  - `all_cpu` + shape gate ok → `cuda_matmul` (CPU-roundtrip,
    `src/cuda/matmul.rs:31-59`) → `with_pooled_device_buffers`
    (`src/cuda/pool_helpers.rs:110-183`): pool_alloc 64 MiB blocks
    for A/B/Out, `cudaMemcpy` H→D, kernel, `cudaMemcpy` D→H,
    `pool_free`.
  - mixed → returns false.

**Surface B — `dispatch_matmul_gpu` (legacy apx4 path)**

Entry: `src/amg/graph.rs:3369-3378` (fallthrough when surface A returns
false). Calls `apx4::gpu_dispatch::dispatch_matmul`
(`src/apx4/gpu_dispatch.rs:37-67`).

- The first branch `if gpu_available()` calls
  `apx4::gpu_context::gpu_available()` (`src/apx4/gpu_context.rs:23-25`),
  which **reads `GPU_CONTEXT.available` — hardcoded `false` at
  construction** (`src/apx4/gpu_context.rs:11-18`).
- Therefore on main today this path **always falls through to
  `matmul_dispatcher::matmul_dispatch` (CPU)**.

**Surface C — `exec_gpu_segment` (planned segment loop)**

`src/gpu/dispatch/executor.rs:38-63`. Gated by `!record_tape` at
`src/amg/graph.rs:2651-2658`. Calls `cuda_matmul_inplace` per node.
Inner gate at line 39: `if !gpu_enabled() return;` where `gpu_enabled()`
= `cuda_available()` (`src/gpu/utils.rs:12-14`).

### Call chain when surface A succeeds (residency case, currently unreachable for 13B)

1. `Graph::execute_single_inner` → MatMul arm (`graph.rs:3085`).
2. `a.ensure_decoded()` / `b.ensure_decoded()` (`graph.rs:3106-3109`)
   — for `CpuBf16Shared` this transitions to owned `Cpu(Vec<f32>)`
   via `bf16_decode_bulk` (`tensor.rs:899-906`), losing Arc sharing
   on the consumer tensor.
3. If both are `Cuda`, `Tensor::zeros_new_cuda` allocates VRAM output
   (`graph.rs:3127-3137`).
4. `gpu_can_run_matmul` (shape gate, includes `cuda_available()` call).
5. `try_gpu_matmul` → all_cuda → `cuda_matmul_inplace` →
   device-pointer kernel.
6. Output stays on VRAM in the node's `output` slot.

### Pool block size

`DEFAULT_BLOCK_SIZE = 64 * 1024 * 1024` (`src/apx4_12/mod.rs:16`).

---

## 2. Gates Blocking 13B from Using GPU Today

13B representative shapes: Q/K/V/O proj `5120×5120` (100 MB F32),
FFN gate/up `5120×13824` (270 MB F32), FFN down `13824×5120`
(270 MB F32), LM head `5120×32000` (655 MB F32).

Gates encountered in evaluation order from `Graph::execute_single`
MatMul arm:

| # | Gate | Location | Effect on 13B |
|---|---|---|---|
| G1 | `record_tape` blocks `exec_gpu_segment` | `graph.rs:2651` | Inference is `record_tape=false`, but Llama segments are 1-node anyway because RmsNorm/Permute/Reshape break runs (segment build at `apx4_3/gpu_plan.rs:14-46` only allows `MatMul`). Surface C effectively dead on Llama. |
| G2 | `ensure_decoded` materialises BF16→F32 owned Vec | `graph.rs:3106-3109` + `tensor.rs:899-906` | Each MatMul cycle materialises a transient F32 copy of the BF16 weight (270 MB for FFN). Sharing breaks for that consumer copy. This is the M5.f.a steady state — it works only because the copy is dropped after the matmul returns. |
| G3 | All operands `Cuda`? | `gpu/dispatch/hooks.rs:175-183` | **False** for 13B. Weights are `CpuBf16Shared`, which after `ensure_decoded` becomes owned `Cpu(Vec<f32>)`. Activations are `Cpu(Vec<f32>)`. Storage is therefore `(Cpu, Cpu, Cpu)` — never `(Cuda, Cuda, Cuda)`. |
| G4 | `cuda_available()` (in `gpu_can_run_matmul`) | `gpu/dispatch/hooks.rs:115` and again at `:170` | On main today this **spawns `nvidia-smi` per call** (`src/cuda/mod.rs:19-21`). Returns true on a CUDA box, but at 50-300 ms cost per call. Today this is masked because gate G5 fails first (sub-millisecond bail). |
| G5 | `max_per_alloc <= 64 MiB` | `gpu/dispatch/hooks.rs:122-129` and `apx4_12/mod.rs:16` | **False** for every 13B weight matmul: 100 MB > 64 MiB, 270 MB > 64 MiB, 655 MB > 64 MiB. Returns false → surface A bails before surface A's roundtrip path can run. |
| G6 | Surface B gate `gpu_available()` | `apx4/gpu_context.rs:11-18, 23-25` | `GPU_CONTEXT.available` is **hardcoded `false`** at construction. `dispatch_matmul` therefore unconditionally routes to `matmul_dispatcher::matmul_dispatch` (CPU AVX2) at `apx4/gpu_dispatch.rs:51-52,62-63`. |

**Net result on M5.f.a main**: every 13B MatMul is decided by gates
**G3 false → G5 false → G6 false → CPU AVX2**. No Llama matmul ever
reaches a CUDA kernel. The R1 microbench (`f520b23`) confirmed the
kernel itself is fast; the gates are what keep 13B off-GPU.

---

## 3. M6 Commit-by-Commit Analysis

### `f520b23` — M6.a (bench-trace + R1 falsifier)

- **Dispatch change**: none. `bench-trace` Cargo feature is cfg-gated
  and default off.
- **New memory**: none in default builds.
- **Thrashing risk**: low. The investigation note (item 3) flags this
  as worth ruling out, but the cfg gates are zero-cost in default
  builds. Kept as "preserved asset" for the next attempt.

### `49fb81c` — M6.b (lift double GPU gate + non-pooled cuda_matmul)

- **Dispatch change**: two surfaces unblocked simultaneously:
  1. `apx4/gpu_context.rs::gpu_available` rewired to delegate to
     `cuda::cuda_available()` (no longer hardcoded false). Gate G6
     lifted.
  2. `gpu_can_run_matmul` pool-size check removed; `try_gpu_matmul`
     gained a `cuda_matmul_non_pooled` branch that calls
     `cudaMalloc`/`cudaFree` directly (bypassing the 64 MiB ceiling).
     Gate G5 effectively lifted for surface A.
- **New memory**: each oversize matmul (270 MB FFN) does
  `cudaMalloc(270 MB) ×3` + memcpy + free per call. No persistence.
  Approx. 25 GB/token PCIe traffic per the M6.c.3 commit message.
- **Thrashing evidence**: this is the commit that **unmasked the
  per-call `nvidia-smi` spawn** (`cuda/mod.rs:19-21`). Pre-M6.b,
  gate G5 short-circuited before G4 spawned the subprocess. Post-M6.b,
  surface A reaches `cuda_available()` for every matmul — ~360
  calls/decode-step × ~100 ms = ~36 s/token of orchestration overhead
  by itself (per M6.c.7 commit message).

### `01d6d0e` — M6.c.1 + .2 (Backend trait + planner)

- **Dispatch change**: zero. Adds `src/gpu/backend.rs` (vendor-neutral
  trait, `CudaBackend` impl) and `src/gpu/residency_planner.rs` (pure
  function). No callers wire to it yet.
- **New memory**: none. Plumbing only.
- **Thrashing risk**: nil on its own.

### `05c9086` — M6.c.3 + .4 (upload_resident_layers + mixed-storage matmul)

- **Dispatch change**:
  - `WeightStore::upload_resident_layers` uploads N layers' weights
    into VRAM at load time (residency).
  - `SharedParam::Gpu` variant added so `to_tensor()` returns
    `TensorStorage::Cuda` directly for resident layers.
  - Mixed-storage path `(a=Cpu activation, b=Cuda weight, out=Cpu)`
    added to `cuda_matmul_inplace` / hooks: small upload (activation),
    kernel on resident weight, small download.
- **New memory**:
  - **Persistent VRAM**: ~5-6 GiB for "resident fraction" (~32% of
    13B layers per the commit message).
  - **Transient F32 upcast**: when the source `SharedParam` is BF16,
    uploading to VRAM requires upcasting BF16 → F32 first (the only
    kernel ABI is F32). The commit doesn't change `bf16_decode_bulk`'s
    allocation behavior — every upload of a 270 MB BF16 weight
    materialises a 540 MB F32 transient before `cudaMemcpy`. Whether
    that transient is freed before the next layer uploads depends on
    the loop structure (which has not been inspected line-by-line
    because the commit is reverted; the asset is at `05c9086`).
  - **The 16 GiB shared-memory GPU heap risk** (investigation
    hypothesis 2): on this laptop, of the GPU's 23.9 GiB total
    addressable, 16 GiB is shared system RAM. Any `cudaMalloc`
    driver-side may map into that shared region under WDDM, reducing
    what's available for the CPU model.
- **Thrashing evidence**: 5-6 GiB persistent VRAM + 540 MB transient
  per upload + ~26 GiB BF16 model already in RAM, on a 32 GiB box →
  exactly the 29.7 GiB working set observed in the bug report. The
  100% disk C: usage is the pagefile spilling under that pressure.

### `3505c90` — M6.c.7 fixes 1-3

- Cached `cuda_available()` via `OnceLock<bool>` (kills the
  `nvidia-smi` per-call spawn — fix 1). Reorders `ATENIA_GPU=0`
  env-check before `cuda_available()` (fix 2). Restricts
  `cuda_matmul_non_pooled` to residency cases only (fix 3).
- This is a partial repair. According to `INVESTIGATION_M6.md`
  Section 4, regression persisted after these fixes (29.7 GiB RAM,
  278 s/tok), so the dominant pressure is from `05c9086`'s
  allocations, not from `cuda_available()` overhead.

### `847bee6` — M6.c.7 fix 4

- Reverts `apx4::gpu_context::gpu_available` to hardcoded `false`.
  Re-closes gate G6. Surface B disabled again.

### `fca7544` — M6.c.7 fix 5

- Reverts `gpu::utils::gpu_enabled` to hardcoded `false`. Disables
  surface C (`exec_gpu_segment`). At this point all three GPU surfaces
  are off — and the regression still persisted, confirming the
  pressure source is not in dispatch but in the residency / upload
  structures from `05c9086`.

**Hypothesis evaluation requested** ("did weight upload duplicate
memory temporarily?"): the code surface that would create the
duplication is `WeightStore::upload_resident_layers` at commit
`05c9086`. The BF16→F32 upcast is unavoidable with the current F32-only
kernel ABI; **whether the upload loop holds the F32 transient + the
BF16 source + the new `SharedParam::Gpu` simultaneously is the
load-bearing question** — if yes, peak is `BF16 (270 MB) + F32
transient (540 MB) + VRAM upload in-flight (540 MB) = 1.35 GiB per
layer in-flight`. Across 13 resident layers that's ~17 GiB of new
pressure on top of the 26 GiB BF16 model. Direct verification requires
inspecting `05c9086`'s exact loop structure, which is outside the
read-only scope on current main.

---

## 4. Architectural Options for 13B-on-RTX-4070

### Per-token matmul count

Reading `Graph::execute_single_inner` and the Llama builder
(`amg/graph.rs:3085` MatMul arm + `BatchMatMul` at `:3700`): per layer,
decode step has Q/K/V/O proj (4 MatMul) + FFN gate/up/down (3 MatMul)
+ 2 BatchMatMul (attention QK·V). 40 layers × 7 MatMul = **280 MatMul
+ ~80 BatchMatMul + 1 LM head** per token.

### Option (a) — Weight residency (some layers permanently in VRAM)

- VRAM per layer (F32): 4 × 100 MB (QKVO) + 3 × 270 MB (FFN) ≈
  1.21 GiB/layer.
- VRAM per layer (BF16, requires kernel ABI change): ~610 MB/layer.
- 8 GiB VRAM ceiling minus ~1 GiB OS/driver overhead ≈ 7 GiB usable.
- F32 budget: ~5 layers fit (5/40 = 12.5%); the other 87.5% still go
  CPU. Throughput uplift modest.
- BF16 budget: ~11 layers fit (~28%). But requires a BF16 kernel
  (does not exist today; only `matmul_f32_launch_device` and
  `launch_linear_f32_device_ptrs`).
- **Code surfaces**: `WeightStore::upload_resident_layers` (revert
  asset at `05c9086`); `SharedParam::Gpu` variant; mixed-storage path
  in `cuda_matmul_inplace` (asset at `05c9086`).

### Option (b) — Per-matmul streaming (upload weight → kernel → free)

- VRAM peak per call: 270 MB (weight) + 80 KB (activation) + 80 KB
  (output) ≈ 270 MB. Trivial for 8 GiB.
- Overhead per matmul: H→D 270 MB at PCIe 4.0 x16 ≈ 17 ms theoretical;
  in practice 50-100 ms with `cudaMalloc`/`cudaFree` on every call.
  360 matmul/token × ~80 ms = ~29 s/token of pure transfer — worse
  than the 14 s/tok M5.f.a CPU baseline.
- **Code surface**: `cuda_matmul_non_pooled` (asset at `49fb81c`)
  + bypassing gate G5.

### Option (c) — Activations on GPU, weights stream BF16

- Requires a BF16 weight kernel: source `b` is `*const u16`, decode
  in-kernel or via tensor cores. Today's ABI is F32-only:
  `matmul_f32_launch_device` (`src/cuda/matmul.rs:9-18`).
- Activation residency between ops would also require RmsNorm/RoPE/
  SiLU GPU kernels (none exist for residency at 5120-dim activation;
  `cuda_fused_linear_silu` exists for fused Linear+SiLU only).
- **VRAM peak**: <500 MB (activations + KV cache slice + one streamed
  BF16 weight 135 MB).
- **Viability**: lowest VRAM pressure but **highest engineering cost**
  — new kernel ABI, new kernels for non-MatMul ops on Cuda storage,
  new residency planner for activations. Not a minimum-viable path.

---

## 5. Minimum Viable Change

Smallest change that puts at least one 13B matmul on the RTX 4070
without re-introducing the M6.c regression:

**Surface to modify**: `gpu/dispatch/hooks.rs:107-132`
(`gpu_can_run_matmul`) and `gpu/dispatch/hooks.rs:151-238`
(`try_gpu_matmul`).

**Specific lines**:

- `gpu/dispatch/hooks.rs:127-129`: the
  `max_per_alloc > DEFAULT_BLOCK_SIZE` early-return is the gate that
  keeps every 13B matmul out of CUDA. Removing it (or routing oversize
  shapes to a non-pooled path) is the surgical change.
- `cuda/mod.rs:19-21`: `cuda_available()` must be cached behind
  `OnceLock<bool>` BEFORE any other gate is lifted, or the regression
  returns immediately. This is non-negotiable — see the M6.c.7 fix 1
  evidence.
- `cuda/matmul.rs:31-59`: `cuda_matmul` would need a non-pooled
  variant for one specific shape class. The existing
  `with_pooled_device_buffers` cannot serve > 64 MiB.

**Invariants that would break**:

1. Pool block contract (`apx4_12/mod.rs:16`): currently `pool_alloc()`
   always returns a 64 MiB block. A non-pooled path circumvents the
   pool's bookkeeping.
2. `as_cpu_slice` panics on `Cuda` storage (`tensor/tensor.rs`): any
   code path that slips through with mixed storage will panic. The
   defensive `ensure_cpu` at `graph.rs:3327-3335` exists for this
   reason.
3. F32 kernel ABI: residency-capable kernels are F32-only. BF16
   weights must be upcast to F32 before VRAM upload (doubling
   transient memory).
4. Counter semantics: `GPU_MATMUL_RESIDENT_COUNT` /
   `_ROUNDTRIP_COUNT` / `_LEGACY_COUNT` partition by dispatch path;
   a new path needs a counter or an explicit binding to one.

**Scope**: 1 file (`gpu/dispatch/hooks.rs`) + 1 file
(`cuda/matmul.rs` for non-pooled variant) + `cuda/mod.rs` for the
cache. Three files. Single matmul shape class to validate.

---

## 6. Validation Protocol Proposal

### Existing instrumentation

- `gpu_matmul_resident_count()` (`gpu/dispatch/hooks.rs:48`):
  residency path hits.
- `gpu_matmul_roundtrip_count()` (`gpu/dispatch/hooks.rs:55`):
  pooled CPU-roundtrip hits.
- `gpu_matmul_legacy_count()` (`gpu/dispatch/hooks.rs:67`):
  apx4 legacy path hits.
- `gpu_matmul_total_count()` (`gpu/dispatch/hooks.rs:77`): union —
  answers "did any matmul reach a CUDA kernel?"
- `APX_TRACE=1` env var produces per-call printlns at
  `gpu/dispatch/hooks.rs:186-188, 226-228`.
- `bench-trace` Cargo feature (preserved at `f520b23`): per-NodeType
  wall-clock accumulators inside `Graph::execute_single`.

### Baseline reference

The right anchor is M5.f.a as measured live on the dev box:

- **14 s/token** (Llama 2 13B Chat, decode after prefill).
- **~26 GiB RAM working set** (BF16 model resident).
- **~0% disk C: usage during decode**.
- **0 GPU usage**.

These numbers come from the bug report in `INVESTIGATION_M6.md`
Section 4 and are the baseline any M6 retry must not regress.

### Rollback criteria

1. **Hard regression**: > 1.5× M5.f.a baseline (i.e. > 21 s/token)
   in either `ATENIA_GPU=0` or default config → revert immediately.
2. **RAM pressure**: peak working set > 28 GiB → revert (close to
   32 GiB physical means pagefile thrashing).
3. **Disk C: sustained > 50%** during decode → revert (pagefile
   spill).
4. **Counter check**: if `gpu_matmul_total_count() == 0` after a
   forward pass with kill-switch off, the change is dispatch-dead and
   adds risk without benefit → revert.

### Per-commit protocol (per the M6 note Section 5)

- One activation per commit. No grouping.
- Smoke immediately after the commit on the dev box, both with
  `ATENIA_GPU=0` and with the new path enabled.
- Capture
  `Get-Process atenia | Select-Object WorkingSet64,PrivateMemorySize64`
  and the relevant counter values.
- If any rollback criterion fires, revert that single commit before
  continuing.

---

## 7. Surprises / Contradictions

### S1 — `apx4::gpu_context::gpu_available` has been hardcoded `false` since long before M6

`src/apx4/gpu_context.rs:11-18` literally constructs `available: false`
with a "Placeholder until real CUDA integration (APX 4.0)" comment.
M6.b (`49fb81c`) lifted this; the fix-4 revert (`847bee6`) put it back.
**On current main (M5.f.a) this gate is closed**, meaning surface B
(the entire legacy `dispatch_matmul_gpu` path) is dead code today.
Yet `src/apx4/gpu_dispatch.rs:11-32` documents that "the whole 13B
forward routes through `dispatch_matmul_gpu`" — that documentation
reflects what M6.b temporarily made true, not what main does. **The
doc comment in `gpu_dispatch.rs` is stale relative to M5.f.a state.**

### S2 — `GPU_MATMUL_LEGACY_COUNT` will always be zero on current main

The counter is only incremented at `apx4/gpu_dispatch.rs:50,61` inside
the `if gpu_available()` branches. With `gpu_available()` hardcoded
false, the counter never fires. The `gpu_matmul_total_count()`
aggregator (`gpu/dispatch/hooks.rs:76-80`) will likewise return 0 for
any Llama 2 13B run on current main, despite documentation suggesting
otherwise.

### S3 — `cuda_available()` on main DOES still spawn `nvidia-smi` per call

`src/cuda/mod.rs:19-21` is unchanged from pre-M6 (the `OnceLock` cache
fix from M6.c.7 was reverted). On current main this only matters for
surfaces that reach it; gates G5 and G6 short-circuit before G4 in
the 13B path, so the spawn doesn't happen on the hot path today.
**But any future change that lifts G5 (the pool-size gate) without
simultaneously caching `cuda_available()` will reproduce the M6.b
regression instantly.** This is the single biggest tripwire for the
next M6 attempt.

### S4 — `ensure_decoded` breaks Arc-sharing for BF16 weights on the MatMul path

`src/amg/graph.rs:3106-3109` calls `a.ensure_decoded()` /
`b.ensure_decoded()` on cloned tensors. For `CpuBf16Shared`,
`ensure_decoded` delegates to `ensure_cpu`, which **transitions the
cloned tensor's storage to owned `Cpu(Vec<f32>)`**
(`tensor/tensor.rs:899-906`). The original Arc remains shared in the
WeightStore, but each MatMul allocates a fresh 270 MB F32 transient
per call. This is an **existing ~270 MB-per-matmul transient on the
M5.f.a steady state** that the system tolerates because it's freed
when the function returns. Any M6 change that retains the F32
transient longer (e.g. by caching it for repeated GPU dispatch)
materially increases peak RAM.

### S5 — `cuda_matmul_inplace` has a confusingly named "fallback" that allocates two clones

`src/cuda/matmul.rs:115-138`: the non-`all_cuda` branch calls
`a.clone()`, `b.clone()`, `ensure_cpu` on each, then `cuda_matmul`.
This is two extra full F32 clones of 270 MB tensors. Reachable only
when storage is mixed (which the comments at `:117-124` claim is
unreachable on the Llama hot path), but if it ever does fire it will
allocate 540 MB transient before the kernel call. The fallback comment
claims "this branch is unreachable on the Llama hot path" — that
claim is correct today but would silently become a memory regression
if anything lifted the storage uniformity.

No other surprises encountered during the read-through that contradict
`INVESTIGATION_M6.md`.
