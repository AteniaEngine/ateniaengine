# Handoff — APX v20 M8 (BF16-resident VRAM kernels, at M8 close)

**Status at handoff**: M8 closed. The tier-aware loader can now
keep BF16 weights resident in VRAM at half the F32 byte cost
(`numel × 2` instead of `numel × 4`), doubling the planner's
effective VRAM capacity. The dispatcher routes BF16-resident
weights through `cublasGemmEx` after upcasting them to F32 on
the device per-matmul (Path B), preserving the M4.7.2.e CPU
path's numerical envelope (drift ≤ 2.4e-2 vs F64 4-model
fixture). Llama 2 7B Chat runs **1.31× faster than the M6
baseline**; Llama 2 13B Chat runs **1.36× faster than the M7.3
baseline**, with the four production checkpoints all passing
ADR-004 with margin 21–12,500×.

After `cargo build --release --bin atenia`, running

```powershell
$env:ATENIA_M8_BF16_KERNEL    = "1"
$env:ATENIA_TIER_AWARE_LOADER = "1"
$env:ATENIA_DISK_TIER_DIR     = "D:\atenia-m8-cache"

cargo run --release --bin atenia -- generate `
    --prompt "Hello, how are you?" `
    --model F:/Proyectos/artenia_engine/atenia-engine/models/llama-2-13b-chat `
    --max-tokens 5
```

loads Llama 2 13B Chat with the M8 plan, places **82 weights as
BF16 in VRAM** (vs 38 as F32 in M7.3 — exactly the 2.16× capacity
benefit M8.3 promised), keeps 197 weights on NVMe (vs 239 in M7.3
— 17.6 % fewer Disk reads per token), runs each VRAM matmul as
`bf16 → f32 transient → cublasGemmEx(F32, F32, F32)`, and
produces the same coherent reply as M6/M7:

```text
[ATENIA] M8 BF16 kernel active: VRAM budget doubles ...
[ATENIA] Adaptive headroom: model 24.24 GiB, free RAM 19.28 GiB
         → RAM headroom 18.75 GiB (8.00 base + 10.75 overflow)
[ATENIA] Tier-aware loader plan:
  VRAM: 82 tensors (6.74 GiB)
  RAM:  124 tensors (0.49 GiB)
  Disk: 197 tensors (17.01 GiB)

> Hello, how are you?

 Hello! I'

---
Generated: 5 tokens in 135.0s (0.04 tok/s)
```

**Last M8 commit**: `6d343c3` (M8.4c — Path B). M8 closes here;
the next active milestone is **M8.7** (Disk → GPU JIT pipeline)
with target ~5–7 s/tok for the 13B based on the M8.0b pipeline
async bench, or **M9** (INT8 quantisation) as an alternative
that attacks the per-byte cost rather than the I/O bound.

---

## Headline result

| Path | Token avg | VRAM tensors | Disk tensors | ADR-004 4-model |
|---|---|---|---|---|
| **M6 (Llama 2 7B)** | 8.22 s/tok | 60 (F32) | 0 | n/a |
| **M8 (Llama 2 7B)** | **6.26 s/tok** (1.31×) | 128 (BF16) | 0 | drift ≤ 2.4e-2 ✓ |
| **M7.3 (Llama 2 13B)** | 36.6 s/tok | 38 (F32) | 239 | n/a |
| **M8 (Llama 2 13B)** | **27.0 s/tok** (1.36×) | 82 (BF16) | 197 | drift ≤ 2.4e-2 ✓ |

ADR-004 4-model F64 validation under M8 BF16 kernel:

| Model            | Drift M8 (Path B) | Drift M8.4-original | Reduction | Argmax |
|------------------|-------------------|---------------------|-----------|--------|
| TinyLlama 1.1B   | 8.8e-5            | 0.901545            | 10,250×   | 4/4 ✓ |
| SmolLM2 1.7B     | 7.31e-4           | 2.331949            | 3,190×    | 4/4 ✓ |
| Qwen 2.5 1.5B    | 2.40e-2           | 0.184907            | 7.7×      | 4/4 ✓ |
| Llama 3.2 1B     | 4.0e-5            | 0.268043            | 6,700×    | **4/4 ✓** (pos 2 fixed) |

All four under the 0.5 ADR-004 threshold with margin 21–12,500×.
The argmax mismatch on Llama 3.2 position 2 that the M8.4-original
path produced (likely a near-tie that BF16 truncation flipped) is
resolved under Path B.

---

## Sub-phase ledger

| Commit   | Sub-phase | Scope |
|----------|-----------|-------|
| `b955669` | M8.0      | cuBLAS BF16 TC bench (gating data 1) — empirical 2.2× over naive F32 on the four 13B decode shapes; established that decode matmuls are bandwidth-bound at M=1 |
| `f71caf0` | M8.0b     | NVMe→PCIe→GPU pipeline async bench (gating data 2) — Config 2 (two-buffer pipeline) at 32.7 ms/iter on FFN-down shape, well under 200 ms threshold; PLAN A confirmed for the M8.7 path |
| `74f7631` | M8.1      | `TensorGPU::dtype` field + `bf16_to_vram_no_upcast` primitive — BF16 resident in VRAM without F32 conversion |
| `956fefc` | M8.2      | `cuda_matmul_bf16_inplace` original (BF16 inputs, F32 output) — first cublasGemmEx wire-up, drift envelope per single-matmul on the four shapes |
| `2e3a920` | M8.3      | `TierPlan::vram_cost_bytes` parametrised by dtype + `ATENIA_M8_BF16_KERNEL` flag — `numel × 2` for BF16, `numel × 4` for F32 |
| `e25cd7e` | M8.4      | End-to-end wire-up (loader gate + dispatcher arm) + shared `BF16_COUNTER_TEST_LOCK` for cross-module test serialisation |
| `a956044` | M8.4b     | Slow-path BF16 fix for weights with `LoadTransform`s (Transpose2D etc.) — every Llama-family `_proj.weight` falls here, the fast-path arm alone covered nothing in production |
| `790d165` | M8.5      | F64 4-model validation test infrastructure |
| `577d0d6` | docs      | `docs/MODELS_LAYOUT.md` — canonical model checkpoint paths so no future Code wastes a cycle re-discovering them |
| `a0ef6db` | M8.5b     | M8.5 counter-assertion fix (sum fast + slow loader counters) |
| `6d343c3` | **M8.4c** | **Path B fix: BF16 weight in VRAM + F32 upcast per-matmul + cublasGemmEx F32 GEMM**. Replaces the M8.4-original BF16 input path that cascaded BF16 activation truncation through 16-28 layers, failing ADR-004 by 2-5× in three of four models. |

---

## Architectural decisions (D76+)

1. **The M8 thesis "BF16 in VRAM ⇒ doubled capacity ⇒ 5× speedup
   from Tensor Cores" was partially right and partially wrong**.
   Capacity doubling is real and delivered: 7B got 60 → 128 VRAM
   tensors, 13B got 38 → 82. But the Tensor Core speedup turned
   out smaller than the bench predicted because decode-step
   matmuls (M=1) are memory-bandwidth-bound, not compute-bound —
   M8.0 measured 2.2× peak speedup of BF16 TC over the naive F32
   kernel, but the *production* contribution to per-token
   latency was always going to be marginal because the VRAM
   layers cost ~30 ms total against the 30 s the Disk-tier
   layers spend on the CPU. The headline wins (1.31× / 1.36×)
   come from **moving more weights off the Disk path**, not from
   faster matmuls.

2. **The cascading drift problem is fundamental, not a bug**.
   M8.5 surfaced that the M8.4-original path (BF16 weight × BF16
   activation, F32 accumulate, on Tensor Cores) drifts 2-5× over
   ADR-004 when run end-to-end on real Llama checkpoints. The
   activation BF16 truncation by itself is responsible: each of
   the 7 matmuls per layer truncates the activation to 7-bit
   mantissa, the truncated activation feeds the next layer's
   residual, and over 16-28 layers the bias compounds. cuBLAS on
   sm_89 does not support mixed `(F32 act × BF16 weight)`
   directly through `cublasGemmEx` (verified via NVIDIA docs);
   `cublasLtMatmul` supports more flexibility but not for this
   particular dtype combination on Ada (FP8 mixed is sm_90+).
   `CUBLAS_COMPUTE_32F_FAST_16BF` is numerically identical to
   the M8.4-original path because cuBLAS rounds F32 to BF16 the
   same way our host cast does.

3. **Path B is the correct architectural answer**. Keep the
   weight as BF16 in VRAM (preserves M8.3 capacity-doubling),
   but upcast it to a fresh F32 transient on-device per-matmul
   via the existing M6 `bf16_to_f32_launch_device` kernel, then
   run `cublasGemmEx(F32, F32, F32)`. The numerics match
   M4.7.2.e (BF16 storage on CPU + F32 matmul) bit-for-bit
   except for cuBLAS internal rounding (1.64e-7 measured single-op
   drift in `m8_4c_dispatcher_bf16_resident_via_f32_upcast_strict_drift`).
   Per-matmul cost is ~3× the M8.4-original BF16 TC path, but
   sub-millisecond regardless and invisible at 13B end-to-end
   scale.

4. **The M8 BF16 path is gated by an adaptive heuristic in
   `pipeline.rs`**, not just the env var. The flag activates iff
   `model_total_bytes > 0.7 × free_ram_bytes`. For 7B-class
   models that fit in RAM with headroom, the flag is ignored
   even when set — Path B's per-matmul upcast cost is a
   regression there. The threshold matches the M7.2 adaptive
   headroom rule, single-sourcing the "model dominates RAM"
   contract.

5. **`WeightMapper::set_bf16_kernel_active(Option<bool>)` is the
   loader's contract knob**. Pipeline computes the effective
   flag once per load and passes it explicitly via this setter.
   The loader internally calls `m8_bf16_kernel_active()` which
   prefers the explicit override and falls back to the env var
   for direct test callers (so `tests/m8_loader_test.rs` keeps
   working with `M8FlagGuard` unchanged).

6. **Counter discipline**: M8 loads (and the dispatcher) bump a
   disjoint set of counters from M6/M7. A single Vram-tier load
   increments exactly one of:
   - `vram_fast_path_count` (M6 F32 fast)
   - `vram_slow_path_count` (M6 F32 slow)
   - `vram_bf16_fast_path_count` (M8 BF16 fast)
   - `vram_bf16_slow_path_count` (M8 BF16 slow)
   And the dispatcher bumps `vram_bf16_matmul_count` exactly
   once per `cuda_matmul_bf16_inplace` call. This makes the M8
   path auditable from any production log via the counter
   summary (4 counts at startup, expected to match
   `proj_weight_count` for the model).

---

## Gaps closed by M8

- **The M6 path's "F32 in VRAM forever" assumption** was the
  reason the M7.3 13B smoke had 38 tensors in VRAM and 239 on
  Disk despite "abundant" 8 GiB free VRAM. M8 doubles the
  capacity by storing BF16, raising VRAM count to 82 and
  cutting Disk count to 197 — 17.6 % fewer NVMe reads per token.

- **The M8.4-original BF16 input path's numerical contract**
  was broken in 3 of 4 production models (M8.5 surfaced this).
  M8.4c replaces it with Path B and matches M4.7.2.e numerics
  exactly, closing the contract.

- **The M8 flag was un-gated in M8.4** — every load that set
  the env var paid the BF16 path cost regardless of whether the
  capacity doubling helped. M8.4c's adaptive heuristic gates
  by `model_total > 0.7 × free_ram` so 7B models that don't
  need M8 don't pay the per-matmul upcast cost in production.

- **Test cross-module flakiness** on the BF16 counter, observed
  during M8.4 development: tests in `cuda::bf16_to_f32::tests`
  and `cuda::matmul::cuda_matmul_bf16_tests` were racing on the
  shared `BF16_RESIDENT_COUNT`. M8.4 introduced
  `BF16_COUNTER_TEST_LOCK` (`pub(crate) static Mutex<()>`) to
  serialise both modules' counter snapshots. Three consecutive
  full `cargo test --lib` runs are clean post-fix (181/181 each).

## Gaps left open (intentional)

- **The 13B Disk-tier path is unchanged**. 197 weights still
  decode-on-CPU per token. That's ~26 of the 27 s/tok (only
  ~1 s went away from moving 42 weights off Disk). The big
  unblock is M8.7 (Disk → GPU JIT pipeline) where the M8.0b
  bench already showed Config 2 at 32.7 ms / 135 MiB, projecting
  ~5–7 s/tok for the full 13B. M8.7 is the natural next step.

- **`CUBLAS_COMPUTE_32F` does not use Tensor Cores on F32 inputs**.
  Path B leaves Ada's BF16 TC peak performance on the table
  (the M8.0 bench's 2.2× was unreachable without the activation
  truncation that broke ADR-004). A future variant could try
  `CUBLAS_COMPUTE_32F_FAST_TF32` (TC TF32, 10-bit mantissa) —
  M8.0 measured 1.04–1.26× over naive F32 with that compute
  type. Numerics estimate: drift would scale ~4× the BF16-only
  case ÷ TF32-vs-BF16 mantissa ratio, so SmolLM2's 7.31e-4
  could rise to ~3e-3, still under ADR-004 with margin. **Not
  in scope; recorded as a follow-up if M8.7 numerics give us
  more headroom.**

- **`CUDA_R_16BF` constant in `src/cuda/matmul.rs` carries
  `#[allow(dead_code)]`** as a documentation anchor for the
  former M8.4-original path. If a future M8.x revisits BF16
  inputs (e.g. for prefill where M is large enough that the
  matmul becomes compute-bound and TC pays off), the constant
  is already there; otherwise it can be removed in a tidy-up
  pass.

- **Qwen 2.5 1.5B's M4.7.2.e drift was the highest of the four
  models** (2.9e-2) and remains the highest under M8 (2.4e-2).
  This is consistent — Qwen has unusual weight statistics that
  show up under any BF16 path. Not a bug; not in scope to fix.

---

## How to resume

Two natural follow-ups, each with its own gating data already
landed in the M8 cycle:

### Option 1 — M8.6 (BF16 KV cache, D62)

Cheap, deterministic, well-scoped. Migrate `KvCache` from F32
to BF16 with cast on write/read. Validate ADR-004 (envelope
matches the M4.7.2.e BF16 storage path on cache too). Saves
1.6 GiB of RAM in seq_len 2048 on 13B. Independent of M8.7
and a clean ~1 day of work.

Files: `src/amg/kv_cache.rs` for the migration; `tests/m4_*` and
the F64 4-model fixture for validation.

### Option 2 — M8.7 (Disk → GPU JIT pipeline)

The big unblock. M8.0b's measured 32.7 ms / iter for the FFN-
down shape projects the full 13B per-token cost to ~5–7 s under
Config 2 (two-buffer pipeline: NVMe read of slot N+1 overlaps
with GPU compute of slot N). Implementation:

1. Producer thread reads safetensors-mmap'd slice for the next
   weight; pinned-host staging (already validated in the bench).
2. `cudaMemcpyAsync` on a copy stream uploads to a VRAM staging
   slot (BF16 — stays as BF16 in VRAM under the M8 contract).
3. `cuda_matmul_bf16_inplace` consumes the staged weight (Path B
   upcasts to F32 transient and runs the matmul) on the compute
   stream, with a per-slot `cudaEvent` synchronising upload →
   compute.
4. After the compute on slot N completes, the slot is recycled
   for slot N+2's read (two-buffer pattern).

The M8.0b bench's `examples/bench_disk_gpu_pipeline.rs` is the
template; the production wire-up reuses the same primitives
plus a new orchestrator on the loader side that decides when
to stream Disk-tier weights through the pipeline vs. fall back
to the existing CPU `ensure_cpu` path.

Decision criterion at M8.7 close: 13B per-token ≤ 8 s. Stretch
target: 5 s. The pipeline is bandwidth-bound on NVMe at ~135
MiB / 32.7 ms = 4.13 GB/s sustained — very close to the M7.0
measured 4.32 GB/s ceiling on cold reads.

### Option 3 — M9 (INT8 quantisation)

Different attack vector — instead of doubling VRAM capacity by
halving per-weight bytes from 4 to 2, halve again to 1. Same
plan structure (TierPlan with `kernel_dtype = Int8`,
`vram_cost_bytes = numel × 1`). Requires per-tensor calibration
(scale / zero-point), which is a substantial new infrastructure
path. Reserved as the milestone after M8.7 unless M8.7's measured
performance is unsatisfactory.

---

## Tests

5 mandatory regression suites green at every M8 commit:

```text
tinyllama_config_test               15 passed (3 ignored)
tinyllama_weight_loading_test       10 passed (1 ignored)
tinyllama_builder_test              10 passed
weight_mapper_test                   5 passed
miniflux_safetensors_roundtrip_test  3 passed
                                  ---
                                   43 tests
```

Plus M8-specific suites:

```text
gpu::tier_plan unit suite           15 passed (5 M6 + 5 M7.2 + 5 M8.3)
cuda::bf16_to_f32 unit suite         3 passed (1 M6 + 2 M8.1)
cuda::matmul unit suite              4 passed (1 M6 + 1 M8.2 + 2 M8.4c)
m6_replan_loader_test                1 passed (2 ignored — CUDA gated)
m7_disk_loader_test                  2 passed (slow path + fast path)
m8_loader_test                       4 passed (1 default + 3 M8.4/M8.4b)
m8_5_full_family_validation_test     4 passed (TinyLlama, SmolLM2, Qwen 2.5,
                                                Llama 3.2 — all under
                                                ADR-004 threshold 0.5)
                                  ---
                                   33 additional tests
```

3 × `cargo test --release --lib` consecutive: 181/181 passed
each run, no flakiness post-`BF16_COUNTER_TEST_LOCK` fix.

---

## Operator quickstart (for the next person)

```powershell
# Hardware: 32 GiB Windows box, NVIDIA RTX 4070 Laptop (8 GiB), NVMe.

# Smoke 7B (M8 activates because 12.55 GiB > 0.7 × ~17 GiB free):
$env:ATENIA_M8_BF16_KERNEL    = "1"
$env:ATENIA_TIER_AWARE_LOADER = "1"
cargo run --release --bin atenia -- generate `
    --prompt "Hello, how are you?" `
    --model F:/Proyectos/artenia_engine/atenia-engine/models/llama-2-7b-chat `
    --max-tokens 5

# Smoke 13B (M8 activates and overflows ~17 GiB to NVMe):
$env:ATENIA_DISK_TIER_DIR = "D:\atenia-m8-cache"
cargo run --release --bin atenia -- generate `
    --prompt "Hello, how are you?" `
    --model F:/Proyectos/artenia_engine/atenia-engine/models/llama-2-13b-chat `
    --max-tokens 5

Remove-Item Env:ATENIA_M8_BF16_KERNEL
Remove-Item Env:ATENIA_TIER_AWARE_LOADER
Remove-Item Env:ATENIA_DISK_TIER_DIR
```

Expected: 7B in ~30 s (5 tokens, 6.26 s/tok), 13B in ~135 s
(5 tokens, 27.0 s/tok). Both should print " Hello! I'" — the
production text from M6/M7. If the text differs or any
counter (`vram_bf16_fast_path_count + vram_bf16_slow_path_count`)
stays at 0 with the flag set, something is wrong — see the
M8.4b counter discipline in this document.

For the F64 4-model regression test, see `docs/MODELS_LAYOUT.md`
for the canonical paths.
