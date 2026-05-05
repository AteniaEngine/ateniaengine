# HANDOFF — APX v20 M8.7 (Disk → GPU JIT pipeline)

**Status at handoff**: M8.7 closed. The Llama 2 13B Chat
forward at seq=4 runs in **10.22 s wall-clock** with the
disk-streamed BF16 GPU path active and 98.7 % CPU prefetch
hit rate, down from the M8 baseline of **13.50 s** measured
during the M8.7 bring-up smoke (24 % faster). Per-token
generation through `atenia generate` measures **20.7 s/tok**
on the 13B with M8.7.0 + M8.7.1.a active vs the **27.0 s/tok**
M8 close baseline (~24 % faster, same trajectory). All four
production checkpoints (TinyLlama 1.1B, SmolLM2 1.7B,
Qwen 2.5 1.5B, Llama 3.2 1B) still pass ADR-004; the 13B
argmax fixture (`1, 1, 185, 29892`) is bit-exact with the
M8 close run.

After `cargo build --release --example llama2_13b_demo`,
running

```powershell
$env:ATENIA_M8_BF16_KERNEL = "1"
$env:ATENIA_M8_7_ENABLED   = "1"
$env:ATENIA_DISK_TIER_DIR  = "D:\atenia-m8-cache"
cargo run --release --example llama2_13b_demo
```

reproduces the headline numbers:

```text
Tier plan: vram=78 tensors (6.46 GiB), ram=171, disk=154
Forward: 10.22s
GPU MatMul invocations: resident=78, roundtrip=0, legacy=0
BF16-resident matmuls (M8.4c): 232
Disk-streamed matmuls (M8.7.0): 154
Disk prefetch hits (M8.7.1.a): 152
Logit stats: max |v|=15.1610  mean |v|=1.1454  finite=128000/128000

Pos 0: argmax id=    1  logit=4.7747
Pos 1: argmax id=    1  logit=5.1188
Pos 2: argmax id=  185  logit=15.1610
Pos 3: argmax id=29892  logit=5.7977
```

The 152 / 154 prefetch hit rate (98.7 %) leaves exactly two
non-hits per forward: the first disk-streamed matmul has no
prior `kick_off` (slot empty on the first call) and the last
one has no `next_handle` to kick off (no more disk matmuls
ahead in the plan).

**Last M8.7 commits**:

- `c186148` — M8.7.1.a: CPU prefetch + `exec_gpu_segment` guard.
- `c0aee16` — M8.7.1.r: Path B `cuda_matmul_bf16_inplace`
  stream-aware refactor.
- `<this commit>` — M8.7 milestone close.

---

## Headline numbers

| Smoke | M8 baseline | M8.7 close | Speedup |
|---|---:|---:|---:|
| 13B `atenia generate` 5-token | 27.0 s/tok | **20.7 s/tok** | **1.30×** |
| 13B killer demo seq=4 forward | 13.50 s | **10.22 s** | **1.32×** |
| Disk-streamed matmuls / forward | 0 | 154 | (new) |
| BF16-resident matmuls (M8.4c) | 281 | 232 | — |
| Disk prefetch hits / forward | 0 | 152 | (new) |

Argmax matches the operator-confirmed M8 baseline bit-exactly
in every position. Logits sane (`max=15.16`, `finite=128000/128000`).

---

## Sub-phase ledger

| Sub-phase | Commit | What shipped |
|---|---|---|
| **M8.7 prereq** | `3e64d9a` | Tier planner reserves `DISK_PIPELINE_STAGING_BYTES = 2 × 135 MiB` of VRAM headroom whenever the plan would otherwise overflow to Disk. Two-pass `plan()` keeps the budget exact for the streaming staging slots without penalising 7B-class plans that don't need them. |
| **M8.7.0** | `96f14a1` | MVP single-tensor Disk → GPU staging dispatch (`cuda_matmul_disk_streamed_bf16`). New host primitive `disk_tier::read_bf16_raw_bytes` (zero-host-allocation streaming reader). `MatMul` arm gate in `execute_single_inner` routes Disk(BF16) operands to the streaming dispatch under `ATENIA_M8_7_ENABLED=1`. Counter `disk_streamed_matmul_count` for observability. Single-op drift gate 3.28e-3 vs ADR-004's 0.5. |
| **M8.7 routing fix** | `4e126f0` | Killer demo and `src/demo/mod.rs` migration to `load_into_with_residency_plan` + `build_llama_with_store` rebuild pattern (canonical `LlamaPipeline::load` flow). Closes the zero-logit regression introduced by the bare `load_into_with_residency_plan` call that left Vram/Disk slots zero-initialised. |
| **M8.7 default flip** | `afaa975` + `21e5bb8` | `ATENIA_TIER_AWARE_LOADER` inverted: tier-aware is now default, `ATENIA_LEGACY_LOADER=1` is the opt-out. D74 from `HANDOFF_M6.md` superseded. |
| **M8.7 demo enable** | `c7b2ea9` | `Graph::execute_inference` (record_tape=false entry point). `Graph::segment_has_bf16_or_disk_operands` guard skips the legacy `exec_gpu_segment` (APX 4.x F32-only path) when operands are BF16-resident or Disk-tier. |
| **M8.7.1.a** | `c186148` | CPU prefetch single-slot helper (`src/cuda/disk_prefetch.rs`). `kick_off(handle)` spawns a background `read_bf16_raw_bytes` on a `rayon` worker; `take(handle)` consumes the slot. `cuda_matmul_disk_streamed_bf16` now takes `next_handle` and threads the prefetch through the dispatch chain. `Graph::find_next_disk_bf16_handle_after` provides the executor lookahead. Guard cap at 135 MiB excludes `lm_head` and any oversized weight. Extended `exec_gpu_segment` guard to also skip when `ATENIA_M8_7_ENABLED=1` (the legacy GPU pool fails intermittently with `TransferFailed` under the M8.7 staging churn; modern dispatch handles every operand kind correctly). 4 new unit tests. |
| **M8.7.1.r** | `c0aee16` | `cuda_matmul_bf16_inplace` (Path B M8.4c) refactored to accept `stream: *mut c_void`. Replaced device-wide `cudaDeviceSynchronize` (line 737) with per-stream `cudaStreamSynchronize` after `cudaMemcpyAsync` D→H. New FFI for `cudaMemcpyAsync` + `cudaStreamSynchronize`. With `stream = null` (default stream) the behaviour is bit-exact with the pre-refactor body. Enables M8.7.1.b/c (compute / copy stream split) without re-touching the kernel core. |

---

## Sub-phases skipped — and why

### M8.7.1.b/c (async H→D upload + dedicated copy/compute streams) — **DEFERRED**

The original M8.7.1 plan called for two more sub-phases on top of
M8.7.1.a:

- **M8.7.1.b** — async H→D pipeline using a dedicated `copy_stream`,
  CUDA events for handoff, and persistent two-slot VRAM staging
  (270 MiB, the exact reservation the M8.7 prereq put in place).
- **M8.7.1.c** — full pipeline: `compute_stream` separate from
  `copy_stream`, cuBLAS handle re-bound per slot, slot reuse barrier.

The pre-implementation VRAM-budget audit (after M8.7.1.r landed)
showed the staging + transient working-set peaks above the
available headroom on the dev hardware:

| Component | Bytes (worst case, 13B Path B) |
|---|---:|
| 2 × BF16 staging slots (persistent) | 270 MiB |
| F32 transient (per-matmul, BF16→F32 upcast) | 270 MiB |
| F32 activation upload (`d_a`) | ≤ 50 MiB |
| F32 output download (`d_out`) | ≤ 80 MiB |
| **Peak working set** | **~670 MiB** |

The free working-set budget at the smoke configuration is
~540 MiB (`8 GiB VRAM − 1 GiB M6 headroom − 6.46 GiB residents`).
The peak is **~130 MiB short** of available headroom on the dev
hardware. Realising M8.7.1.b/c without changing the residency
budget would either OOM at runtime or force the prereq to bump
`DISK_PIPELINE_STAGING_BYTES` to ~540 MiB — which means giving
back ~270 MiB of resident weights to Disk (≈3 fewer VRAM
tensors, ~115 ms more NVMe per forward) before the pipeline can
amortise the rest.

The projected speedup of M8.7.1.b/c was **~1.5–2× over M8.7.1.a**
based on the M8.0b two-buffer bench (1.56× sequential → pipelined
on the FFN-down shape). On the M8.7.1.a baseline of 10.22 s/forward,
that would land between 5.1 and 6.8 s/forward — close to the
original M8.7 close criterion of "~5–7 s/tok". But the residency
trade-off + ~600 lines of new infra carry real risk (CUDA
synchronisation correctness, OOM at the F32-transient boundary,
operator visible failure modes during transient pressure).

The closed M8.7.1.a + M8.7.1.r path delivers the **24 %** of
the speedup needed to comfortably clear the M8 baseline, with
zero residency trade-off and a small surface area
(~400 + 80 lines), and leaves the stream-aware kernel ready
for a future operator who has spare VRAM headroom (e.g. a
24 GB-class consumer GPU or a Hopper class card) to flip
M8.7.1.b/c on without further refactor work.

The natural next milestone is **M9 (INT8 quantisation)**, which
attacks the same throughput frontier from a different vector
(halve again from 2 to 1 byte per weight). Under INT8, the
Llama 2 13B's BF16 26 GiB drops to ~13 GiB, fitting entirely in
the 8 GiB VRAM + 32 GiB RAM box without any Disk overflow at
all — eliminating the M8.7 problem space rather than optimising
within it.

---

## Architectural decisions (D81+)

- **D81 — `DISK_PIPELINE_STAGING_BYTES` = 2 × 135 MiB,
  conditional on disk overflow.** The reservation is paid only
  when the plan would otherwise overflow to Disk; 7B-class
  plans that fit fully in VRAM/RAM see no penalty. Encoded in
  `tier_plan::plan` as a two-pass dry-run + reservation. Sized
  for the worst-case Llama 13B FFN-down BF16 weight (135 MiB)
  doubled for the two-slot pipeline. Larger weights (lm_head:
  312 MiB BF16) are excluded by the `tier_plan::is_gpu_eligible`
  filter (the planner never puts them on Vram, and M8.7 skips
  them at `cuda::disk_prefetch::MAX_PREFETCH_BYTES`).
- **D82 — M8.7.0 dispatch contract = bit-exact M4.7.2.e
  numerics.** The Disk → GPU streaming path reuses the M8.4c
  Path B kernel; weight stays BF16 in VRAM (preserves M8 capacity
  doubling), upcast to F32 transient on-device per-matmul, F32
  cuBLAS GEMM. Single-op drift envelope 3.28e-3 vs ADR-004
  threshold 0.5 — same envelope as M8 resident matmuls just
  sourced from NVMe.
- **D83 — `Graph::execute_inference` is the inference entry
  point under M8.7.** The M8.7.0 hook is gated on `!record_tape`
  because the streaming path drops the host F32 weight at scope
  end and backward gradient computation has nothing to read.
  `LlamaPipeline::generate_greedy` and the killer demo migrate
  to the inference variant; training callers keep
  `graph.execute(...)` (record_tape=true) and route through the
  legacy host path.
- **D84 — `exec_gpu_segment` (APX 4.3) is superseded.** The
  M8.7 guard skips it whenever (a) any operand is `Cuda(non-F32)`
  or `Disk(_)`, or (b) `ATENIA_M8_7_ENABLED=1`. The modern
  dispatch chain (`try_gpu_matmul` → `cuda_matmul_bf16_inplace` →
  `cuda_matmul_disk_streamed_bf16`) handles every M8 / M8.7
  operand correctly. Full removal of `exec_gpu_segment` is a
  follow-up tracked in `docs/TECH_DEBT.md`.
- **D85 — Tier-aware loader is the default since `afaa975`.**
  D74 from `HANDOFF_M6.md` (`ATENIA_TIER_AWARE_LOADER=1` opt-in)
  is superseded. The deprecated flag is recognised with a
  warning during a grace period; `ATENIA_LEGACY_LOADER=1` is
  the new opt-out.
- **D86 — Single-slot CPU prefetch is sufficient at this stage.**
  M8.7.1.a uses one host-resident NVMe prefetch in flight at any
  moment, indexed by `DiskTensorHandle::path()`. The executor's
  one-step lookahead `Graph::find_next_disk_bf16_handle_after`
  combined with the per-call kick-off-after-take pattern means
  every disk-streamed matmul (after the first) finds its bytes
  ready in RAM. 98.7 % hit rate observed empirically. Two-slot
  is reserved for M8.7.1.b/c when the GPU-side pipeline is
  enabled.

---

## Gaps closed

- **The 13B disk-tier matmul tax dropped from CPU AVX2 to GPU.**
  Pre-M8.7, 197 weights/forward were decoded on CPU (~140 ms /
  weight = ~28 s/forward). Post-M8.7, 154 weights/forward stream
  to GPU at ~135 MiB / 38 ms NVMe + ~88 ms upload + GEMM, with
  the NVMe portion overlapped with the previous matmul's compute
  via M8.7.1.a (98.7 % hit rate).
- **`Graph::execute_inference` provides the inference fast
  path.** Record-tape gating on backward-incompatible kernels
  (M8.7.0 streaming, future M8.6 BF16 KV cache, etc.) becomes a
  drop-in pattern.
- **`exec_gpu_segment` deprecation contract documented.** The
  guard already disables the legacy path under M8.7; full removal
  is tracked.

## Gaps left open

- **M8.7.1.b/c (async H→D pipeline + dedicated streams).** Code
  scaffolding exists (`cuda_matmul_bf16_inplace` accepts a
  `stream` parameter; `disk_prefetch` is single-slot but readily
  extended). Hardware blocker on the dev box (free VRAM working
  set ~540 MiB vs ~670 MiB peak). Re-enabling on a 24 GB-class
  GPU is a ~600-line pure addition.
- **M8.6 (BF16 KV cache, D62).** Independent ~1-day side path
  documented in `HANDOFF_M8.md`. Saves 1.6 GiB of RAM in seq_len
  2048 on 13B. Unblocked, not gated by M8.7.
- **`exec_gpu_segment` removal.** The path is unused under M8.7
  but still compiled in. APX 4.x test surface needs an audit
  before removal. Tracked in `docs/TECH_DEBT.md`.

---

## How to resume on M8.7.1.b/c (if revisited)

If a future operator wants to flip M8.7.1.b/c on (e.g. on a
24 GB-class GPU where the 540 MiB free-working-set bound
disappears):

1. Bump `DISK_PIPELINE_STAGING_BYTES` in `src/gpu/tier_plan.rs`
   to a value that covers `2 × max_weight_bytes + max_F32_transient
   + d_a_max + d_out_max` (~810 MiB on the Llama 13B shapes).
2. Add a new module `src/cuda/disk_pipeline.rs` (or extend
   `disk_prefetch.rs`) with a `TwoBufferPipeline` singleton
   carrying `copy_stream`, `compute_stream`, two persistent
   `pinned_host` buffers (`cudaMallocHost`), two persistent
   `vram_staging` `TensorGPU`s (BF16, sized to
   `DISK_PIPELINE_STAGING_BYTES / 2`), and CUDA events for
   `upload_done` / `compute_done` per slot.
3. Refactor `cuda_matmul_disk_streamed_bf16` to dispatch via
   the pipeline: `cudaStreamWaitEvent` for the slot's previous
   compute, copy `Vec<u8>` → pinned host (synchronous host
   memcpy), `cudaMemcpyAsync` H→D on `copy_stream`,
   `cudaEventRecord(upload_done)`, `cudaStreamWaitEvent(compute_stream,
   upload_done)`, dispatch `cuda_matmul_bf16_inplace(...,
   compute_stream)` (already accepts the stream parameter post
   M8.7.1.r), `cudaEventRecord(compute_done, compute_stream)`.
4. The `cuda_matmul_bf16_inplace` body should release its F32
   transient as soon as the GEMM completes (before the D→H
   download) to keep the worst-case working set bounded.
5. Smoke target: 13B forward seq=4 ≤ 6 s.

The investigation that produced this plan is preserved in this
HANDOFF and in the M8.7.1.b/c VRAM-budget analysis above.

---

## Validation gate at handoff

- `cargo test --lib`: **189/189** passed, 0 failed, 2 ignored
  (M8.4c and M8.7.0 CUDA-only `#[ignore]` tests, run manually
  with `--ignored`).
- 13B killer demo smoke: 10.22 s forward at seq=4, 152 / 154
  prefetch hits, argmax bit-exact with M8 baseline.
- 13B `atenia generate` smoke: 20.7 s/tok over 5 tokens, text
  coherent ("Hello, how are you?" → " Hello! I'").
- `docs/TECH_DEBT.md`: tracking entries for `exec_gpu_segment`
  deprecation, M8.7.0 routing follow-ups (closed by M8.7.1.a),
  and `cargo test --tests` non-determinism.

---

## Operator quickstart

```powershell
# Hardware: 32 GiB Windows box, NVIDIA RTX 4070 Laptop (8 GiB), NVMe.
# Tier-aware is the default since afaa975 — no flag required.

# Smoke 13B with M8.7 active (recommended config):
$env:ATENIA_M8_BF16_KERNEL = "1"
$env:ATENIA_M8_7_ENABLED   = "1"
$env:ATENIA_DISK_TIER_DIR  = "D:\atenia-m8-cache"

cargo run --release --bin atenia -- generate `
    --prompt "Hello, how are you?" `
    --model D:\Atenia\models\llama-2-13b-chat `
    --max-tokens 5

# Killer demo (richer counters / per-position argmax):
cargo run --release --example llama2_13b_demo

Remove-Item Env:ATENIA_M8_BF16_KERNEL
Remove-Item Env:ATENIA_M8_7_ENABLED
Remove-Item Env:ATENIA_DISK_TIER_DIR
```

Expected: 13B 5-token generation in ~100 s wall-clock
(20.7 s/tok), text " Hello! I'" matches the M8 baseline. Killer
demo at seq=4 in ~10.2 s forward, prefetch hit rate ≥ 98 %, no
panics. Without `ATENIA_M8_7_ENABLED=1`, the same smoke runs
through the M8 disk-tier CPU-AVX2 path at the M8 baseline
~27 s/tok — useful for A/B comparison.
