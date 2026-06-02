# HANDOFF — MOE-PERF-2: GPU expert FFN (`NumericPolicy::Fast`) — implemented, certified, **but slower** (STOP + why)

Wires the expert FFN matmul to CUDA under `NumericPolicy::Fast`, with a tolerance
certificate and an automatic CPU fallback. The path is **correct, certified, and
fallback-safe** — but on the real workload it is **~15 % slower** than CPU. Per
the series rule ("if CUDA does not improve: STOP and explain exactly why"), this
documents the implementation, the measurement, and the precise reason, and scopes
the GPU approach that *can* win. Predecessor: `8c9ca4c`.

## FASE 1 — Audit (what to wire)

`MoeDenseExpert::forward` computes `gate/up/down` as CPU matvecs (`dense.rs`),
**bypassing** the GraphBuilder GPU dispatch — which is why enabling the GPU in
MOE-PERF-1 gave 0 speedup. The wiring point is `cuda/matmul.rs::cuda_matmul(a, b,
m, k, n)`: CPU-in/CPU-out, uploads/launches/downloads internally, and **falls
back to the exact CPU f32 matmul** (`a.matmul(b)`) on any CUDA failure or in a
CUDA-less build. The expert weights (11.5–46 MB) are **under** the 64 MiB pool
block limit, so the pooled GPU path actually runs (unlike the 13B dense case).

## FASE 2-4 — Design + wiring + fallback

- `NumericPolicy::ffn_uses_cuda()` → only `Fast`.
- `dense.rs::matvec_cuda(w, rows, cols, x)`: builds CPU `Tensor`s and calls
  `cuda_matmul(W[rows,cols], x[cols,1], rows, cols, 1)` → `y[rows]`. f32.
- `matvec_policy` now dispatches: `Certified` → f64 CPU, `Strict` → f32 CPU,
  `Fast` → `matvec_cuda` (GPU f32, **CPU-f32 fallback**). The router stays f64.
- **Fallback (FASE 4):** `cuda_matmul` returns the exact CPU f32 result on any
  GPU error / no-GPU / CUDA-less build — so `Fast` degrades to `Strict`
  numerically and **never fails**. Default policy is still `Certified`.

## FASE 5 — Certification

Unit (`dense::matvec_backends_certify_against_f64`): the three backends
(f64 / f32-CPU / GPU-or-CPU-f32) agree within `STRICT_LOGIT_TOLERANCE`. Real
model: token ids **identical** across all three policies → `Fast` **certifies**
(tokens match, argmax agrees).

## FASE 6-7 — Real benchmark + VRAM (Qwen1.5-MoE, prompt `22,25,29`, max-new 2)

| Policy | wall | prefill exec | decode exec | token ids | VRAM peak |
|---|---|---|---|---|---|
| Certified (f64 CPU) | **180 s** | 76.5 s | 17.1 s | `16, 15` | — |
| Strict (f32 CPU) | **179 s** | 75.0 s | 17.7 s | `16, 15` | — |
| **Fast (f32 GPU)** | **207 s** | 96.6 s | 23.7 s | `16, 15` | **625 MiB** |

- **Fast is ~15 % SLOWER** (207 vs 179–180 s); GPU was genuinely used (VRAM
  625 MiB / 8 GB), and the output is **certified** (identical tokens).
- RAM: unchanged from the CPU path (experts still resolve to host f32 first).
- Throughput: ~0.011 tok/s (Fast) vs ~0.011 tok/s (CPU) — both load-dominated at
  max-new 2; the generation-compute delta is the +20 s prefill-exec on Fast.

## FASE 8 — Validation

- `moe::` lib suite (156) green; `tests/numeric_policy_test` (override mechanism,
  isolated process) green; full MoE regression bit-exact under the default
  `Certified`. (Fixed a test race: the policy-mutating tests now run in a
  separate process / via direct backend calls so they never race the in-process
  FFN-equality tests on the global policy.)

## Why CUDA does NOT improve here (the STOP, exactly)

The expert FFN is, per token, three **GEMVs** (`M = 1`) over **transient
weights** — each routed expert is resolved from the disk tier, used **once**, and
dropped. So every GPU matvec pays:

1. a **host copy** of the weight (`w.to_vec()` to build the Tensor), then
2. a **full H→D upload** of the 11.5–46 MB weight over PCIe, then
3. a tiny **`M=1` GEMV** (memory-bound, ~microseconds of compute), then
4. a **D→H download** of the result.

The upload (2) alone is ~the same number of bytes the **CPU** matvec reads from
already-resident RAM — but the CPU does *only* that read, on 24 cores, with no
PCIe hop and no host copy. So per-token GPU offload is **transfer-bound and
strictly loses**. VRAM (625 MiB) confirms the GPU ran; it just can't amortise the
upload because **the weight is never reused** (M=1, used once).

The GPU wins only when **either** the weights are **resident in VRAM and reused**
(amortising the upload over many tokens) **or** the GEMM has a **large `M`**
(batched/multi-position so compute hides the transfer). Neither holds for the
current per-token, transient-weight expert FFN.

## Recommendation (next block — the GPU approach that *can* win)

`Fast` stays available (certified, fallback-safe) but is **not** the default and
**not** recommended for this single-stream workload. The GPU win requires:

1. **VRAM-resident reused weights** — keep the **shared expert** (already cached
   once per layer, MOE-PROD-6) **resident in VRAM** and run its FFN there every
   token (upload once, reuse ~all tokens). Extend the `ExpertCache` pinned slot
   to a VRAM tensor. This is the highest-ROI GPU step.
2. **Batched prefill** — run the prefill's multiple positions as one `M=seq`
   GEMM per expert (compute hides transfer) instead of per-position GEMVs.
3. **Disk→VRAM streaming for routed experts** (M8.7 `cuda_matmul_disk_streamed_bf16`)
   only if 1–2 prove the GPU GEMM beats the 24-core CPU for these shapes.

Until then, **`Strict` (f32 CPU) is the fast default** (certified, ~marginally
faster than Certified, no GPU overhead) and **`Certified` (f64) the reference**.

## Files

- `src/moe/numeric_policy.rs` — `ffn_uses_cuda`; pure tests only (mutating tests
  moved out).
- `src/moe/dense.rs` — `matvec_cuda`, `Fast` dispatch, direct-backend cert test.
- `tests/numeric_policy_test.rs` (new) — isolated override-mechanism tests.
- `docs/HANDOFF_MOE_PERF_2.md` (this) + `docs/STATUS.md`.
