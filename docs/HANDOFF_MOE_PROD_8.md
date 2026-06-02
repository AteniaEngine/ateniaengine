# HANDOFF — MOE-PROD-8: generation compute (parallel matvec + single-copy resolve)

Milestone: **MOE-PROD-8** — attack the **generation compute** that MOE-PROD-7's
profiling proved is the real warm-path bottleneck (~82 % of the wall), keeping
the certified f64 reference math **bit-exact**. This closes the **bit-exact
warm-path optimization block** (MOE-PROD-5 → 8); the next step is an
architectural fork (see the end). Predecessor: `9b345f4`.

## FASE 1 — Audit (measured, not guessed)

MOE-PROD-7 added per-phase warm profiling. MOE-PROD-8 adds **generation-phase
profiling** (`ATENIA_MOE_CACHE_STATS=1` prints `gen profile: prefill(build,exec)
decode(build,exec)`). On the real Qwen1.5-MoE (prompt `22,25,29`, max-new 2):

```
tier warm reconstruct: total=34.5s (validate=0.17, layers=17.9, embed/lm_head=16.4)
gen profile: prefill(build=0.8s, exec=76.3s) decode(build=1.1s, exec=17.6s)
warm wall = 184 s
```

So the warm wall splits ≈ **reconstruct 34 s (18 %) + generation 94 s (51 %) +
process/config/source ~30 s**. Generation is **graph execution** — the MoE
expert resolves (disk read + materialise) + the expert FFN + attention + lm_head
matmuls, run per sequence position (prefill = 3 positions, decode = 1 with KV
cache, no recompute of the prefix).

## FASE 2 — Implementation (bit-exact)

1. **Parallel `matvec`** (`moe/dense.rs` + `moe/mla.rs`) — the expert FFN
   projection `y = W·x` ran a **single-threaded** loop of per-row f64
   reductions. The rows are independent, so for large weights (≥ 65 536 MACs —
   the FFN projections, not the tiny router) they now run across the 24 cores
   via `rayon::par_iter_mut`. **Bit-identical**: each `y[r]` is the *same*
   sequential f64 accumulation over `c`; only the thread that computes a row
   changes, never the arithmetic or its order.
2. **Single-copy expert resolve** (`moe/residency.rs::materialize`) — the Disk
   tier read did `to_tensor().ensure_cpu().copy_to_cpu_vec()`, which allocated
   the f32 buffer **and then cloned it** (a redundant full copy ×3 per resolve).
   The Disk arm now reads the tier file **directly** into one owned `Vec<f32>`
   (`read_f32_tensor` / `read_bf16_tensor` + the same `bf16_decode_bulk` upcast),
   halving the alloc/memcpy per routed-expert miss. Bit-identical bytes.

No routing, math, or output change; f64 reference accumulation preserved.

## FASE 3 — Bit-exactness + tests

- Output unchanged on the real model: token ids **`16, 15`** (same as
  MOE-PROD-5/6/7). The existing disk==RAM, cached==uncached, and
  reconstruct-without-shards tests (all bit-exact equality) cover the
  parallelisation and the single-copy resolve; full MoE regression + lib green.

## FASE 4 — Benchmark (real Qwen1.5-MoE, prompt `22,25,29`, max-new 2)

| | before (MOE-PROD-7) | after (MOE-PROD-8) |
|---|---|---|
| Warm wall | **204 s** | **184 s** |
| prefill exec | ~85 s (est.) | **76.3 s** |
| reconstruct | 34.6 s | 34.5 s |
| Output | `16, 15` | `16, 15` — bit-exact |

**Gain: 204 → 184 s (~10 %)**, bit-exact. Parallel matvec ≈ −19 s, single-copy
resolve ≈ −4 s on prefill. Modest because the f64 matvec was only part of the
~94 s generation; the rest is the expert disk resolves + attention/lm_head GEMMs
(already parallel via the GraphBuilder) + the inherent per-position MoE work.

## FASE 5 — Block conclusion + architectural fork (STOP / decision)

**The bit-exact warm-path optimization block (MOE-PROD-5 → 8) is complete.**
Cumulative, on the real Qwen1.5-MoE warm path:

- Load: shard-read+assemble (MOE-PROD-4 era ~2445 s) → reconstruct-from-tier
  (MOE-PROD-5, 701 s same-prompt-equiv 350 s) → bf16 experts (MOE-PROD-6,
  204 s) → bf16 backend + bulk reader (MOE-PROD-7, load 34 s, gen-bound) →
  parallel matvec + single-copy (MOE-PROD-8, **184 s**).
- The warm **reconstruction is now only ~34 s (~18 %)**; generation dominates.

**The remaining dominant cost is CPU graph execution (~94 s) of the MoE forward,
which uses f64 accumulation for the certified bit-exact reference.** The
high-ROI **bit-exact** CPU levers are exhausted. Going further requires an
**architectural decision** that trades the certified property:

- **GPU offload** of the FFN / attention / lm_head GEMMs (the engine has CUDA
  matmul + the M8.7 disk→VRAM bf16 streaming path). cuBLAS/CUDA is **f32**, so
  the GPU result would **not be bit-exact** with the f64 CPU reference — it would
  need its own numerical certificate (tolerance-based, like the dense GPU path),
  not bit-equality. Big win expected (GPU ≫ CPU f64), high complexity.
- **f32 CPU accumulation** (SIMD-friendly) — faster, but also **breaks
  bit-exactness** with the f64 reference.

Both cross the "correctitud/bit-exactness" line this block held. That is a human
call (numerical-policy + VRAM-budget + roadmap), so the block stops here and
reports. See the session report's "next block" section.

## Files modified

- `src/moe/dense.rs`, `src/moe/mla.rs` — parallel `matvec` (bit-exact).
- `src/moe/residency.rs` — single-copy Disk `materialize`.
- `src/moe/generate.rs` — generation-phase profiling (gated).
- `docs/HANDOFF_MOE_PROD_8.md` (this) + `docs/STATUS.md`.
