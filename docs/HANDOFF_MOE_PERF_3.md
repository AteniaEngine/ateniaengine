# HANDOFF — MOE-PERF-3: shared-expert VRAM residency — **STOP (no ROI), with hard evidence**

The plan was to keep the shared expert resident in VRAM (it runs every token and
is already cached, so it could amortise the GPU upload that sank MOE-PERF-2).
Before building the risky CUDA-resident path, this block **measured the ceiling**
of that optimisation — and the measurement says the ROI is **essentially zero**.
Per the explicit stop criterion ("if the resident shared expert does NOT produce
a measurable improvement: STOP and explain exactly why; do not continue without
ROI evidence"), this is a documented STOP. Predecessor: `7101f67`.

## FASE 1 — Audit

- The GPU-resident primitives exist: `Tensor::ensure_gpu` / `zeros_new_cuda`,
  `cuda_matmul_inplace` (all-Cuda → device-pointer GEMM, no host traffic for
  resident operands), `cuda_fused_linear_silu`, `cuda_available` — all
  `#[cfg(atenia_cuda)]`. A residency cache is *technically* feasible.
- The shared expert lives in `ExpertCache.shared` (pinned, one per layer,
  MOE-PROD-6), resolved once and reused; under Fast its weight is currently
  re-uploaded every token (`matvec_cuda`).
- **What would stay resident:** the shared expert's gate/up/down
  (`[5632,2048]×2 + [2048,5632]` ≈ 138 MB f32/layer × 24 ≈ 3.3 GB VRAM).

## FASE 2 — Measure the ceiling BEFORE implementing (ROI gate)

Added bit-exact instrumentation (`CacheStats.shared_fwd_nanos` /
`routed_fwd_nanos`, reported by `ATENIA_MOE_CACHE_STATS=1`) that times the
**shared-expert `forward`** (the matmul VRAM residency would accelerate) and the
routed-expert `forward`, separately from the tier resolve.

Real Qwen1.5-MoE (prompt `22,25,29`, Strict CPU):

| max-new | wall | **shared fwd (matmul)** | routed fwd | shared % of wall |
|---|---|---|---|---|
| 2 | 185 s | **0.24 s** | 0.42 s | **0.13 %** |
| 8 | 237 s | **0.88 s** | 1.34 s | **0.37 %** |

**The shared-expert matmul is 0.24–0.88 s of a 185–237 s wall (< 0.4 %).** Even
if VRAM residency made it **instantaneous**, the maximum possible saving is
**< 1 s**. The prefill-exec (≈ 80 s) is **not** the expert matmul (shared +
routed = 0.66 s); it is the **disk-tier resolves** (297 misses reading 12.67 GiB
of NVMe at max-new 2), the GraphBuilder attention/lm_head, and the per-position
graph execution.

## Decision — STOP (evidence-based)

VRAM residency of the shared expert **cannot produce a measurable improvement**:
its entire compute is < 0.4 % of the wall. Building the CUDA-resident path
(device-pointer FFI, per-token x/h transfers, VRAM lifecycle, `cfg(atenia_cuda)`
gating + CPU fallback) to chase **< 1 s** would add real correctness/maintenance
risk for no benefit — violating "correctness > performance" and the explicit "no
ROI evidence → stop" rule. So MOE-PERF-3 stops here, **with the measurement as
the deliverable** (not an unmeasured assumption).

## Why the whole "expert FFN compute" direction is a dead end here

MOE-PERF-1/2/3 converge on one finding: **the MoE expert FFN compute (CPU *or*
GPU) is < 1 % of the wall.** The generation is **I/O- and load-bound**, not
compute-bound:

- **Load** (~34 s reconstruct) — addressed by MOE-PROD-5/7.
- **Per-token disk-tier resolves** (reading routed expert weights from NVMe; ~13
  GiB at max-new 2) — the dominant *generation* cost (MOE-PROD-3/6 domain).
- **GraphBuilder attention/lm_head + per-position execution** — the rest.
- **Expert matmul (shared + routed): < 1 s** — negligible.

No expert-matmul precision/offload change (f32, GPU, Tensor Cores, residency)
can move a wall that is < 1 % matmul. The earlier `Strict` win (~13 %) came from
f32 *vectorisability of the FFN plus reduced memory traffic*, not the matmul
FLOPs — and it is already in.

## Recommendation (next, if MoE perf is pursued — but verify ROI first)

The only remaining high-ROI levers target **I/O and load**, not compute:

1. **Bigger / smarter expert cache** — at max-new 8 the hit ratio already rose to
   55 % (vs 22.7 % at max-new 2); a larger or prefetching cache cuts the
   dominant per-token NVMe resolves. Measure RAM vs hit-ratio.
2. **Fewer tier files** — consolidate the 4659 per-tensor files into per-layer
   blobs to cut Windows per-file open overhead inside the resolves.
3. **GraphBuilder attention/lm_head** — profile whether these (already
   GPU-dispatchable) are a meaningful slice; if so, a `Fast` GPU path *there*
   (resident lm_head, large `M=seq` GEMM) could help — unlike the per-token
   expert GEMVs.

Each must pass the same **ROI-ceiling measurement** this block established before
any risky implementation. `Strict` (f32 CPU) remains the fast default;
`Certified` (f64) the reference; `Fast` available but not recommended.

## Files

- `src/moe/residency.rs` — `CacheStats.shared_fwd_nanos`/`routed_fwd_nanos` +
  timers in `forward_cached`.
- `src/moe/graph_op.rs` — aggregate the new stats.
- `src/bin/atenia.rs` — report `MoE fwd compute: shared=.. routed=..`.
- `docs/HANDOFF_MOE_PERF_3.md` (this) + `docs/STATUS.md`.

No numeric/behaviour change; bit-exact; default `Certified` unchanged.
