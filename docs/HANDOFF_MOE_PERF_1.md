# HANDOFF — MOE-PERF-1: CPU f32 result + CUDA feasibility audit

Companion to [HANDOFF_NUMERIC_POLICY_1.md](./HANDOFF_NUMERIC_POLICY_1.md). The
policy/certificate framework landed there; this records the **first performance
backend** (`Strict` = f32 CPU) measured on the real model, and the **CUDA
feasibility audit** that scopes the next block.

## 1. CPU f32 (`Strict`) — measured

Real Qwen1.5-MoE-A2.7B (prompt `22,25,29`, max-new 2, CPU, tier reused):

| Policy | wall | prefill exec | tokens | drift |
|---|---|---|---|---|
| Certified (f64) | 206 s | 97.3 s | `16, 15` | reference |
| **Strict (f32 FFN)** | **180 s** | **75.7 s** | `16, 15` | **tokens identical** |

- **Speedup: ~13 % wall / ~22 % prefill-exec**, from f32 accumulation on the
  three expert-FFN projections (the router stays f64 → routing identical).
- **Drift: zero at the token level** (`16,15` identical) → passes the
  certificate's decisive criterion; logit-level drift is bounded (unit-tested).
- Modest because the FFN matvec is already rayon-parallel (MOE-PROD-8); f32 adds
  vectorisability on top. The expert **disk resolves** and the per-position graph
  execution are the rest of prefill-exec and are unaffected by precision.

## 2. CUDA feasibility audit (FASE 5)

**Available:** `atenia_cuda` compiles (CUDA v13.2), RTX 4070 Laptop, **~7.95 GB
VRAM free**. Existing infra: `cuda/matmul.rs` (GEMM), `cuda_matmul_disk_streamed_bf16`
(M8.7 disk→VRAM bf16), `gpu/tier_plan.rs` (placement), apx4/apx4_5 GraphBuilder
GPU dispatch, the MOE-PROD-6 **bf16 expert tier** (perfect streaming source).

**The blocking finding:** enabling the GPU on the MoE generate path gave **no
speedup (193 s vs 184 s CPU)**. Reason: the **expert FFN — the dominant compute —
does not flow through the engine's GPU matmul dispatch.** It is a direct CPU call
(`MoeDenseExpert::forward` → `dense.rs::matvec`), not a GraphBuilder op, so the
apx4 GPU dispatch never sees it. Only attention/lm_head go through GraphBuilder,
and those are a small fraction (and may themselves fall back to CPU for these
sizes).

**What to offload first (priority, given the audit):**
1. **Routed + shared expert FFN GEMM** — the dominant cost; stream the bf16 tier
   weights disk→VRAM via the M8.7 path (no host f32 materialise — also kills the
   MOE-PROD-8 resolve cost). Highest ROI.
2. **lm_head** — one big `[seq,2048]×[151936,2048]ᵀ`; resident in VRAM (~0.3 GB
   f16); easy, already GraphBuilder.
3. **Attention** q/k/v/o — small per token; resident; lower marginal ROI.

**VRAM budget (8 GB):** resident = attention + lm_head + the 24 shared experts
(~1.7 GB f16) ≈ fits with headroom; routed experts **streamed** from the bf16
tier (double-buffered). Open question the spike must answer: does NVMe→VRAM
streaming **hide under** the GEMM, or become the new bottleneck?

## 3. Why CUDA is NOT implemented in this block (STOP — explained)

Per the series rule "correctitud > performance" and "only one or two CUDA items
if the block grows", a real CUDA offload is **a full block, not a safe spike
alongside the policy framework**, because it requires *all* of:

1. A **GPU GEMM path for the expert FFN** that does not exist yet (wire
   `MoeDenseExpert::forward` / the resident `resolve+forward` to `cuda_matmul`
   or the disk-streamed kernel) — the experts currently bypass the GPU entirely.
2. **VRAM residency + streaming management** under the 8 GB budget (what stays
   resident, what streams, double-buffering, eviction).
3. A **CPU fallback** for every offloaded op (no-CUDA build, OOM, kernel error).
4. A **`Fast`-tier tolerance certificate** (cuBLAS f32/TF32 ≠ f64 — a *new*,
   looser certificate, validated on a real prompt set).

Doing this safely needs its own audit→spike→cert→benchmark cycle. The
NUMERIC-POLICY-1 framework (policy selection + `PolicyCertificate` + fallback)
delivered here is **exactly the foundation** that block reuses; the bf16 tier is
its streaming source. So the stop is at a clean seam, not a dead end.

## 4. Recommendation (next block = MOE-PERF-2: GPU expert FFN)

- **MP-2a:** wire the expert FFN GEMM to `cuda_matmul` behind `NumericPolicy::Fast`
  (resident weights first, smallest viable spike), CPU fallback, `Fast`
  certificate. Measure speed + VRAM + token drift vs Certified.
- **MP-2b (if 2a wins):** switch the routed experts to the **disk-streamed bf16**
  kernel (M8.7) to remove both the CPU GEMM and the host resolve; measure the
  streaming-vs-compute overlap (the open question).

If MP-2a shows the GPU GEMM does not beat the 24-core f32 CPU path for these
shapes (single-token, hidden 2048), report that and keep `Strict` CPU as the
fast default.
