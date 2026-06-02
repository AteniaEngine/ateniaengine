# HANDOFF — NUMERIC-POLICY-1: explicit numeric policy + tolerance certification

First block of the **NUMERIC-POLICY / MOE-PERF** series. It makes the MoE
compute precision an **explicit, selectable, certifiable** choice instead of a
hard-coded f64, **without breaking** the certified f64 path. Predecessor:
`72ce2c5`.

## FASE 1 — Audit (bit-exact vs tolerance, CUDA, risks)

- **Bit-exact today (`Certified`):** the CPU MoE forward — expert FFN
  (`dense.rs` `matvec`, f64), router softmax (f64), SwiGLU (f64), the
  GraphBuilder attention/lm_head on CPU. This is what the MOE-PROD series
  certified (reproducible `16,15`).
- **Tolerance candidates:** the **expert FFN** matmul (f32 — the bulk of MoE
  compute); later attention/lm_head (GPU f32/TF32) and the routed-expert GEMM on
  GPU. **Not** the router matmul — an f32 router could flip a top-k pick, which
  is a *different computation*, not a rounding difference.
- **CUDA infra present:** `cuda/matmul.rs` (GEMM), `cuda_matmul_disk_streamed_bf16`
  (M8.7 disk→VRAM), `gpu/tier_plan.rs`, the apx4/apx4_5 GraphBuilder GPU
  dispatch. `atenia_cuda` compiles here (CUDA v13.2, RTX 4070, ~8 GB free).
- **Empirical CUDA finding:** enabling the GPU gave **no speedup** (193 s vs
  184 s CPU) because the **expert FFN bypasses the GPU dispatch** — it is a
  direct CPU call in `dense.rs`. GPU offload of the experts therefore needs
  *explicit wiring* (a separate block — see HANDOFF_MOE_PERF_1.md).
- **Risks:** f32 routing drift (→ router stays f64); f32 FFN accumulation drift
  over wide reductions (→ certify by tolerance + token equality); GPU f32/TF32 ≠
  f64 (→ a looser `Fast` certificate, future); silent precision downgrade (→
  policy is logged, Certified is default + fallback).

## FASE 2 — NumericPolicy (`src/moe/numeric_policy.rs`)

`NumericPolicy { Certified, Strict, Fast }`:

- **Certified** — f64 accumulation, bit-exact reference, **default and fallback**.
- **Strict** — f32 accumulation on the **expert FFN** only; bounded drift.
- **Fast** — reserved for the GPU/TF32/BF16 path (MOE-PERF); currently == Strict.

Resolved from `ATENIA_NUMERIC_POLICY={certified,strict,fast}` (cached once),
overridable in-process by `set_numeric_policy` (the cert harness runs the same
generation under two policies). **Any doubt → Certified** (unknown/unset →
bit-exact). The active policy is logged (`ATENIA_MOE_CACHE_STATS=1`).

Wiring: `dense.rs` gains `matvec_f32` + `matvec_policy`; `MoeDenseExpert::forward`
uses `matvec_policy` for gate/up/down. The router keeps f64 `matvec`. This is the
**resident/disk-tier** generation path too (`resolve → MoeDenseExpert::forward`),
so the policy reaches real MoE generation. Attention/lm_head/softmax/SiLU
unchanged in this block.

## FASE 3 — Tolerance certification

`PolicyCertificate::compare(reference_rows, candidate_rows, ref_tokens,
cand_tokens)` (reuses the f64 `NumericalMetrics`): aggregates per-token-logits
`max_abs_diff` / `mean_abs_diff` / `rmse`, the **argmax-match rate**, and whether
the **generated token ids are identical**. `passes(τ)` requires **every** row's
argmax to agree, the **tokens to be identical**, and `max_abs_diff ≤ τ`
(`STRICT_LOGIT_TOLERANCE = 0.5`, published, tightenable). A policy that fails →
the caller uses **Certified** (the fallback is the safe default, never a silent
wrong answer).

## FASE 4 — Tests

- `numeric_policy`: default-Certified + override round-trip; `from_str` unknown →
  None; certificate **passes on identical / bounded drift**, **fails on token
  mismatch / argmax flip**.
- `dense::expert_forward_certified_is_f64_strict_is_bounded`: a 64×256 expert —
  Certified is deterministic; **Strict certifies vs Certified** within tolerance.
- Full MoE regression (bf16 tier, residency, reconstruct, …) green with the
  default (Certified) path **bit-exact unchanged** (`16,15`).

## FASE 7 — Real benchmark (Qwen1.5-MoE, prompt `22,25,29`, max-new 2, CPU)

Same-session A/B:

| Policy | wall | prefill exec | token ids | certified? |
|---|---|---|---|---|
| **Certified** (f64) | 206 s | 97.3 s | `16, 15` | reference |
| **Strict** (f32 FFN) | **180 s** | **75.7 s** | `16, 15` | **yes (tokens identical)** |

**Strict is ~13 % faster wall (−26 s), ~22 % faster prefill-exec (−21.6 s on the
FFN), and produces the identical tokens `16,15`** → certified. (Run-to-run wall
variance is ~±10 % on this box; the same-session A/B and the prefill-exec delta
are the reliable signals.)

## Outcome

- `Certified` (default) unchanged and bit-exact; `Strict` available, certified,
  and meaningfully faster; `Fast` reserved.
- The certification machinery (metrics + token equality + fallback) is in place
  for every later (GPU) backend to reuse.

See **HANDOFF_MOE_PERF_1.md** for the CPU-f32 perf result + the CUDA feasibility
audit + the next-block scope.

## Files

- `src/moe/numeric_policy.rs` (new) — policy + certificate.
- `src/moe/dense.rs` — `matvec_f32`, `matvec_policy`, expert FFN wiring, test.
- `src/moe/mod.rs` — module; `src/bin/atenia.rs` — policy log.
- `docs/HANDOFF_NUMERIC_POLICY_1.md` (this), `docs/HANDOFF_MOE_PERF_1.md`,
  `docs/STATUS.md`.
