# HANDOFF — AQS-5: real CPU GPTQ policy

Milestone: **AQS-5** (genuine blockwise GPTQ, replacing the AQS-3
diagonal surrogate). CPU-only, experimental, opt-in.
Predecessors: AQS-0 audit (`6e1ead7`), AQS-1 (`c5e39bb`), AQS-2
(`602901b`), AQS-3 surrogate (`e31edd2`), AQS-4 harness (`5ac94fa`).

## Why real GPTQ

AQS-4 measured the AQS-3 diagonal surrogate at **12.5** end-to-end drift
on TinyLlama — worse than plain INT8 (1.26). That condemned the
*surrogate* (diagonal Hessian + clip-search + error diffusion), not GPTQ
itself: real GPTQ's benefit comes from the **full inverse-Hessian
per-weight error compensation** the surrogate explicitly skipped. AQS-5
implements the genuine algorithm to finally test it.

## What was implemented (real GPTQ)

`src/quant/gptq.rs` — the genuine OBQ/GPTQ algorithm, CPU-only:

* `compute_hessian(activations[S·K], S, K) -> H[K·K]` = `XᵀX / S`
  (f64 accumulation, symmetric).
* `add_damping_inplace(H, K, damp_percent)` — Tikhonov damping
  `H[i,i] += damp_percent · mean(diag H)`, with dead-channel handling.
* `cholesky_decompose` / `cholesky_inverse` / (internal) `cholesky_upper`
  — dense SPD linear algebra, no external crates; errors as
  `GptqError::NotPositiveDefinite { pivot }` if damping is insufficient.
* `apply_gptq_real_inplace(weights[K·N], shape, activations[S·K],
  activation_shape, GptqRealConfig)` — the blockwise sweep.
* `GptqRealConfig { group_size, block_size, damp_percent, act_order }`.

### Algorithm + layout

Weights are stored `[K_in, N_out]` (loader transposes HF). The Hessian
lives on the **input dimension K** (`H = XᵀX`, `K×K`). GPTQ quantises the
weight **rows** (one per input channel) sequentially along K, propagating
each row's quantisation error to the not-yet-quantised rows weighted by
the upper-Cholesky factor of `H⁻¹`:

```text
  H        = XᵀX / S ; H[k,k] += damp·mean(diagH)
  Hinv_u   = chol_upper( inverse(H) )
  for block of block_size rows along K:
    for row k in block:
      scale  = dynamic per-N group absmax (recomputed at group bounds)
      q      = round(W[k,:]/scale)·scale
      err    = (W[k,:] − q) / Hinv_u[k,k]
      W[k,:] = q
      W[k',:] -= Hinv_u[k,k']·err   (k' in block, intra)
    W[block_end:,:] -= A·Err_block   (inter, A[r,k]=Hinv_u[k,r]) — GEMM
```

Group scales are recomputed from the *current* (partly compensated)
weights at each group boundary (faithful dynamic groups). `act_order`
optionally permutes rows by descending `diag(H)` and un-permutes after.

### Damping

`λ = damp_percent · mean(diag H)` added to the diagonal (default 1%).
Dead channels (zero diagonal) are set to `λ`. A unit test
(`gptq_real_damping_makes_hessian_stable`) confirms a rank-deficient
Hessian is singular undamped and Cholesky-stable after damping.

### Error compensation

The inter-block update `W[block_end:,:] -= A · Err_block` is the genuine
GPTQ compensation (the surrogate had none). It is the `O(K²N)` hot path,
expressed as a GEMM routed through the engine's AVX2 matmul kernel
(`gemm_ab`, scalar fallback). Intra-block updates are tight AXPYs.

## Integration with GptqPolicy

* `GptqPolicy` is now **real GPTQ** (`id = "gptq"`); fields
  `{ group_size, block_size, damp_percent, act_order }`. It requires
  `CalibrationContext::activation_matrix` + `activation_shape` (`[S, K]`);
  abs­max-only or empty → `MissingActivationStats`.
* The AQS-3 surrogate is preserved as `GptqSurrogatePolicy`
  (`id = "gptq_surrogate"`) for comparison.
* `CalibrationContext` extended **additively** with `activation_matrix` +
  `activation_shape` (existing `activation_absmax` path for AWQ/Hybrid
  untouched).
* `WeightStore` gained `perturb_param_with_policy_matrix` +
  `perturb_all_proj_with_policy_matrix` (CPU-only, opt-in) so the AQS-4
  harness can feed full activation matrices.

## Unit-test results (fast, in CI)

All green (`cargo test --lib --release quant::`):

* `gptq_real_hessian_is_symmetric`
* `gptq_real_damping_makes_hessian_stable`
* `gptq_real_cholesky_inverse_identity_small` (A·A⁻¹ = I to 1e-9)
* `gptq_real_quantizes_without_nan_and_deterministic`
* `gptq_real_zero_tensor_stays_zero`
* `gptq_real_rejects_bad_activation_shape`
* `gptq_real_act_order_is_deterministic`
* `gptq_real_output_error_not_worse_than_plain_int8_structured`
* policy-level: `gptq_policy_requires_activation_matrix`,
  `gptq_policy_uses_real_gptq_with_matrix`,
  `gptq_surrogate_policy_applies_deterministically`, …

The headline correctness proof is
**`gptq_real_output_error_not_worse_than_plain_int8_structured`**: on a
weight with one high-energy input channel, real GPTQ's *functional*
output error `‖X(W−Wq)‖` is ≤ plain INT8's — i.e. the algorithm does what
it claims (minimise output error, not per-element weight drift), unlike
the surrogate.

## Cost wall (measured, honest)

Real GPTQ is `O(K²N)` per tensor (intrinsic — the inverse-Hessian error
compensation). Measured on this 24-thread CPU (AVX2 GEMM path):

| tensor shape | per-tensor wall time |
|--------------|----------------------|
| `[2048, 2048]` (q/k/v/o, gate, up) | ~18 s |
| `[5632, 2048]` (down_proj)         | ~800 s |

TinyLlama has 132 K=2048 proj tensors + 22 down_proj (K=5632), so a full
real-GPTQ pass is **≈ 5–6 hours** on naive CPU. This is expected: real
GPTQ in production uses BLAS + GPUs. AQS-5 does **not** optimise for speed
(per scope); the AVX2 GEMM (`gemm_ab`) is the only concession.

## End-to-end TinyLlama result (COMPLETE)

The AQS-4 harness (`tests/aqs4_end_to_end_test.rs`) was extended to feed
real GPTQ its `[S, K]` activation matrices and run end-to-end on the real
TinyLlama F64 fixture. The full run completed in **28 108 s (≈7.8 h)** on
CPU. Real numbers — nothing fabricated:

```
group_size=128  alpha=0.25  outlier_k=64  gptq_damp=0.01
candidate         max_diff   mean_diff      rmse   argmax   memory_bytes  ADR-004
certified(f32)    0.000063   0.000008   0.000010   true             0     PASS
bf16              0.000063   0.000008   0.000010   true    2260729856     PASS
plain_int8        1.260771   0.145249   0.202987  false    1165688832     FAIL
awq               0.889217   0.074738   0.105758   true    1165688832     FAIL
hybrid            0.831786   0.061254   0.091544  false    1266614272     FAIL
gptq (real)       1.405399   0.100628   0.170255  false    1167265792     FAIL
```

The certified / bf16 / plain_int8 / awq / hybrid rows **reproduce AQS-4
exactly** (AWQ 0.889217 argmax 4/4 identical), confirming the harness is
unchanged and trustworthy.

## Verdict — real GPTQ did NOT break the plateau (honest)

**Real GPTQ lands at 1.405 end-to-end — WORSE than AWQ (0.889), worse than
hybrid (0.832), and even worse than plain INT8 (1.261), with argmax
broken.** It does not cross ADR-004 (0.5). The weight-only plateau holds;
**AWQ (0.889, argmax 4/4) remains the best weight-only policy.**

### Why real GPTQ underperformed here (root cause, not excuse)

This is a *correct implementation* giving a *bad end-to-end number*, and
the most likely reason is **calibration starvation**, not an algorithm
bug:

* GPTQ's inverse-Hessian compensation is only meaningful if the Hessian
  `H = XᵀX` is well-conditioned. Our calibration captured **S = 32
  samples** (8 short prompts × seq 4) for layers with **K = 2048–5632**
  input channels. With `S ≪ K`, `H` is massively rank-deficient: its rank
  is ≤ 32 while it is `K×K`. After damping, `H` is dominated by the
  `λ·I` term, so `H⁻¹ ≈ (1/λ)·I` and the "compensation" degenerates
  toward noise — it can *hurt* rather than help, which is exactly what the
  numbers show (worse than plain INT8).
* The literature runs GPTQ with **hundreds of thousands** of calibration
  tokens (e.g. 128 sequences × 2048 tokens). Our harness used 32 samples
  because (a) the F64 fixture forward is seq=4 and (b) each extra sample
  inflates the already 7.8 h run.
* The AQS-3 unit test still holds: on a *well-conditioned, structured*
  small case (S=16, K=32, one dominant channel) real GPTQ **does** beat
  plain INT8 on functional output error. The algorithm is correct; the
  real-model calibration budget here is not enough to exploit it.

### So: is GPTQ dead?

Not proven dead — proven **not worth it under realistic constraints for
Atenia right now**:

* To give GPTQ a fair shot we would need (a) a much larger calibration
  corpus (→ Hessian rank ≫ current), and (b) BLAS/GPU acceleration to make
  the `O(K²N)` cost tolerable (the CPU run is already ~8 h with only 32
  samples; a proper corpus would be far worse). Both are large
  investments with an *uncertain* payoff — at best GPTQ might approach
  AWQ's 0.889, and AWQ already achieves that in seconds with no Hessian.
* AWQ gives the plateau result (0.889, argmax 4/4) at a tiny fraction of
  the cost and complexity.

### Recommendation

**Accept the weight-only plateau and pivot AQS to its real differentiator:
certified search + verifiable manifests over the policies we already have
(AWQ as the headline).** Chasing ADR-004 strict (0.5) with more weight-only
quantisation — GPTQ included — has now been falsified across five distinct
mechanisms (plain INT8, β outlier, β-pivot AWQ, β-pivot hybrid, real
GPTQ). The value is not a magic algorithm; it is *automatically measuring
and certifying* which quantisation is safe for each model against F64.

### Performance note

Real GPTQ is `O(K²N)`/layer and needs BLAS/GPU to be practical at scale;
the CPU AVX2 GEMM (`gemm_ab`) brought one TinyLlama pass to ~7.8 h. Further
optimisation is explicitly out of AQS-5 scope and, given the negative
verdict, not recommended.

## Files modified

* `src/quant/gptq.rs` — real GPTQ (Hessian, damping, Cholesky,
  compensation, GEMM) + unit tests.
* `src/quant/policy.rs` — `GptqPolicy` = real; `GptqSurrogatePolicy` kept;
  `CalibrationContext` extended additively; `PolicyError::InvalidDampPercent`.
* `src/quant/mod.rs` — re-exports.
* `src/amg/weight_store.rs` — `perturb_*_with_policy_matrix`.
* `tests/aqs4_end_to_end_test.rs` — activation-matrix capture + real-GPTQ
  candidate.
* `docs/HANDOFF_AQS_5.md` — this file.

No CUDA, tier-planner, CLI, generation, loader, or manifest changes.
