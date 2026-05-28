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

## End-to-end TinyLlama result

<!-- AQS5_E2E_RESULTS -->
The AQS-4 harness (`tests/aqs4_end_to_end_test.rs`) was extended to feed
real GPTQ its `[S, K]` activation matrices. The heavy run
(`aqs4_tinyllama_policy_comparison`, `#[ignore]`) compares certified /
bf16 / plain_int8 / awq / hybrid / **gptq (real)** under the real F64
fixture.

Because real GPTQ on the full model takes ~5–6 h on CPU, the end-to-end
GPTQ row is produced by a long background run. The fast policies
re-confirm the AQS-4 numbers (AWQ 0.889 argmax 4/4, hybrid 0.832, INT8
1.261, BF16 ≈ certified). **The real-GPTQ end-to-end drift is reported in
the run log and folded in here when the run completes — no number is
fabricated.**

> Status at commit time: real-GPTQ end-to-end run **in progress** (CPU
> `O(K²N)` cost, ETA ~5–6 h). Implementation + unit correctness are
> complete and committed; the end-to-end verdict row is appended in a
> follow-up once the run finishes. Per scope rule 14/15: honest partial
> delivery, no fabricated results.

## Verdict (interim)

* **Correctness: achieved.** Real GPTQ is implemented faithfully and
  unit-validated, including the functional proof that it beats plain INT8
  on output error for structured activations — exactly where the AQS-3
  surrogate failed.
* **Plateau question:** pending the end-to-end number. If real GPTQ lands
  below AWQ's 0.889 it is the new best weight-only policy; if it still
  exceeds ADR-004's 0.5, the weight-only plateau holds and the honest
  recommendation is to pivot AQS to *certified search + verifiable
  manifests* over AWQ (the audit's real differentiator) rather than chase
  ADR-004 strict further.
* **Performance:** real GPTQ needs BLAS/GPU to be practical at model
  scale; that optimisation is explicitly out of AQS-5 scope.

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
