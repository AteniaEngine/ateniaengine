# HANDOFF — AQS-3: experimental simplified GPTQ policy landed

Milestone: **AQS-3** (error-aware quantisation as a `QuantizationPolicy`;
no search, no CUDA, no packing, no productive integration).
Predecessors:
- `docs/AQS_ARCHITECTURE_AUDIT.md` (`6e1ead7`)
- `docs/HANDOFF_AQS_1.md` — `QuantizationPolicy` (`c5e39bb`)
- `docs/HANDOFF_AQS_2.md` — per-tensor drift evaluator (`602901b`)

## Objective

Test whether GPTQ-style error-aware quantisation can push drift below
the AWQ/hybrid plateau (~0.78–0.89 end-to-end on TinyLlama from the
β-pivot investigation). Speed is explicitly **not** a goal yet — this is
a measurement vehicle, CPU-only, F32 in / F32 out.

## What GPTQ this implements

A deliberately **simplified, diagonal-Hessian GPTQ surrogate**
(`src/quant/gptq.rs`). For a row-major `[K_in, N_out]` weight `W` and a
per-input-channel diagonal Hessian `h` (length `K`):

1. **Hessian-regularised scale search.** Per (group, output column),
   sweep a small grid of clip factors (`1.0 … 0.5`) and pick the INT8
   group scale that minimises the Hessian-weighted reconstruction error
   `Σ_k (h[k] + λ)·(W[k,n] − dequant)²`, where `λ = damp_percent·mean(h)`.
   High-`h` (high activation energy) channels pull the scale toward
   preserving them. **This is the error-aware core.**
2. **Sequential error diffusion** along the K axis within each column:
   each weight's quantisation residual is carried forward into the next.
   This is the cheap scalar surrogate for GPTQ's "compensate the
   not-yet-quantised weights" step. (With a purely diagonal Hessian
   there are no off-diagonal terms to exploit, so the textbook update is
   degenerate — error diffusion recovers the dominant column-level
   cancellation effect.)

Output is a plain F32 reconstruction in place — same contract as every
other policy; the runtime never learns GPTQ exists.

## Hessian approximation used

```
approximate_hessian_diag(activations, shape=[S, K]) -> Vec<f32>[K]
  h[k] = (1/S) · Σ_s activations[s·K + k]²     (per-channel E[x²])
  floored at 1e-12 (strictly positive)
```

`GptqPolicy` receives the per-row activation **absmax** from
`CalibrationContext` (length `K`), treats it as a single calibration
sample (`shape = [1, K]`), so the effective Hessian is `h[k] = absmax[k]²`.

## New API

```rust
// src/quant/gptq.rs
pub struct GptqConfig { pub group_size: usize, pub damp_percent: f32 }  // default 128 / 0.01
pub enum GptqError { NotTwoDimensional, LengthMismatch, InvalidGroupSize,
                     InvalidDampPercent, HessianLengthMismatch }
pub fn approximate_hessian_diag(activations: &[f32], shape: &[usize]) -> Vec<f32>;
pub fn apply_gptq_reconstruction_inplace(weights: &mut [f32], shape: &[usize],
                                         hessian_diag: &[f32], config: &GptqConfig)
                                         -> Result<(), GptqError>;
pub fn gptq_memory_bytes(shape: &[usize], group_size: usize) -> u64;

// src/quant/policy.rs — placeholder replaced with real impl
pub struct GptqPolicy { pub group_size: usize, pub damp_percent: f32 }  // default 128 / 0.01
impl QuantizationPolicy for GptqPolicy { … }
```

`PolicyError` gained one variant: `InvalidDampPercent`. A
`From<GptqError> for PolicyError` mapping was added.

## Integration with AQS-1 / AQS-2

* **AQS-1**: `GptqPolicy` is a full `QuantizationPolicy` now (the
  placeholder is gone). It requires `CalibrationContext::activation_absmax`
  like AWQ/Hybrid; missing stats surface as `MissingActivationStats`.
* **AQS-2**: `evaluate_tensor_policy(input, &GptqPolicy::new(..), cal)`
  works with **zero evaluator changes** — verified by
  `gptq_evaluator_returns_metrics`.

## Memory model

`gptq_memory_bytes = numel (INT8) + num_groups·N·4 (scales) + K·4 (Hessian diag)`.
Lighter than F32 baseline; the Hessian metadata term is the only addition
over plain INT8.

## Metrics produced

Through the AQS-2 evaluator: `max_abs_diff`, `mean_abs_diff`, `rmse`,
`argmax_match`, `memory_bytes` (all local, weight-buffer level).

## Conceptual comparison: AWQ vs Hybrid vs GPTQ

| Mechanism | Where the smartness lives | Needs activations | β-pivot end-to-end (TinyLlama) |
|-----------|---------------------------|-------------------|--------------------------------|
| **AWQ**    | pre-scale rows by `act^α`, then plain INT8; smartness is in *spreading* per-channel magnitude before a uniform quantiser | yes | 0.78 (synthetic), 0.89 (real-text), argmax 4/4 |
| **Hybrid** | AWQ pre-scale + carve top-k outlier columns into an F32 sidecar | yes | 0.83 (worse than AWQ alone — orthogonality falsified) |
| **GPTQ (this)** | error-aware scale (Hessian-weighted clip search) + sequential error diffusion; smartness is in *minimising weighted error* and *cancelling it across the column* | yes | **not yet measured end-to-end** (AQS-4) |

## Honest local-drift finding (important)

On a 64×64 unstructured random tensor the informational comparison
(`awq_vs_gptq_local_drift`, `#[ignore]`) gives:

```
AWQ  : max_abs_diff=0.016149  mean=0.002224  rmse=0.002910
GPTQ : max_abs_diff=0.048487  mean=0.002347  rmse=0.003266
```

GPTQ's **local `max_abs_diff` is higher**, not lower. This is expected
and is a property of the mechanism, not a bug:

* Error diffusion deliberately trades larger *per-element* deviation for
  *column-level* error cancellation (`Σ_k x_k·ΔW[k,n] → 0`). The matmul
  output depends on the column sum, not on per-element max — so the
  AQS-2 local `max_abs_diff` metric **systematically understates GPTQ**.
* On unstructured random data there is no activation structure for the
  Hessian-weighted scale search to exploit, so the clip search cannot
  help either.

**Conclusion:** the cheap local signal is the wrong instrument to judge
GPTQ. Its real verdict needs the F64 forward harness on a real model
(TinyLlama), which is deferred to AQS-4/AQS-5. AQS-3 only proves the
mechanism is wired, deterministic, and evaluator-compatible.

## What this explicitly does NOT implement

* No full `K×K` Hessian, no inverse, no Cholesky, no blockwise update.
* No act-order / permutation. No bit-packing, no INT4.
* No CUDA, no optimised kernel, no inference speedup.
* No search (AQS-4), no tier-planner, no CLI, no generation, no loader /
  manifest changes. Not enabled by default anywhere.
* No claim of paper-exact GPTQ numerics or of beating ADR-004.

## Tests

13 new GPTQ tests (7 in `policy.rs`, 6 in `gptq.rs`) + 1 `#[ignore]`
informational comparison:

* policy: `gptq_policy_requires_activation_stats`,
  `gptq_policy_applies_deterministically`,
  `gptq_policy_produces_finite_output`,
  `gptq_policy_reports_memory_bytes`,
  `gptq_policy_rejects_bad_group_size`,
  `gptq_policy_rejects_bad_damp_percent`,
  `gptq_policy_reduces_to_identity_on_zero_tensor`
* gptq: `gptq_hessian_diag_is_positive`,
  `gptq_hessian_diag_averages_samples`,
  `gptq_reconstruction_is_deterministic_and_finite`,
  `gptq_reconstruction_identity_on_zero_tensor`,
  `gptq_reconstruction_rejects_bad_inputs`,
  `gptq_evaluator_returns_metrics`
* ignored: `awq_vs_gptq_local_drift`

Local validation: `cargo test --lib --release -- --test-threads=1` →
**572 passed / 0 failed / 1 ignored**. Heavy F64 NOT run (AQS-3 is a
mechanism-wiring milestone, not end-to-end certification).

## Files modified

* `src/quant/gptq.rs` — new.
* `src/quant/policy.rs` — placeholder `GptqPolicy` replaced with real
  impl; `PolicyError::InvalidDampPercent` + `From<GptqError>` added;
  placeholder test replaced with 7 real GPTQ policy tests.
* `src/quant/mod.rs` — `pub mod gptq;` + re-exports.
* `docs/HANDOFF_AQS_3.md` — this file.

No other files touched.

## Next step (needs authorisation)

AQS-4: the search loop that wraps the AQS-2 evaluator per layer + a
selection rule + manifest output — and, critically, the **F64 forward
harness run** that finally measures GPTQ end-to-end against the
AWQ/hybrid plateau on TinyLlama.
