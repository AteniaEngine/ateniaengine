# HANDOFF — AQS-1: `QuantizationPolicy` abstraction landed

Milestone: **AQS-1** (architecture only; no search, no GPTQ, no productive integration).
Predecessor: `docs/AQS_ARCHITECTURE_AUDIT.md` (commit `6e1ead7`).

## What landed

A new experimental module:

```
src/quant/
  mod.rs       — public re-exports
  policy.rs    — trait + 5 policies + tests
```

Wired into `src/lib.rs` as `pub mod quant;` (always-on, but with no
productive caller — purely opt-in via `&dyn QuantizationPolicy`).

## New surface

```rust
pub trait QuantizationPolicy {
    fn id(&self) -> &'static str;
    fn validate(&self, shape: &[usize]) -> Result<(), PolicyError>;
    fn apply_inplace(
        &self,
        weights: &mut [f32],
        shape: &[usize],
        cal: &CalibrationContext<'_>,
    ) -> Result<(), PolicyError>;
    fn memory_bytes(&self, shape: &[usize]) -> u64;
}

pub struct CalibrationContext<'a> {
    pub activation_absmax: Option<&'a [f32]>,
    pub corpus_hash: Option<[u8; 32]>,
}

pub enum PolicyError {
    NotTwoDimensional { rank: usize },
    LengthMismatch { expected: usize, actual: usize },
    InvalidGroupSize,
    OutlierKExceedsColumns { k: usize, n: usize },
    InvalidAlpha,
    MissingActivationStats,
    ActivationStatsLengthMismatch { expected: usize, actual: usize },
    InnerError(String),
}
```

## Initial policies

| Policy            | Wraps existing helper                                 | Calibration |
|-------------------|-------------------------------------------------------|-------------|
| `Bf16Fallback`    | (no-op, identity)                                     | none        |
| `PlainInt8`       | `absmax_per_group_symmetric` + inline dequant         | none        |
| `AwqPolicy`       | `awq_per_row_scales_from_activations` + `apply_awq_perturbation_inplace` | activation absmax |
| `HybridPolicy`    | same scales + `apply_hybrid_awq_outlier_perturbation_inplace` | activation absmax |
| `GptqPolicy`      | **placeholder** — both `validate` and `apply_inplace` return `PolicyError::InnerError("not yet implemented")` | n/a |

All math is delegated to the existing `crate::tensor::quantizer`
helpers; no numerical logic was duplicated, copied, or altered.

## What is explicitly NOT in this milestone

* No search / evaluator / cost model — that is AQS-2.
* No GPTQ implementation — that is AQS-3.
* No `WeightStore::perturb_param_with_policy` helper. The four
  existing `perturb_param_with_*` methods remain the only path from
  `WeightStore` into these helpers. Adding a `&dyn`-based delegation
  would have required either an extra dependency edge from
  `weight_store.rs` into `crate::quant` or carrying the policy
  enum/trait across the entire amg module surface; either was more
  than AQS-1 was scoped for. Deferred to AQS-2.
* No CUDA, no tier-planner, no CLI, no generation, no loader, no
  manifest changes.
* No default-on behaviour. Nothing reaches `quant::*` unless a caller
  constructs a policy explicitly (today: only the unit tests).

## Tests

`src/quant/policy.rs` carries 15 unit tests covering:

* `bf16_policy_is_noop`
* `plain_int8_policy_changes_weights_deterministically`
* `awq_policy_requires_activation_stats`
* `awq_policy_applies_with_valid_stats`
* `awq_policy_rejects_wrong_activation_length`
* `hybrid_policy_requires_activation_stats`
* `hybrid_policy_applies_with_valid_stats`
* `hybrid_policy_rejects_excessive_outlier_k`
* `policy_rejects_non_2d_shape`
* `policy_rejects_bad_group_size`
* `policy_rejects_bad_alpha`
* `policy_memory_bytes_are_monotonic_reasonable`
* `policy_ids_are_stable`
* `gptq_policy_is_placeholder`
* `dyn_dispatch_works`

Local validation: `cargo test --lib --release -- --test-threads=1` →
**550 passed / 0 failed** (535 pre-existing + 15 new).

Heavy F64 harness was deliberately NOT re-run — AQS-1 is a refactor,
not a numerical change; the β / β-pivot results in
`docs/HANDOFF_INT8_OUTLIER_BETA.md` remain authoritative.

## Files modified

* `src/quant/mod.rs` — new.
* `src/quant/policy.rs` — new.
* `src/lib.rs` — added `pub mod quant;` declaration only.
* `docs/HANDOFF_AQS_1.md` — this file.

No other files touched.

## Next step

AQS-2: policy evaluator + per-layer drift estimator + a small
`policy_registry` so candidates can be enumerated by id. Requires
explicit user authorisation.
