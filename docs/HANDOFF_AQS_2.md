# HANDOFF — AQS-2: per-tensor drift evaluator landed

Milestone: **AQS-2** (cheap local drift signal; no search, no GPTQ, no
forward, no productive integration).
Predecessors:
- `docs/AQS_ARCHITECTURE_AUDIT.md` (commit `6e1ead7`)
- `docs/HANDOFF_AQS_1.md` — `QuantizationPolicy` (commit `c5e39bb`)

## Objective

Build the base brick the future AQS search (AQS-4) will call to rank
candidate policies per layer:

```text
tensor + policy
  → apply policy on an F32 copy
  → compare against the F32 original
  → report drift + local argmax match + memory estimate
```

## What landed

New experimental module:

```
src/quant/evaluator.rs
```

Re-exported from `src/quant/mod.rs`.

## New API

```rust
pub struct TensorEvalInput<'a> {
    pub name: &'a str,
    pub values: &'a [f32],
    pub shape: &'a [usize],
}

pub struct TensorEvalResult {
    pub tensor_name: String,
    pub policy_id: String,
    pub numel: usize,
    pub memory_bytes: u64,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub rmse: f32,
    pub argmax_match: Option<bool>,
}
// helpers: baseline_f32_bytes(), compression_ratio()

pub enum EvalError {
    EmptyInput,
    ShapeMismatch { expected: usize, actual: usize },
    PolicyValidationFailed(PolicyError),
    PolicyApplicationFailed(PolicyError),
    NonFiniteResult { index: usize },
}

pub fn evaluate_tensor_policy(
    input: TensorEvalInput<'_>,
    policy: &dyn QuantizationPolicy,
    cal: &CalibrationContext<'_>,
) -> Result<TensorEvalResult, EvalError>;

pub fn evaluate_tensor_policies(
    input: TensorEvalInput<'_>,
    policies: &[&dyn QuantizationPolicy],
    cal: &CalibrationContext<'_>,
) -> Result<Vec<TensorEvalResult>, EvalError>;
```

## Metrics computed

| Metric | Definition | Notes |
|--------|------------|-------|
| `max_abs_diff` | `max\|orig[i] − pert[i]\|` (L∞) | same metric ADR-004 gates on, but **local** on the weight buffer |
| `mean_abs_diff` | `mean\|Δ\|` (L1 / numel) | f64 accumulation, f32 report |
| `rmse` | `sqrt(mean(Δ²))` | f64 accumulation, f32 report |
| `argmax_match` | `argmax(orig) == argmax(pert)` over tensor values | **local argmax over values, NOT logits argmax** |
| `memory_bytes` | `policy.memory_bytes(shape)` | target store cost |

`compression_ratio()` = `memory_bytes / (numel*4)`.

## Policies it can evaluate

Any `&dyn QuantizationPolicy` — i.e. all AQS-1 policies:
`Bf16Fallback`, `PlainInt8`, `AwqPolicy`, `HybridPolicy`. `GptqPolicy`
surfaces its placeholder error through `EvalError::PolicyValidationFailed`.
Calibration-dependent policies (AWQ, Hybrid) require
`CalibrationContext::with_activations(..)`; missing stats surface as
`EvalError::PolicyApplicationFailed(PolicyError::MissingActivationStats)`.

## Batch evaluator

`evaluate_tensor_policies` **was** implemented. It runs a list of
policies against one tensor and returns results **in input order**. It
is explicitly NOT a search: no sorting, no selection, no pruning. It
fails fast on the first erroring policy.

## Limits — read before using

* **This is a cheap local signal, not final certification.** It does
  NOT run a model forward. Low per-tensor drift does not imply
  ADR-004 pass: the β.4 → β.5 projection was off by ~60× because
  quantisation error cascades through the network
  (`docs/HANDOFF_INT8_OUTLIER_BETA.md`).
* `argmax_match` is a coarse "did the dominant weight element survive"
  probe — it is unrelated to token/logits argmax.
* No search (AQS-4), no GPTQ (AQS-3), no CLI, no generation, no CUDA,
  no tier-planner, no loader / manifest changes.

## How this prepares AQS-3 / AQS-4

* AQS-3 (GPTQ): once `GptqPolicy` is implemented, the evaluator scores
  it with zero new code — it is just another `&dyn QuantizationPolicy`.
* AQS-4 (search): the evaluator is the inner loop. The search will
  wrap `evaluate_tensor_policies` per layer, add a selection rule
  (drift budget vs memory), and persist the winning policy ids into a
  manifest. Selection logic stays out of AQS-2 by design.

## Tests

`src/quant/evaluator.rs` carries 10 unit tests:
`evaluate_bf16_policy_has_zero_drift`,
`evaluate_plain_int8_reports_positive_drift`,
`evaluate_awq_requires_activation_stats`,
`evaluate_awq_with_activation_stats_returns_metrics`,
`evaluate_hybrid_returns_metrics`,
`evaluate_reports_memory_bytes_from_policy`,
`evaluate_rejects_shape_mismatch`,
`evaluate_rejects_empty_values`,
`evaluate_detects_non_finite_output`,
`evaluate_batch_preserves_policy_order`.

Local validation: `cargo test --lib --release -- --test-threads=1` →
**560 passed / 0 failed** (550 pre-existing + 10 new). Heavy F64 harness
deliberately NOT run — AQS-2 is a measurement brick, not a numerical
change.

## Files modified

* `src/quant/evaluator.rs` — new.
* `src/quant/mod.rs` — added `pub mod evaluator;` + re-exports.
* `docs/HANDOFF_AQS_2.md` — this file.

No other files touched.
