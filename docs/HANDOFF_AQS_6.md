# HANDOFF — AQS-6: certification report + manifest draft

Milestone: **AQS-6** (auditable certification layer over the AQS-4/AQS-5
results). No new techniques, no CLI, no runtime manifest, no runtime
changes. Experimental, CPU-only, opt-in.
Predecessors: AQS-0..5 (`6e1ead7`, `c5e39bb`, `602901b`, `e31edd2`,
`5ac94fa`, `a5ca24e`, `2eb950b`).

## Decision: accept the plateau, build the differentiator

AQS-1..5 falsified **five** distinct weight-only quantisation mechanisms
against ADR-004 strict (`max_abs_diff < 0.5`) on TinyLlama:

| policy | max_abs_diff | argmax | ADR-004 |
|--------|-------------:|:------:|:-------:|
| certified(f32) | 0.000063 | true | PASS |
| bf16 | 0.000063 | true | PASS |
| plain_int8 | 1.260771 | false | FAIL |
| awq | 0.889217 | true | FAIL (best useful lossy) |
| hybrid | 0.831786 | false | FAIL |
| gptq (real) | 1.405399 | false | FAIL |

No weight-only technique crosses the gate. Chasing a sixth has a poor
expected payoff (real GPTQ already cost 7.8 h CPU to land *worse* than
AWQ). **AQS-6 therefore stops technique hunting and delivers the audit's
real differentiator: automatically classifying and certifying which
quantisation is safe for a given model against the F64 reference.** The
product value is the *report*, not a magic algorithm.

## What was implemented

`src/quant/certification.rs` — a pure report layer consuming the existing
`EndToEndEvalResult` from the AQS-4/AQS-5 harness. No new quantisation
logic.

## Certification statuses

```rust
pub enum CertificationStatus { Adr004Certified, UsefulLossy, Failed }
```

Classification rules (per policy result):

| status | condition |
|--------|-----------|
| `Adr004Certified` | finite metrics AND `argmax_match` AND `max_abs_diff < 0.5` |
| `UsefulLossy` | finite AND `argmax_match` AND `max_abs_diff ≥ 0.5` AND `compression_ratio > 1.0` |
| `Failed` | `!argmax_match`, non-finite metrics, or lossy without compression |

On the real TinyLlama set this yields exactly:
- **bf16 / f32 → `Adr004Certified`**
- **awq → `UsefulLossy`** (argmax true, drift 0.889 ≥ 0.5, compresses ~1.94×) — **never** labelled certified, satisfying scope rule 12.
- **plain_int8 / hybrid / gptq → `Failed`** (argmax false).

## Compression ratio

`compression_ratio = baseline_memory_bytes / policy.memory_bytes`,
`None` when either is 0 (e.g. the harness' certified baseline row carries
`memory_bytes = 0` — we do not invent a ratio). Baseline is supplied by
the caller (typically the BF16 footprint, 2 260 729 856 bytes for
TinyLlama → AWQ ratio ≈ 1.94×).

## Deterministic ranking

Group order: `Adr004Certified` < `UsefulLossy` < `Failed`. Within a group:
1. lower `max_abs_diff`
2. higher `compression_ratio`
3. lower `memory_bytes`
4. `candidate_name` ascending

Implemented via a total-order key (`OrderedF32` wraps f32 so NaN sorts
last instead of panicking). 100% deterministic — `render_human_report`
is idempotent (unit-tested).

## Selection

* `best_certified()` — best `Adr004Certified` (first after rank sort).
* `best_useful_lossy()` — best `UsefulLossy`.
* `recommended_policy()` — returns the **best certified** policy and
  **never** auto-substitutes a lossy one. The lossy option is surfaced
  separately. On TinyLlama: recommended = `bf16`, lossy recommendation =
  `awq`. (Unit test `recommended_policy_does_not_replace_certified_with_lossy`.)

## Manifest draft (experimental, NOT runtime)

`render_manifest_draft()` emits a hand-rendered YAML-ish string with an
explicit `3.0.0-draft` schema suffix. It is **never** consumed by the
runtime and does **not** touch the productive `numcert` manifest. Example
for the real TinyLlama set:

```yaml
schema_version: "3.0.0-draft"
model: tinyllama-1.1b
adr004:
  gate: 0.5
  certified_policy: bf16
lossy_recommendation:
  policy: awq
  reason: "argmax stable, best useful lossy compression"
policies:
  - name: bf16
    status: adr004_certified
    max_abs_diff: 0.000063
    argmax_match: true
    ...
  - name: awq
    status: useful_lossy
    max_abs_diff: 0.889217
    argmax_match: true
    ...
```

## Human report renderer

`render_human_report()` produces a deterministic table:

```
candidate          status                 max_diff   argmax   compression
-------------------------------------------------------------------------
bf16               adr004_certified       0.000063    true          1.00x
awq                useful_lossy           0.889217    true          1.94x
hybrid             failed                 0.831786   false          1.78x
plain_int8         failed                 1.260771   false          1.94x
gptq               failed                 1.405399   false          1.94x
best certified     : bf16
best useful lossy  : awq
```

## AQS-4 heavy test integration

**Not wired** into the heavy `aqs4_tinyllama_policy_comparison` test. The
report builds directly from the `EndToEndEvalResult` vector the harness
already produces, and the AQS-6 unit tests exercise it on the **real
consolidated TinyLlama numbers** (hard-coded from the AQS-5 run). Re-running
the 7.8 h heavy test only to print a table would add no signal and was
left out per scope (Phase 10 is optional). A caller can do
`AqsCertificationReport::build("tinyllama", &results, Some(bf16_bytes))`
on the harness output at any time.

## Tests (12, all green, in CI)

`classifies_bf16_as_certified`,
`classifies_awq_as_useful_lossy_not_certified`,
`classifies_argmax_false_as_failed`,
`ranking_prefers_certified_over_lossy`,
`ranking_prefers_lower_drift_within_status`,
`ranking_uses_compression_as_tiebreaker`,
`selects_best_certified_and_best_lossy`,
`manifest_draft_contains_gate_and_policy_status`,
`human_report_renderer_is_deterministic`,
`recommended_policy_does_not_replace_certified_with_lossy`,
`no_certified_yields_none_recommendation_but_keeps_lossy`,
`lossy_without_compression_is_failed`.

Local validation: `cargo test --lib --release` → **597 passed / 0 failed /
1 ignored** (585 + 12).

## Limits

* **Draft only.** The manifest is `3.0.0-draft` and never reaches the
  runtime. The productive `numcert` manifest is untouched.
* **No CLI**, no automatic search across α / group_size / etc. (AQS-7+).
* **No new techniques, no GPTQ changes, no CUDA, no tier-planner, no
  generation, no loader, no productive manifest.**

## Files modified

* `src/quant/certification.rs` — new (report layer + 12 tests).
* `src/quant/mod.rs` — `pub mod certification;` + re-exports.
* `docs/HANDOFF_AQS_6.md` — this file.

No other files touched.

## Next step (needs authorisation)

AQS-7: the **automatic search** — sweep policy hyper-parameters (AWQ α,
group_size, …) per model, feed each candidate through the AQS-4 harness,
and let AQS-6 rank/certify the results into a manifest. That is the point
where AQS becomes a one-command "certify this model" tool. CLI + runtime
manifest integration would follow as a separate, explicitly-authorised
step.
