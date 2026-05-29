# HANDOFF — AQS-9: runner wired into the TinyLlama harness

Milestone: **AQS-9** (the callback runner driving the *real* TinyLlama
end-to-end harness from one entry point). Experimental, CPU-only, opt-in.
No CLI, no runtime manifest, no new techniques.
Predecessors: AQS-0..8 (`6e1ead7` … `f3221bf`).

## What was wired

`tests/aqs4_end_to_end_test.rs` now closes the loop: a single heavy test
builds an `AqsRunnerConfig` (default grid), supplies a **real evaluator
callback**, and runs `run_aqs_with_evaluator` — which produces the AQS-7
search result + AQS-6 certification report + manifest draft from genuine
TinyLlama forwards.

```text
AqsRunnerConfig (default grid)
  → run_aqs_with_evaluator(cfg, capabilities, evaluator)
      evaluator(candidate):
        build_policy(candidate.kind)        (AQS-7 factory)
        clone WeightStore                   (original never mutated)
        perturb_all_proj_with_policy[_matrix]
        forward_drift vs F64 fixture        (AQS-4 helper)
        → EndToEndEvalResult
  → search_from_end_to_end_results          (AQS-7)
  → certification report + manifest draft   (AQS-6)
```

## How the real evaluator callback works

It reuses the existing AQS-4 helpers — no forward logic is duplicated:

* `spec.build_policy()` constructs the policy via the AQS-7 stable factory.
* A fresh `WeightStore` clone is perturbed with
  `perturb_all_proj_with_policy` (absmax path) or
  `perturb_all_proj_with_policy_matrix` (GPTQ matrix path), chosen by
  `spec.kind.needs_activation_matrix()`.
* `forward_drift(...)` runs the CPU F32 forward and compares logits to the
  F64 fixture (the same helper the AQS-4 test uses).
* Any perturbation error becomes `AqsRunnerError::EvaluationFailed` —
  never a panic.

One shared calibration pass feeds every candidate (absmax map for
AWQ/Hybrid, `[S,K]` matrix map for GPTQ).

## Candidates evaluated by default

The default grid minus GPTQ real:

```
bf16
plain_int8_g128
awq_g128_a0.20
awq_g128_a0.25
awq_g128_a0.30
hybrid_g128_a0.25_k64
gptq_g128_b128_d0.01   ← SKIPPED by default
```

## How GPTQ real is skipped (and how to enable it)

Capabilities gate it:

```rust
AqsEvaluatorCapabilities {
    activation_absmax: true,
    activation_matrix: env ATENIA_AQS_RUN_GPTQ_REAL == "1",   // false by default
}
```

GPTQ needs an activation matrix; with `activation_matrix: false` the
runner records it in `skipped` (reason string) and never evaluates it —
no panic, no ~7.8 h cost. Set `ATENIA_AQS_RUN_GPTQ_REAL=1` to include real
GPTQ (documented as ~7.8 h on CPU). **Not run by default.**

## Real result (heavy test, GPTQ skipped, 120 s)

`aqs9_tinyllama_runner_produces_search_report` (`#[ignore]`) ran on the
real TinyLlama in **120 s** and produced:

```
candidate              status              max_diff   argmax   compression
bf16                   adr004_certified    0.000063    true         1.00x
awq_g128_a0.25         useful_lossy        0.889217    true         1.94x
awq_g128_a0.20         useful_lossy        1.337593    true         1.94x
hybrid_g128_a0.25_k64  failed              0.831786   false         1.78x
awq_g128_a0.30         failed              0.925052   false         1.94x
plain_int8_g128        failed              1.260771   false         1.94x
gptq_g128_b128_d0.01   SKIPPED (matrix capability off)
best certified     : bf16
best useful lossy  : awq_g128_a0.25
```

* **bf16 → certified**, **awq α=0.25 → best useful lossy** (0.889, argmax
  true, ~1.94× — reproduces the AQS-5 headline number exactly), **gptq
  skipped**, **errors empty**.
* New signal from the α-sweep: α=0.25 is the sweet spot — α=0.20 keeps
  argmax but drifts more (1.34), and α=0.30 actually *breaks* argmax
  (failed). The search engine surfaced this automatically.
* AWQ is **never** labelled certified (drift ≥ 0.5).

All heavy-test assertions held: `best_certified == bf16`,
`recommended == bf16`, best useful lossy is an AWQ variant and is not
certified, manifest draft contains `schema_version: "3.0.0-draft"`, GPTQ
skipped.

## Tests

* **Light (CI, no model)** in `mod aqs9_light`:
  `runner_default_skips_gptq_and_certifies_bf16`,
  `runner_manifest_from_output_is_draft`,
  `runner_with_matrix_capability_evaluates_gptq`.
* **Heavy (`#[ignore]`)**: `aqs9_tinyllama_runner_produces_search_report`
  — the table above; passed in 120 s with GPTQ skipped.

Local validation: `cargo test --lib --release` → **619 passed / 0 failed /
1 ignored**. `cargo test --test aqs4_end_to_end_test` → 8 passed / 2
ignored. Heavy AQS-9 run reported above.

## Limits

* **No CLI** (`atenia search` is the next, separately-authorised step).
* **No productive manifest** — the manifest stays a `3.0.0-draft` string.
* **No runtime integration**, no CUDA, no tier-planner, no generation, no
  loader changes, no new techniques.
* The heavy test stays `#[ignore]`; GPTQ real stays off by default.

## Files modified

* `tests/aqs4_end_to_end_test.rs` — `mod aqs9_light` (3 tests) + heavy
  `aqs9_tinyllama_runner_produces_search_report`.
* `docs/HANDOFF_AQS_9.md` — this file.

No `src/` changes were needed (the AQS-8 runner API was sufficient). No
other files touched.

## Next step (needs authorisation)

AQS-10: the `atenia search` CLI that wraps this runner for one-command
"certify this model" + optional productive-manifest emission — a
separate, explicitly-authorised step (it would touch the CLI surface,
out of scope here).
