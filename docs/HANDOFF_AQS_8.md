# HANDOFF — AQS-8: callback-based model runner

Milestone: **AQS-8** (the runner that closes the search loop). Callback-
based (Option A), experimental, CPU-only, opt-in, deterministic. No CLI,
no runtime manifest, no new techniques.
Predecessors: AQS-0..7 (`6e1ead7` … `5f9dbd2`).

## What the runner does

`src/quant/runner.rs` drives an [`AqsSearchConfig`] candidate grid through
an **evaluator callback**, accumulates one `EndToEndEvalResult` per
candidate, and feeds them into AQS-7's `search_from_end_to_end_results`:

```text
AqsSearchConfig (grid)
  → for each candidate, in order:
      check support vs AqsEvaluatorCapabilities
      → evaluator(spec) -> EndToEndEvalResult
  → search_from_end_to_end_results(...)   (AQS-7)
  → AqsRunnerResult { search_result, evaluated, skipped, errors }
```

Entry point:

```rust
pub fn run_aqs_with_evaluator<F>(
    config: AqsRunnerConfig,
    capabilities: AqsEvaluatorCapabilities,
    evaluator: F,
) -> AqsRunnerResult
where F: FnMut(&AqsCandidateSpec) -> Result<EndToEndEvalResult, AqsRunnerError>;
```

## Why callback-based (Option A)

`src/quant/runner.rs` must NOT load or run a real model — that would
couple `quant` to the loader / model runtime and duplicate the AQS-4
harness. The runner instead receives a closure that produces one result
per candidate. Benefits:

* **Decoupled** — no model I/O lives in `quant`.
* **Deterministic + unit-testable** — fake evaluators, no model required;
  all 10 tests run in milliseconds.
* **Reusable** — the heavy AQS-4 harness can pass a *real* evaluator that
  runs the forward, while `runner.rs` stays oblivious to it.

Option B (runner loads the model) was explicitly rejected for AQS-8 per
the scope's strong preference.

## How it connects to AQS-7

The runner does **not** re-implement classification, ranking, or
selection. It collects results and calls
`search_from_end_to_end_results(...)`, so the AQS-6 certification report
and AQS-7 search remain the single source of truth. `AqsRunnerResult`
exposes the resulting `AqsSearchResult` (with `best_certified()`,
`best_useful_lossy()`, `recommended_policy()`, `manifest_draft()`).

## How it handles unsupported candidates

`AqsEvaluatorCapabilities { activation_absmax, activation_matrix }`
declares what the evaluator can provide. Before calling the evaluator,
the runner checks `capabilities.supports(spec)` using AQS-7's
`needs_activation_matrix()` / `needs_activation_absmax()`:

* unsupported + `skip_unsupported = true` → recorded in `skipped`
  (reason string), never evaluated, **no panic**;
* unsupported + `skip_unsupported = false` → recorded as an
  `UnsupportedCandidate` error (and, if `stop_on_first_error`, aborts).

Example: capabilities `{absmax: true, matrix: false}` → the GPTQ
candidate (needs a matrix) is skipped while AWQ/Hybrid (need absmax) run.
Capabilities `none()` → only BF16 + plain INT8 are supported.

## Error model (no panics)

```rust
pub enum AqsRunnerError {
    UnsupportedCandidate(String),
    EvaluationFailed(String),
    EmptyCandidateList,
}
```

Evaluation errors are recorded per-candidate (`AqsRunnerErrorRecord`); the
run continues unless `stop_on_first_error`. An empty candidate list yields
a result with an `EmptyCandidateList` error and an empty report — never a
panic.

## Heavy harness

**Not modified** (Phase 8 optional). The runner is fully exercised by fake
evaluators on the real consolidated TinyLlama numbers. Wiring the AQS-4
heavy test to call `run_aqs_with_evaluator` with a real forward evaluator
is left for AQS-9 (it would re-run the ~8 h forward, adds no new signal
here). The integration point is clean: the heavy test already computes one
`EndToEndEvalResult` per policy; that loop becomes the evaluator closure.

## Tests (10, all green, in CI)

`runner_evaluates_candidates_in_order`, `runner_accumulates_results`,
`runner_skips_unsupported_matrix_candidate`,
`runner_errors_on_unsupported_when_skip_false`,
`runner_calls_search_from_results`,
`runner_preserves_best_certified_and_best_lossy`,
`runner_handles_empty_candidate_list`,
`runner_records_evaluation_error_and_continues_when_configured`,
`runner_stops_on_first_error_when_configured`,
`runner_result_is_deterministic`.

Local validation: `cargo test --lib --release` → **619 passed / 0 failed /
1 ignored** (609 + 10).

## Limits

* **No model loading in `src/quant/runner.rs`** — callback only.
* **No CLI** (`atenia search` is a later, separately-authorised step).
* **No productive manifest**, no runtime integration — the manifest stays
  a `3.0.0-draft` string.
* **No new techniques, no CUDA, no tier-planner, no generation, no
  loaders.**

## Files modified

* `src/quant/runner.rs` — new (capabilities, config, runner, 10 tests).
* `src/quant/mod.rs` — `pub mod runner;` + re-exports.
* `docs/HANDOFF_AQS_8.md` — this file.

No other files touched.

## Next step (needs authorisation)

AQS-9: wire the AQS-4 heavy harness to call `run_aqs_with_evaluator` with
a *real* forward evaluator (and `AqsEvaluatorCapabilities::all()` once a
calibration matrix is captured), producing a real end-to-end search +
manifest draft from one entry point. After that, the `atenia search` CLI +
optional productive manifest emission would be the final, separately-
authorised step.
