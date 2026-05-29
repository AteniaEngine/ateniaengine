# HANDOFF — AQS-7: deterministic policy search engine

Milestone: **AQS-7** (the orchestrator that turns the AQS pieces into a
search → classify → rank → certify workflow). Experimental, CPU-only,
opt-in, deterministic. No CLI, no runtime manifest, no new techniques.
Predecessors: AQS-0..6 (`6e1ead7` … `d5484b4`).

## What the search engine does

`src/quant/search.rs` ties the existing AQS pieces into the audit's
headline workflow:

```text
generate candidates  (default grid or explicit AqsSearchConfig)
  → build policies     (stable AqsPolicyKind factory)
  → evaluate           (end-to-end results, supplied by the AQS-4 harness)
  → classify + rank     (AQS-6 AqsCertificationReport)
  → best_certified + best_useful_lossy + manifest draft
```

The primary entry point consumes **already-computed** end-to-end results,
so the search is fully unit-testable without any model and never
duplicates the harness:

```rust
pub fn search_from_end_to_end_results(
    model_name: &str,
    adr004_gate: f32,
    baseline_memory_bytes: Option<u64>,
    results: Vec<EndToEndEvalResult>,
) -> AqsSearchResult
```

`AqsSearchResult` exposes `best_certified()`, `best_useful_lossy()`,
`recommended_policy()` (never substitutes lossy for certified),
`manifest_draft()`, `human_report()`, plus `evaluated_count` /
`skipped_count`. Classification, ranking and selection are delegated
entirely to the AQS-6 `AqsCertificationReport` — no logic is duplicated.

## What it does NOT do (this phase)

* **No real models from `search.rs`.** The heavy TinyLlama forward stays
  in the AQS-4 harness; `search.rs` consumes its output. This keeps the
  search deterministic and testable in milliseconds.
* **No CLI** (`atenia search` is a later, explicitly-authorised step).
* **No productive manifest** integration — the manifest stays a
  `3.0.0-draft` string from AQS-6.
* **No new quantisation techniques**, no GPTQ changes, no CUDA, no
  tier-planner, no generation, no loader changes.
* **No massive grid** — the default grid is deliberately small (7
  candidates), avoiding the combinatorial explosion the AQS-0 audit
  warned about.

## Default candidate grid (deterministic, conservative)

```text
bf16
plain_int8           g128
awq                  g128 alpha=0.20
awq                  g128 alpha=0.25
awq                  g128 alpha=0.30
hybrid               g128 alpha=0.25 k64
gptq                 g128 block128 damp=0.01
```

Stable order, stable names (`bf16`, `plain_int8_g128`, `awq_g128_a0.20`,
…). `default_candidate_grid()` is unit-tested for determinism.

## Policy factory

`policy_for_kind(AqsPolicyKind) -> Box<dyn QuantizationPolicy>` delegates
to the **existing** constructors (`Bf16Fallback`, `PlainInt8::new`,
`AwqPolicy::new`, `HybridPolicy::new`, `GptqPolicy::with_config`). No new
logic; policy `id()`s stay unchanged. `AqsPolicyKind` also reports
`needs_activation_matrix()` (GPTQ) / `needs_activation_absmax()` (AWQ,
Hybrid) so a future model-driving runner can mark candidates *unsupported*
(skip, never panic) when calibration data is missing.

## How it uses AQS-6

`search_from_end_to_end_results` calls `AqsCertificationReport::build`,
which classifies each result (`Adr004Certified` / `UsefulLossy` /
`Failed`), ranks them deterministically, and selects best certified / best
useful lossy. On the real TinyLlama AQS-5 numbers the search reproduces:

| selection | value |
|-----------|-------|
| `best_certified` | `bf16` |
| `recommended_policy` | `bf16` (never the lossy one) |
| `best_useful_lossy` | `awq` (0.889, argmax true, ~1.94×) |
| failed | plain_int8, hybrid, gptq |

## Optional local per-tensor search (LOCAL SIGNAL ONLY)

`search_tensor_local(candidates, values, shape, cal)` runs candidates
through the **AQS-2 evaluator** on one F32 tensor and ranks by ascending
*local* weight-buffer drift. It is explicitly a cheap pre-filter, **not**
certification — AQS-3/AQS-4 proved local drift can mis-rank policies
end-to-end (GPTQ). Candidates that error (AWQ/GPTQ without calibration)
are **skipped, never panicked**; the function returns
`(rankings, skipped_count)`. Kept clearly separate from the end-to-end
certification path.

## AQS-4 heavy test integration

Not wired (Phase 9 optional). The search builds from the harness'
`EndToEndEvalResult` vector directly; unit tests exercise it on the real
consolidated TinyLlama numbers. Re-running the ~8 h heavy test only to
print a search report adds no signal.

## Tests (12, all green, in CI)

`default_grid_is_deterministic`, `policy_factory_builds_supported_policies`,
`search_from_results_selects_best_certified`,
`search_from_results_selects_best_useful_lossy`,
`search_from_results_marks_awq_useful_lossy`,
`search_from_results_keeps_failed_candidates`,
`search_result_manifest_draft_is_stable`,
`search_does_not_promote_lossy_over_certified`,
`search_handles_empty_results`, `search_counts_evaluated_and_skipped`,
`local_search_ranks_and_skips_uncalibrated`, `local_search_is_deterministic`.

Local validation: `cargo test --lib --release` → **609 passed / 0 failed /
1 ignored** (597 + 12).

## Files modified

* `src/quant/search.rs` — new (config, grid, factory, search, local
  search, 12 tests).
* `src/quant/mod.rs` — `pub mod search;` + re-exports.
* `docs/HANDOFF_AQS_7.md` — this file.

No other files touched. No CUDA, tier-planner, CLI, generation, loader, or
productive-manifest changes.

## Next step (needs authorisation)

AQS-8 would be the **model-driving runner** that closes the loop: take an
`AqsSearchConfig`, run each candidate through the AQS-4 forward on a real
model (skipping unsupported ones via the `needs_*` flags), and feed the
results straight into `search_from_end_to_end_results`. After that, a
`atenia search` CLI + optional manifest emission would be the final,
separately-authorised step.
