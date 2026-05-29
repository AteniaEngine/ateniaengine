# HANDOFF — AQS-10: `atenia search` CLI command

Milestone: **AQS-10** (experimental `atenia search` CLI front-end for the
AQS certification stack). Experimental, CPU-only, opt-in. No model
loading, no forward, no fixtures, no productive manifest.
Predecessors: AQS-0..9 (`6e1ead7` … `b47c658`).

## The architectural decision behind this design

A faithful `atenia search --model <path>` that produces real ADR-004
certification is **not possible** for an arbitrary model: the end-to-end
drift metric is computed against a **per-model F64 reference** that only
exists as test fixtures (generated offline) for four models, with
hardcoded inputs. Running a real forward + comparison lives in the heavy
`#[ignore]` harness, deliberately outside the shipped binary.

So AQS-10 is the **honest, decoupled front-end** (chosen option: "CLI
scaffold + results file"): it consumes a pre-computed end-to-end results
file and renders the AQS-6 certification report + manifest draft. It never
loads a model, runs a forward, uses fixtures, or simulates F64 — and never
promises certification beyond what the results file actually contains.

## What was implemented

* `src/cli/search.rs` — `SearchArgs`, `run_search`, a serde results-file
  schema, and `report_from_json` (the testable core).
* `atenia search` subcommand wired into `src/bin/atenia.rs`.
* `pub mod search;` in `src/cli/mod.rs`.

## How `atenia search` works

```text
--results <FILE>
  → read JSON  (serde; serde_json already a dependency)
  → deserialize into EndToEndEvalResult[]
  → search_from_end_to_end_results(...)   (AQS-7)
  → AqsCertificationReport                 (AQS-6: classify + rank + select)
  → print best certified + best useful lossy
  → --report   → render_human_report()
  → --manifest → render_manifest_draft()   (3.0.0-draft, never productive)
```

No `--results` → a clean `E-CLI-INVALID-ARGS` explaining that AQS CLI
requires a results file from the end-to-end harness (exit code 2).
Missing file → `E-IO-NOT-FOUND` (exit 2). Invalid JSON →
`E-CLI-INVALID-ARGS` with the parse error (exit 2).

## Flags

| flag | effect |
|------|--------|
| `--results <FILE>` | path to the pre-computed end-to-end results JSON (required) |
| `--report` | print the human-readable certification table |
| `--manifest` | print the `3.0.0-draft` manifest |
| `--include-gptq` | reserved for a future model-driving path; **inert** on the results-file path (prints a note) |

If neither `--report` nor `--manifest` is given, the report is shown by
default (an empty run would be useless).

## Results file schema

```json
{
  "model": "tinyllama-1.1b",
  "baseline_memory_bytes": 2260729856,
  "results": [
    {
      "candidate_name": "bf16",
      "max_abs_diff": 0.000063,
      "mean_abs_diff": 0.000008,
      "rmse": 0.00001,
      "argmax_match": true,
      "memory_bytes": 2260729856
    }
  ]
}
```

`baseline_memory_bytes` is optional (omit → no compression ratios). Each
entry mirrors `EndToEndEvalResult`.

## Example (real output)

```text
$ atenia search --results aqs-results.json --report --manifest
AQS Search Report

model: tinyllama-1.1b
best certified   : bf16
best useful lossy: awq_g128_a0.25

AQS certification report — model: tinyllama-1.1b (ADR-004 gate = 0.5)
candidate          status                 max_diff   argmax   compression
-------------------------------------------------------------------------
bf16               adr004_certified       0.000063     true         1.00x
awq_g128_a0.25     useful_lossy           0.889217     true         1.94x
plain_int8         failed                 1.260771    false         1.94x
best certified     : bf16
best useful lossy  : awq_g128_a0.25

Manifest Draft

schema_version: "3.0.0-draft"
model: tinyllama-1.1b
adr004:
  gate: 0.5
  certified_policy: bf16
lossy_recommendation:
  policy: awq_g128_a0.25
  reason: "argmax stable, best useful lossy compression"
policies:
  - name: bf16
    status: adr004_certified
    ...
```

## Tests (8, all green, in CI)

`report_from_json_classifies_and_selects`,
`report_human_render_is_deterministic`,
`report_manifest_render_is_draft`,
`invalid_json_is_a_clean_error`,
`missing_baseline_yields_no_compression_ratio`,
`run_search_without_results_is_user_error` (exit 2),
`run_search_missing_file_is_user_error` (exit 2),
`include_gptq_flag_parses_and_is_inert_on_results_path` (exit 0).

No real models. Local validation: `cargo test --lib --release` →
**627 passed / 0 failed / 1 ignored** (619 + 8). Binary builds; a manual
smoke run of `atenia search --results … --report --manifest` produces the
table above.

## Limitations

* **No model loading, no forward, no test fixtures, no F64 simulation.**
* **No runtime / `numcert` manifest integration** — manifest is a
  `3.0.0-draft` string.
* **No CUDA, no tier-planner, no generation, no loaders, no new
  techniques.**
* `--include-gptq` is parsed but inert on the results-file path.
* The command certifies exactly what the results file contains — it makes
  no promise about a model whose results were not honestly measured.

## Files modified

* `src/cli/search.rs` — new (command + schema + 8 tests).
* `src/cli/mod.rs` — `pub mod search;`.
* `src/bin/atenia.rs` — `Search` subcommand + `SearchArgs` + dispatch.
* `docs/HANDOFF_AQS_10.md` — this file.

No runtime, CUDA, tier-planner, generation, loader, or productive-manifest
changes.

## Next step (needs authorisation)

A future **model-driving `atenia search --model`** would require moving
the load + calibrate + forward + F64-compare evaluator into `src` (reusing
the public engine APIs the harness already uses) AND a strategy for
obtaining a per-model F64 reference. That is a substantially larger, model-
coupled step and is intentionally left for an explicitly-authorised future
milestone.
