# HANDOFF — MOE-16: numerical equivalence validation

Milestone: **MOE-16** (validate Atenia's MoE-block output numerically against
a reference, not just "it runs"). Correctness-first, CPU-only, experimental,
fixture-based. The MOE-2 fail-loud guard is **unchanged**. No CUDA, ROCm,
Metal, tier-planner, CLI, generation, Adapter Toolkit, batching, optimisation.
No model downloaded at test time, no HF/Python in CI. Predecessor: MOE-0..15
(`8148114`).

## Methodology

Compares Atenia's real layer-0 MoE-block forward (`RealMoeLayer::forward`,
f32, f64-accumulating) against **two f64 references** generated **offline**
and committed as small fixtures (ADR-004 structure: assert against an f64
reference of the defined operation; keep the external reference informative):

1. **PRIMARY (gates pass/fail)** — `atenia_ref`: an independent f64
   reimplementation of the *exact* operation Atenia performs (softmax →
   top-k, lower-index tiebreak → renormalise selected → SwiGLU experts →
   weighted sum → + shared expert **ungated**). Atenia's f32 output must match
   within `max_abs_diff < 0.5` (ADR-004) and argmax must agree. This catches
   indexing / transpose / packed-split bugs.
2. **INFORMATIVE (reported, not asserted gate)** — `hf_ref`: the **real
   HuggingFace `transformers` MoE block** (`Qwen2MoeSparseMoeBlock` /
   Mixtral) run in f64 (`block.double()`). Reveals convention gaps Atenia does
   not implement.

All metrics (`max_abs_diff`, `mean_abs_diff`, `rmse`, `argmax_match`) are
accumulated in **f64** (`src/moe/numerical.rs`).

## Reference used

- Generator: `fixtures/moe/generate_reference.py` (committed reproducibility
  artifact; **never run in CI**, never imported by Rust). It reads tiny local
  checkpoints, casts weights bf16 → **f32** (what Atenia consumes), writes the
  layer-0 MoE tensors to a small **F32 safetensors fixture**, and computes
  both references in **f64** from those f32 weights + a fixed f32 input.
- `transformers` 5.6.2 / `torch` 2.5.1+cpu / `numpy` 2.2.6.
- Fixtures committed under `fixtures/moe/<model>_layer0.{safetensors,json}`.

## Fixtures generated

| Model | Source repo | Expert format | shared | norm_topk_prob |
|---|---|---|---|---|
| `qwen15_moe` | `katuni4ka/tiny-random-qwen1.5-moe` | packed | yes | false |
| `qwen2_moe` | `hf-internal-testing/tiny-random-Qwen2MoeForCausalLM` | packed | yes | false |
| `mixtral` | `hf-internal-testing/tiny-random-MixtralForCausalLM` | packed | no | false |

(All three modern tiny checkpoints turned out to use the packed/fused expert
layout — MOE-15 is what makes them assemble.)

## Results

### PRIMARY — Atenia f32 vs f64-of-same-operation (gates pass/fail)

| Model | MaxDiff | MeanDiff | RMSE | Argmax | Pass |
|---|---|---|---|---|---|
| qwen15_moe | 5.821e-11 | 1.279e-11 | 1.898e-11 | ✅ | ✅ |
| qwen2_moe  | 8.731e-11 | 2.402e-11 | 3.195e-11 | ✅ | ✅ |
| mixtral    | 1.164e-10 | 3.143e-11 | 4.309e-11 | ✅ | ✅ |

Atenia's Rust f32 implementation reproduces the independent f64 reference of
the same operation to ~1e-10 — far inside the 0.5 tolerance. This is a direct
numerical proof that the routing, SwiGLU experts, renormalisation, shared
expert, and the **MOE-15 packed gate/up split** are all computed correctly.

### INFORMATIVE — Atenia vs HuggingFace transformers block (f64)

| Model | MaxDiff | MeanDiff | RMSE | Argmax |
|---|---|---|---|---|
| qwen15_moe | 2.369e-4 | 9.104e-5 | 1.108e-4 | ✅ |
| qwen2_moe  | 3.647e-4 | 1.383e-4 | 1.652e-4 | ✅ |
| mixtral    | 1.164e-10 | 2.731e-11 | 3.885e-11 | ✅ |

- **Mixtral** agrees with HF to ~1e-10: Mixtral's convention (softmax → top-k
  → renormalise, no shared expert) is exactly Atenia's, so they are
  numerically identical.
- **Qwen2-MoE / Qwen1.5-MoE** agree with HF within < 4e-4 and the **argmax
  matches** on all three. The small (non-zero) divergence is attributable to
  two HF conventions Atenia does not implement: `norm_topk_prob = false` (HF
  does not renormalise the selected weights; Atenia always renormalises) and a
  **sigmoid-gated shared expert** (HF scales the shared output by
  `sigmoid(shared_expert_gate · x)`; Atenia adds it ungated). On these tiny,
  near-uniform-weight checkpoints the effect is < 4e-4; on other models it
  could be larger.

## Were there divergences?

Yes, but small and explained: Atenia matches HF **exactly** for Mixtral and
within **< 4e-4 (argmax-matching)** for the Qwen-MoE family, the residual being
the `norm_topk_prob` + sigmoid-shared-gate conventions. No silent disagreement;
no failing gate. Atenia is numerically correct for the operation it defines and
numerically close to HF for all three real checkpoints.

## What was NOT implemented

- The `norm_topk_prob = false` path and the sigmoid-gated shared expert (HF
  conventions) — deferred; today Atenia always renormalises and adds the
  shared expert ungated. This is the only source of the < 4e-4 HF divergence.
- No full-model logits comparison (Atenia has no transformer stack yet — only
  the MoE block is compared).
- No fail-loud lift, no Adapter Toolkit / loader load-path / CLI / generation
  change, no CUDA/ROCm/Metal, tier-planner, batching, optimisation.

## Limitations

- Single layer-0 block per model; tiny near-random weights (the convention
  divergence may be larger on production-scale checkpoints).
- The references are f64 of f32-origin weights (matching Atenia's input), not
  the original bf16 — by design, for an apples-to-apples comparison.

## Confirmation

- **Fail-loud preserved**: the fixtures are real MoE tensors; `detect_moe`
  still fires (`fail_loud_still_active`); the productive loader is untouched.
- No Adapter Toolkit / CLI / CUDA / generation changes.

## Tests

- `src/moe/numerical.rs` — 7 unit tests (metric correctness, determinism,
  argmax gate, tolerance gate, report build).
- `tests/moe_numerical_equivalence_test.rs` — 7 integration tests:
  `qwen15_moe_matches_reference`, `qwen2_moe_matches_reference`,
  `mixtral_matches_reference`, `validation_report_builds`,
  `metrics_are_deterministic`, `fail_loud_still_active`,
  `dense_models_still_load`.

Local validation: `cargo test --lib --release -- --test-threads=1` →
**722 passed / 0 failed / 1 ignored** (was 715). Numerical suite: 7/7.

## Files modified

* `src/moe/numerical.rs` — new (metrics + report + 7 unit tests).
* `src/moe/mod.rs` — re-exports.
* `tests/moe_numerical_equivalence_test.rs` — new (7 integration tests).
* `fixtures/moe/generate_reference.py` — new (offline generator, not run in CI).
* `fixtures/moe/{qwen15_moe,qwen2_moe,mixtral}_layer0.{safetensors,json}` — new
  reference fixtures.
* `docs/HANDOFF_MOE_16.md` — this file.

No loader load-path, Adapter Toolkit, CLI, generation, CUDA, ROCm, Metal,
tier-planner, or graph changes. Fail-loud preserved.

## How this prepares MOE-17

A natural next step is to implement the HF conventions behind a flag
(`norm_topk_prob`, sigmoid-gated shared expert) and re-run MOE-16 to drive the
HF divergence to ~1e-10 for the Qwen-MoE family as well — turning "close to HF"
into "matches HF" across all supported families.
