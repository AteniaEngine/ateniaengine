# HANDOFF — MOE-18: automatic convention selection

Milestone: **MOE-18** (infer the MoE execution convention from existing
metadata so the caller no longer has to pick one). Correctness-first,
CPU-only, experimental. The MOE-2 fail-loud guard is **unchanged**. The MoE
math is **unchanged** (MOE-17 numerics preserved exactly per convention). No
Adapter Toolkit, CLI, generation, CUDA, ROCm, Metal, tier-planner, batching,
optimisation. Predecessor: MOE-0..17 (`f1ef6fd`).

## How the convention is detected

`MoeConventionResolver` (in `src/moe/convention.rs`) reads **metadata that
already exists** — no `config.json` parsing, no new metadata:

```text
MoeWeightMap  →  any layer has a `shared_expert_gate` tensor?
                   yes → MoeExecutionConvention::HuggingFaceQwen
                   no  → MoeExecutionConvention::Atenia
```

Two equivalent entry points:

- `MoeConventionResolver::from_weight_map(&MoeWeightMap)` — scans the metadata
  map for a tensor whose name contains `shared_expert_gate`.
- `RealMoeLayer::resolve_convention()` — uses the assembly-captured
  `shared_gate: Option<Vec<f32>>` (set by MOE-17). `Some` → `HuggingFaceQwen`,
  `None` → `Atenia`.

`RealMoeLayer::forward_auto(x)` runs `forward_with(x, resolve_convention())`.
The explicit `forward_with(x, convention)` remains available unchanged.

## Signals used

| Model | Convention | Detection signal |
|---|---|---|
| qwen15_moe | HuggingFaceQwen | `…mlp.shared_expert_gate.weight` present |
| qwen2_moe  | HuggingFaceQwen | `…mlp.shared_expert_gate.weight` present |
| mixtral    | Atenia | no shared-expert gate tensor |

The `shared_expert_gate` tensor is precisely what the `HuggingFaceQwen`
forward consumes (the sigmoid gate weight), so detection and execution rest on
the same fact. A checkpoint with a shared expert but **no** gate tensor
resolves to `Atenia` (ungated) — the safe default.

## Why the math is not changed

MOE-18 only **chooses** which already-correct forward runs. It does not touch
`top_k_routing*`, the SwiGLU experts, the renormalisation flag, or the
sigmoid gate — those are MOE-17. `forward_auto` is exactly
`forward_with(resolve_convention())`, so each model's numbers are bit-for-bit
the MOE-17 explicit-convention results.

## Results

`forward_auto` reproduces the MOE-17 parity numbers against the HF
transformers reference (argmax matched in all cases):

| Model | Resolved convention | `forward_auto` vs HF (`max_abs_diff`) |
|---|---|---|
| qwen15_moe | HuggingFaceQwen | ~2.9e-11 |
| qwen2_moe  | HuggingFaceQwen | ~5.8e-11 |
| mixtral    | Atenia | ~1.2e-10 |

The integration test asserts `forward_auto == forward_with(explicit)` per
model and that the HF parity holds (`max_abs_diff < 1e-6`, argmax match).

## What was NOT implemented

- No `config.json` parsing / Adapter Toolkit integration (signal is purely the
  tensor-name metadata; config-driven selection is deferred).
- No change to the MoE math or to the explicit `forward_with` API.
- No DeepSeek-MoE-specific convention (only Qwen vs Mixtral/Atenia today).
- No fail-loud lift, no CLI / generation / CUDA / ROCm / Metal / tier-planner /
  batching / optimisation.

## Confirmation

- **Fail-loud preserved** (`fail_loud_still_active`): fixtures are real MoE
  tensors; `detect_moe` still fires; the productive loader is untouched.
- No Adapter Toolkit / CLI / CUDA / generation changes.
- Atenia remains the default for any checkpoint without a shared-expert gate.

## Tests

- `src/moe/convention.rs` — 4 unit tests (qwen-with-gate → HF, mixtral → Atenia,
  shared-without-gate → Atenia, helper).
- `tests/moe_auto_convention_test.rs` — 7 integration tests:
  `resolver_detects_qwen_convention`, `resolver_detects_mixtral_convention`,
  `auto_forward_matches_explicit_qwen`, `auto_forward_matches_explicit_mixtral`,
  `metrics_preserved`, `fail_loud_still_active`, `dense_models_still_load`.

Local validation: `cargo test --lib --release -- --test-threads=1` →
**726 passed / 0 failed / 1 ignored** (was 722). Auto-convention suite: 7/7.
MOE-16/17 suites still green (math + defaults unchanged).

## Files modified

* `src/moe/convention.rs` — new (resolver + 4 unit tests).
* `src/moe/layer.rs` — `RealMoeLayer::resolve_convention` + `forward_auto`.
* `src/moe/mod.rs` — re-exports.
* `tests/moe_auto_convention_test.rs` — new (7 integration tests).
* `docs/HANDOFF_MOE_18.md` — this file.

No loader load-path, Adapter Toolkit, CLI, generation, CUDA, ROCm, Metal,
tier-planner, or graph changes. Fail-loud preserved; MoE math + Atenia default
preserved.

## How this prepares MOE-19

When the productive Adapter Toolkit / loader path is eventually allowed to
carry MoE, the same resolver can be fed the real config (`norm_topk_prob`,
shared-expert presence) to confirm or override the tensor-name heuristic —
turning the metadata signal into a config-validated convention selection.
