# HANDOFF — MOE-13: real MoE checkpoint validation harness

Milestone: **MOE-13** (a single opt-in path that validates a real-format MoE
checkpoint end-to-end: metadata → derived config → stack → minimal forward →
report). Correctness-first, CPU-only, experimental, **validation-only**. The
MOE-2 fail-loud guard is **unchanged** — a real MoE checkpoint still refuses
to load as a model. No CUDA, ROCm, Metal, tier-planner, CLI, generation,
Adapter Toolkit, batching, or optimisation. Predecessor: MOE-0..12 (`549cffc`).

## 1. What checkpoint was validated

**Target family: Qwen-MoE** (priority 1 of the spec). A real Qwen1.5-MoE
(~14B params) or Mixtral (~47B) does **not** fit CI and is **not** downloaded
(forbidden by scope). The harness is therefore **target-agnostic**: it
validates *any* safetensors MoE checkpoint supplied as a `(name, shape)`
listing + byte resolver. In CI it is exercised against a **tiny synthetic
checkpoint built in-memory that uses real Qwen-MoE tensor naming**
(`model.layers.{L}.mlp.gate` / `.experts.{E}.{gate,up,down}_proj` /
`.shared_expert.*`), multi-layer, with shared experts.

> **What `forward_pass_ok = true` does NOT mean.** It means "metadata → stack
> → forward executed and produced finite numbers", NOT "this model is correct,
> supported, or numerically certified". No real Mixtral / Qwen-MoE is claimed
> supported.

## 2. What real parts executed

The harness chains the already-certified MOE-9..12 pieces over the
checkpoint's real tensor bytes:

- **Metadata plane** (MOE-9): `MoeWeightMap` from `reader.iter()`.
- **Config derivation (new, MOE-13)**: a per-layer `MoeLayerConfig` derived
  **purely from tensor shapes** — `num_experts` from the expert count,
  `d_ff`/`d_model` from an expert gate shape `[d_ff, d_model]`,
  `has_shared_expert` from shared-tensor presence. `experts_per_token` is the
  one value not encoded in the weights, so it is supplied by the caller and
  clamped to `[1, num_experts]`. **No `config.json` is parsed.**
- **Real binding + assembly** (MOE-10/11): router + routed experts + optional
  shared expert, from real bytes.
- **Stack** (MOE-12): all layers composed sequentially.
- **Minimal opt-in forward**: a deterministic probe vector is run through the
  stack; output length + finiteness checked.

The result is a `ValidationReport { layers_detected, experts_detected,
shared_experts, d_model, forward_pass_ok, errors }`.

## 3. The opt-in / validation-only boundary

`RealMoeCheckpointValidation::validate(...)` is **not** the productive loader,
runtime, or inference path. It consumes the same `(name, shape)` listing a
reader already exposes — exactly as MOE-9..12 do — and never loads a model
"as a model". The MOE-2 loader guard still fires on the same reader
(re-asserted by `fail_loud_still_active`), so a real MoE checkpoint continues
to refuse to load normally. The harness never panics: every failure is
recorded in `report.errors` and reflected in `forward_pass_ok`.

## What is still missing for real Mixtral / Qwen-MoE

- **A real checkpoint run**: CI uses a synthetic-format checkpoint; validating
  an actual downloaded Qwen-MoE/Mixtral needs an out-of-CI, opt-in download +
  a host that fits it.
- **Transformer structure**: no residuals, RMSNorm, attention, embeddings,
  lm_head, positional encoding, multi-token / KV cache. The stack is a pure
  MoE-layer composition, not a model.
- **`config.json`**: `experts_per_token` (and validation of the derived
  topology against published config) is not read (Adapter Toolkit, deferred).
- **Shared-expert gating** (sigmoid in some Qwen variants): the shared output
  is added ungated.
- **Fail-loud lift**: the loader still refuses real MoE checkpoints; the
  validation path does **not** lift the guard.
- **Graph integration**: the MOE-5..8 graph ops still use the process-global
  registry; validation does not populate it.

## What was NOT implemented

- No real model download, no full model execution, no transformer.
- No `config.json` parsing, no fail-loud lift, no Adapter Toolkit / loader
  load-path change, no shared-expert gating.
- No CUDA/ROCm/Metal, tier-planner, CLI, generation, batching, optimisation.

## How this prepares MOE-14

MOE-14 can wrap the validated stack in transformer-block structure
(RMSNorm + residual, then attention), parse `config.json` to supply
`experts_per_token` + cross-check the derived topology, and/or run the
harness against a real (opt-in, out-of-CI) small MoE checkpoint — moving
toward a full MoE forward behind an explicit, validated opt-in that lifts
fail-loud only for the certified path.

## Tests

- `src/moe/validation.rs` — 7 unit tests (report builds, expert detection,
  per-layer config derivation, forward runs, missing-data recorded as error,
  experts_per_token clamping, dense → empty report).
- `tests/moe_real_checkpoint_validation_test.rs` — 6 integration tests on the
  production `SafetensorsReader`: `validation_report_builds`,
  `validation_detects_experts`, `validation_builds_stack`,
  `validation_runs_forward`, `fail_loud_still_active`, `dense_models_still_load`.

Local validation: `cargo test --lib --release -- --test-threads=1` →
**706 passed / 0 failed / 1 ignored** (was 699). Integration suite: 6/6.

## Files modified

* `src/moe/validation.rs` — new (harness + report + config derivation + 7 unit tests).
* `src/moe/mod.rs` — re-exports.
* `tests/moe_real_checkpoint_validation_test.rs` — new (6 integration tests).
* `docs/HANDOFF_MOE_13.md` — this file.

No loader load-path, Adapter Toolkit, CLI, generation, CUDA, ROCm, Metal,
tier-planner, or graph changes. Fail-loud preserved.
