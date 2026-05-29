# HANDOFF — MOE-11: real MoE layer assembly

Milestone: **MOE-11** (assemble a complete, runnable MoE layer — router +
routed experts + optional shared expert — from real bound tensors).
Correctness-first, CPU-only, experimental, opt-in. The MOE-2 fail-loud guard
is **unchanged** — a real MoE checkpoint still refuses to load as a model. No
Mixtral / Qwen-MoE end-to-end support is claimed. No CUDA, ROCm, Metal,
tier-planner, CLI, generation, Adapter Toolkit, batching, or optimisation.
Predecessor: MOE-0..10 (`cf4d200`).

## What was added

`src/moe/layer.rs`:

```text
MoeLayerConfig (fixture)
  + MoeWeightMap (MOE-9)  + byte resolver (MOE-10)
  → build_real_layer       → router + routed experts   (MOE-10)
  → build_shared_expert    → optional shared expert
  → RealMoeLayer
  → forward: route → top-k → dispatch → combine → (+ shared) → output
```

This is the first complete executable MoE *layer* in Atenia: real bytes →
real experts → real router → real (optional) shared expert → real forward.

## 1. How a real layer is assembled

`RealMoeLayer::assemble(map, layer_id, config, &resolve)`:

1. Validates the fixture `MoeLayerConfig`.
2. Builds the router + routed experts via the MOE-10 `build_real_layer`
   (conceptual top-k = `config.experts_per_token`).
3. **Cross-checks** the resolved topology against the config: expert count,
   `d_model`, `d_ff` must match, else `ConfigMismatch`.
4. If `config.has_shared_expert`, builds the shared expert and verifies its
   `d_model`.

The result is a `RealMoeLayer { config, routed: MoeDenseLayer, shared:
Option<MoeDenseExpert> }`.

### MoeLayerConfig (fixture, not parsed)

`MoeLayerConfig { num_experts, experts_per_token, has_shared_expert,
d_model, d_ff }`, validated (`num_experts > 0`, `0 < experts_per_token <=
num_experts`, non-zero dims). It is a **caller-supplied fixture** — MOE-11
does **not** parse `config.json` (that is Adapter Toolkit work, deferred).
Assembly cross-checks it against the real tensors so a wrong fixture fails
loudly rather than silently mis-assembling.

## 2. How the shared expert works

The MOE-2 classifier collapses every `shared_expert*` tensor to the single
role `MoeSharedExpert`, so `MoeLayerMap.shared` is an unstructured `Vec`.
`build_shared_expert` recovers gate/up/down by **name substring**
(`.gate_proj.` / `.up_proj.` / `.down_proj.`), wraps them in an
`ExpertTensors`, and reuses the MOE-10 `RealExpertTensorBinding::resolve` so
shape inference + byte resolution + validation stay in one place. A
non-projection shared tensor (e.g. a `shared_expert_gate` router) is ignored.

The shared expert:

* is **optional** (only built when `has_shared_expert`);
* is **separate** from the routed experts (its own `MoeDenseExpert`, with its
  own — typically larger — `d_ff`);
* runs on **every** token and its output is **added** on top of the routed
  combine: `output = sparse_routed(x) + shared(x)` — the Qwen-MoE /
  DeepSeek-MoE convention.

## 3. How it is validated against the sparse reference

`RealMoeLayer::forward_routed` is exactly the MOE-4 sparse reference path
(`MoeDenseLayer::forward_sparse` with `k = experts_per_token`). The tests
assert that, **with no shared expert**, the full-layer `forward` equals
`forward_sparse` over the same experts within `1e-5`; and **with** a shared
expert, `forward == forward_routed + shared.forward` within `1e-5`. So the
assembly adds no numerical drift over the certified sparse path.

## What is still missing for real Mixtral / Qwen-MoE

- **Full model**: only a single layer is assembled from an explicit reader.
  No multi-layer transformer, attention, norms, embedding/lm_head, KV cache,
  multi-token sequence.
- **Config parsing**: `num_experts` / `experts_per_token` / shared-expert
  presence still come from a hand-built fixture, not `config.json`.
- **Shared-expert gating**: some Qwen-MoE variants gate the shared expert by
  a sigmoid (`shared_expert_gate`); MOE-11 adds the shared output ungated.
- **Fail-loud lift**: the loader still refuses real MoE checkpoints; assembly
  in isolation does **not** lift the guard.
- **Graph integration**: the MOE-5..8 graph ops still use the process-global
  registry; assembly does not populate it from real data.

## What was NOT implemented

- No full model execution, no Mixtral/Qwen-MoE end-to-end, no real download.
- No `config.json` parsing, no fail-loud lift, no Adapter Toolkit / loader
  load-path change, no shared-expert gating.
- No graph / runtime wiring, no transformer, no multi-layer/multi-token.
- No CUDA/ROCm/Metal, tier-planner, CLI, generation, batching, optimisation.

## How this prepares MOE-12

MOE-12 can stack assembled `RealMoeLayer`s across layers, read the fixture
config from `config.json` (Adapter Toolkit boundary), add the shared-expert
gate where present, and/or populate the MOE-5..8 graph registry from real
assembled layers — incrementally toward a full MoE forward behind an
explicit, validated opt-in that lifts fail-loud only for the certified path.

## Tests

- `src/moe/layer.rs` — 7 unit tests (config validation, Mixtral assembly,
  forward == sparse reference, Qwen assembly with shared expert, shared adds
  on top, config mismatch rejected, missing shared expert reported).
- `tests/moe_real_layer_test.rs` — 7 integration tests on the production
  `SafetensorsReader`: `real_moe_layer_assembly`, `real_moe_layer_forward`,
  `real_moe_layer_with_shared_expert`, `router_and_expert_counts_match_config`,
  `sparse_forward_matches_assembled_layer`, `fail_loud_still_active`,
  `dense_models_still_load`.

Local validation: `cargo test --lib --release -- --test-threads=1` →
**690 passed / 0 failed / 1 ignored** (was 683). Integration suite: 7/7.

## Files modified

* `src/moe/layer.rs` — new (config + assembly + shared expert + 7 unit tests).
* `src/moe/mod.rs` — re-exports.
* `tests/moe_real_layer_test.rs` — new (7 integration tests).
* `docs/HANDOFF_MOE_11.md` — this file.

No loader load-path, Adapter Toolkit, CLI, generation, CUDA, ROCm, Metal,
tier-planner, or graph changes. Fail-loud preserved.
