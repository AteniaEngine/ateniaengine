# HANDOFF — MOE-9: real MoE data plane (metadata + expert registry)

Milestone: **MOE-9** (connect real expert tensor *metadata* into a
structured data plane). Correctness-first, CPU-only, experimental. The
MOE-2 fail-loud guard is **unchanged** (real MoE checkpoints still refuse
to load as a model); no Mixtral/Qwen-MoE full support is claimed. No CUDA,
ROCm, Metal, tier-planner, CLI, generation, optimisation, or batching. The
**Adapter Toolkit was NOT modified** (rule 15) — see scope note below.
Predecessors: MOE-0..8 (`64a7339`).

## What was connected

`src/moe/data_plane.rs` turns a checkpoint's `(tensor_name, shape)`
listing — exactly what a `SafetensorsReader.iter()` yields — into a
structured `MoeWeightMap`:

```text
checkpoint names+shapes
  → detect::classify_tensor_name (MOE-2, shared source of truth)
  → MoeWeightMap { layers: { layer_id → MoeLayerMap } }
      MoeLayerMap { router, experts: { expert_id → ExpertTensors{gate,up,down} }, shared }
  → registry lookups: router_weight(L), expert(L,E), num_experts(L), ...
```

This is the **data-plane metadata + lookup** layer: the graph/registry can
now *see* which experts exist, their roles, shapes and names, without
bespoke per-call parsing.

## What MoE metadata exists now

- `MoeTensorEntry { name, layer_id, expert_id, role, shape }` — one MoE
  tensor's structured metadata (no data).
- `ExpertTensors { gate, up, down }` + `is_complete()`.
- `MoeLayerMap { router, experts (BTreeMap), shared }` + `num_experts()`.
- `MoeWeightMap` — the whole-checkpoint registry: `from_tensors(...)`,
  `is_empty()`, `layer()`, `router_weight()`, `expert()`, `num_experts()`,
  `total_expert_tensors()`, `all_experts_complete()`.

Roles reuse the MOE-2 `TensorRole` enum (Mixtral `w1/w3/w2` →
gate/up/down; Qwen/DeepSeek `gate_proj/up_proj/down_proj`; routers; shared
experts). Deterministic (BTreeMap-backed).

## How the graph accesses experts

Via `MoeWeightMap` as an **expert registry**: `map.expert(layer_id,
expert_id) -> Option<&ExpertTensors>` and `map.router_weight(layer_id)`.
The integration test `graph_can_access_registered_expert` builds the map
from a real `SafetensorsReader` (synthetic Mixtral-named buffer) and looks
up expert `(0, 1)`'s complete gate/up/down projections with correct shapes
— demonstrating loader → data-plane → lookup end to end.

## Scope note — why the Adapter Toolkit was not touched

MOE-9 deliberately builds the data plane as a **pure `src/moe/` metadata
layer consuming `(name, shape)`**, which is the same listing the loader and
any adapter already produce. Wiring it *into* the productive loader load
path or the Adapter Toolkit would (a) require lifting the MOE-2 fail-loud
guard (forbidden this milestone) and (b) risk an Adapter Toolkit redesign
(rule 15 → STOP). So the connection is demonstrated at the
`reader.iter() → MoeWeightMap` boundary, which is honest and risk-free: no
productive path changed, fail-loud intact.

## What is still missing for real Mixtral / Qwen-MoE

- **Tensor data loading**: only names+shapes are consumed; the actual
  expert weight bytes are not loaded into `MoeDenseExpert`/the graph yet.
  A future milestone must (with fail-loud lifted under an explicit gate)
  load the data and bind `MoeWeightMap` entries to the MOE-8 conditional
  expert nodes.
- **Lifting fail-loud** behind an opt-in flag once a full MoE forward is
  wired (the loader still refuses real MoE checkpoints today).
- **Full transformer integration**: multi-layer / multi-token MoE block,
  attention + norms around the MoE layer.
- **Config parsing**: expert_count / experts_per_token / shared-expert
  config from `config.json` (Adapter Toolkit work — deferred).

## What was NOT implemented

- No tensor *data* load, no real model execution, no Mixtral/Qwen-MoE
  end-to-end.
- No fail-loud lift, no Adapter Toolkit change, no loader load-path change.
- No CUDA/ROCm/Metal, tier-planner, CLI, generation, optimisation, batching.

## How this prepares MOE-10

MOE-10 can bind a `MoeWeightMap`'s expert entries to actual loaded tensor
data and feed them into the MOE-8 `ConditionalExpert` nodes — i.e. replace
the synthetic registry layer with one populated from a real (small) MoE
checkpoint, behind an explicit opt-in that lifts fail-loud only for the
validated path.

## Tests

- `src/moe/data_plane.rs` — 7 unit tests (Mixtral/Qwen mapping, registry
  lookup, router lookup, metadata roundtrip, dense → empty map, multi-layer
  sorted/complete).
- `tests/moe_data_plane_test.rs` — 4 integration tests on a real
  `SafetensorsReader`: `graph_can_access_registered_expert`,
  `qwen_moe_data_plane_via_reader`, `fail_loud_still_active`,
  `dense_models_still_load`.

Local validation: `cargo test --lib --release` → **676 passed / 0 failed /
1 ignored** (was 669). MoE integration suites still green.

## Files modified

* `src/moe/data_plane.rs` — new (metadata + registry + 7 tests).
* `src/moe/mod.rs` — re-exports.
* `tests/moe_data_plane_test.rs` — new (4 integration tests).
* `docs/HANDOFF_MOE_9.md` — this file.

No loader load-path, Adapter Toolkit, CLI, generation, CUDA, ROCm, Metal,
or tier-planner changes. Fail-loud preserved.
