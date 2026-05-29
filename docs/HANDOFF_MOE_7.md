# HANDOFF — MOE-7: experimental dynamic expert dispatch

Milestone: **MOE-7** (conditional execution of selected experts).
Correctness-first, CPU-only, experimental, synthetic-fixture-only. The
fused op (`MoeSparseReference`) and the MOE-6 primitives are **kept
intact**. No real Mixtral, no checkpoint loading (MOE-2 fail-loud
preserved), no loaders, no adapters, no CLI, no generation, no CUDA, no
tier-planner, no batching, no optimisation. Predecessors: MOE-0..6
(`b368e61`).

## Design chosen: **Option A — fused dynamic-dispatch op**

A single experimental node, `NodeType::MoeDynamicDispatch { layer_id,
d_model, num_experts }`, whose executor runs **only the selected experts**.
Option B (a general conditional-subgraph scheduler) was rejected per the
milestone's strong preference and rule 15 — it would be a global scheduler
redesign.

### Why no global scheduler redesign was needed

The conditionality lives **inside the op's executor arm**, not in the
graph plan. The static execution plan is unchanged; when the node runs, it
reads the selection tensor and calls `forward` on **only** those experts.
The compiler-measured ripple was the same minimal pattern as MOE-5/6: one
forward arm in `execute_single_inner` + one arity-validator arm (backward
is tape-based → no arm). Rule 15 was not triggered.

## What was implemented

- `NodeType::MoeDynamicDispatch { layer_id, d_model, num_experts }`
  (`src/amg/nodes.rs`), 2 inputs: `[input_vector, selection]`.
- Executor arm (`src/amg/graph.rs`) → `crate::moe::execute_dynamic_dispatch`.
- Arity-validator arm (2 inputs).
- `GraphBuilder::moe_dynamic_dispatch(input_id, selection_id, layer_id,
  d_model, num_experts)`.
- `src/moe/graph_op.rs`: `execute_dynamic_dispatch(layer_id, input,
  selection) -> DynamicDispatchOutput { output, executed_experts }`.

## How only the selected experts run

`execute_dynamic_dispatch` decodes the `MoeTopK` selection tensor
`[idx0, w0, idx1, w1, …]`, then loops over **only those pairs**, calling
`layer.experts[idx].forward(input)` for each selected `idx` and
accumulating `Σ wᵢ · yᵢ`. Experts not in the selection are **never**
`forward`-ed. `executed_experts` returns the exact ids that ran — the test
`dynamic_dispatch_executes_only_selected_experts` asserts
`executed_experts.len() == k`, equals the top-k indices, and is strictly
less than `num_experts`. So conditional execution is proven by
construction, not just by output equality.

## Validation against MOE-6 / MOE-5 / MOE-4

The required equivalence chain is established:

```text
graph MoeDynamicDispatch  ==  fused MoeSparseReference (MOE-5)
                          ==  forward_sparse (MOE-4)
                          ==  dense restricted oracle (MOE-3)
```

- `dynamic_dispatch_matches_fused_reference`: full graph
  `logits → RouterSoftmax → TopK → MoeDynamicDispatch` equals the fused
  `MoeSparseReference` op (1e-5).
- `primitive_router_topk_dispatch_pipeline_matches_fused`: the same
  pipeline equals `forward_sparse` (1e-5).
- `dynamic_dispatch_matches_sparse_reference`: the registry call equals
  `forward_sparse`.

Note: the MOE-6 `MoeSparseCombine` (which takes *pre-computed* concatenated
expert outputs) is now superseded for the end-to-end path by
`MoeDynamicDispatch` (which *executes* the selected experts internally),
but `MoeSparseCombine` is kept — both remain valid graph ops.

## What was NOT implemented

- No general conditional-subgraph scheduler (each expert as its own gated
  graph subgraph) — the dispatch is internal to one fused op. A
  fully-graph-native per-expert dispatch is deferred to MOE-8.
- No real Mixtral / checkpoint load (MOE-2 fail-loud preserved); the
  registry is populated only by tests / experimental callers.
- No loaders, adapters, CLI, generation, CUDA/ROCm/Metal, tier-planner,
  batching, optimisation, or autograd for the op.

## How this prepares MOE-8

MOE-8 can lift expert execution into actual graph subgraphs gated by the
selection (a real conditional scheduler), validating each step against
this fused dynamic-dispatch op — itself pinned to the MOE-4 reference and
the MOE-3 oracle. The dispatch *semantics* (run only selected, combine by
renormalised weights) are now fixed and certified.

## Tests

- `src/moe/graph_op.rs` — registry roundtrip + unknown-id (unchanged from
  MOE-5, still green).
- `tests/moe_dynamic_dispatch_test.rs` — 8 integration tests:
  `dynamic_dispatch_matches_sparse_reference`,
  `dynamic_dispatch_matches_fused_reference`,
  `dynamic_dispatch_executes_only_selected_experts`,
  `dynamic_dispatch_rejects_bad_selection`,
  `dynamic_dispatch_rejects_unknown_layer`,
  `dynamic_dispatch_is_deterministic`,
  `primitive_router_topk_dispatch_pipeline_matches_fused`,
  `existing_fused_op_still_passes`.

Local validation: `cargo test --lib --release` → **669 passed / 0 failed /
1 ignored**. `moe_graph_op_test` (MOE-5) + `moe_primitive_ops_test`
(MOE-6) still green.

## Files modified

* `src/amg/nodes.rs` — `MoeDynamicDispatch` variant.
* `src/amg/graph.rs` — forward exec arm + arity-validator arm.
* `src/amg/builder.rs` — `moe_dynamic_dispatch` helper.
* `src/moe/graph_op.rs` — `execute_dynamic_dispatch` + `DynamicDispatchOutput`.
* `src/moe/mod.rs` — re-exports.
* `tests/moe_dynamic_dispatch_test.rs` — new (8 tests).
* `docs/HANDOFF_MOE_7.md` — this file.

No loader, adapter-toolkit, CLI, generation, CUDA, ROCm, Metal, or
tier-planner changes. Fused op + primitives intact.
