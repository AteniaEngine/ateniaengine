# HANDOFF — MOE-6: primitive MoE graph ops

Milestone: **MOE-6** (first decomposition of the MOE-5 fused op into
explicit primitives). Correctness-first, CPU-only, experimental,
synthetic-fixture-only. The fused op (`MoeSparseReference`) is **kept
intact**. No dynamic dispatch / scheduler, no real Mixtral, no loaders, no
adapters, no CLI, no generation, no CUDA, no tier-planner, no batching, no
optimisation. Predecessors: MOE-0..5 (`602b66d`).

## Primitives added

Three experimental `NodeType` variants (`src/amg/nodes.rs`), each wired
with the same minimal pattern as MOE-5 (one forward arm in
`execute_single_inner` + one arity-validator arm; backward is tape-based,
so no backward arm). The compiler confirmed the ripple was exactly those
two arms per variant — **no scheduler redesign** (rule 17 not triggered).

- **`MoeRouterSoftmax`** — 1 input (router logits `[N]`), output routing
  weights `[N]`.
- **`MoeTopK { k }`** — 1 input (routing weights `[N]`), output a flat
  selection tensor of length `2k`.
- **`MoeSparseCombine { d_model, num_experts }`** — 2 inputs (selection,
  concatenated expert outputs), output `[d_model]`.

Builder helpers: `GraphBuilder::moe_router_softmax`, `moe_topk`,
`moe_sparse_combine`.

## How RouterSoftmax works

Delegates to `crate::moe::softmax` (the same numerically-stable,
f64-internal softmax MOE-3/4 use). Validates input → CPU, output weights
sum to ≈1. Test `graph_moe_router_softmax_matches_moe_softmax` asserts the
graph output equals `moe::softmax(logits)` bit-for-bit (within 1e-6).

## How TopK is represented

The selection is a **flat `f32` tensor `[idx0, w0, idx1, w1, …]`** of
length `2k`: expert indices encoded as `f32`, interleaved with their
**renormalised** weights. This is deliberately a simple, documented,
experimental representation (indices-as-f32 is ugly but exact for the
small expert counts in scope, and trivially testable). The op delegates to
`crate::moe::top_k_routing`, preserving deterministic tie-break-by-lower-
index and renormalisation. A future milestone may switch to two separate
tensors; the flat form keeps the graph change minimal here.

## How SparseCombine works

Inputs: the `MoeTopK` selection `[idx, w, …]` and the concatenation of all
expert outputs `[num_experts * d_model]`. The op decodes the pairs and
computes `Σ_i w_i · expert_outputs[idx_i * d_model .. +d_model]` via the
pure helper `crate::moe::combine_selected`, output length `d_model`.
`d_model` / `num_experts` are carried as node params (`Eq`-safe `usize`).

## Does the primitive pipeline match the fused op?

**Yes.** `primitive_pipeline_matches_fused_reference_for_fixture` builds:

```text
logits ─► MoeRouterSoftmax ─► MoeTopK(k) ┐
expert_outputs ─────────────────────────► MoeSparseCombine ─► output
```

and asserts the output equals `MoeDenseLayer::forward_sparse(x, k)` (MOE-4,
itself pinned to the MOE-3 dense oracle) within 1e-5. The router logits and
the expert outputs are computed outside the graph (no dynamic dispatch
yet); the primitives handle softmax → top-k → combine inside the graph.

## Why there is still no dynamic dispatch

The experts are still executed **outside** any conditional/gated graph
path — their outputs are pre-computed and concatenated, then combined.
MOE-6 only makes the *routing + selection + combine* explicit as graph
ops; it does not yet gate which experts run inside the graph. Conditional
expert execution (running only the selected experts as graph subgraphs) is
the deeper scheduler work deferred to **MOE-7**. The fused
`MoeSparseReference` op remains the end-to-end path in the meantime.

## How this prepares MOE-7

MOE-7 can replace the "pre-computed concatenated expert outputs" input of
`MoeSparseCombine` with actual per-expert subgraphs executed conditionally
on the `MoeTopK` selection — the dynamic-dispatch scheduler change. Each
step validates against this primitive pipeline, which is pinned to the
MOE-4 reference and the MOE-3 oracle. The numerical contract for router,
top-k and combine is now fixed as independent graph ops.

## What was NOT implemented

- No dynamic / conditional expert dispatch (MOE-7).
- The fused `MoeSparseReference` op is **kept** (not removed, not broken).
- No real Mixtral / checkpoint load (MOE-2 fail-loud preserved).
- No loaders, adapters, CLI, generation, CUDA/ROCm/Metal, tier-planner,
  batching, optimisation, or autograd for the new ops.

## Tests

- `src/moe/sparse.rs` — `combine_selected` exercised via the new op tests.
- `tests/moe_primitive_ops_test.rs` — 8 integration tests:
  `graph_moe_router_softmax_matches_moe_softmax`,
  `graph_moe_topk_matches_sparse_topk`,
  `graph_moe_topk_is_deterministic`,
  `graph_moe_sparse_combine_matches_manual`,
  `primitive_pipeline_matches_fused_reference_for_fixture`,
  `primitive_ops_reject_bad_k`,
  `existing_moe_sparse_reference_still_passes`,
  `existing_dense_graph_tests_still_pass`.

Local validation: `cargo test --lib --release` → **669 passed / 0 failed /
1 ignored**. `moe_graph_op_test` (MOE-5) still green. No real models.

## Files modified

* `src/amg/nodes.rs` — 3 primitive `NodeType` variants.
* `src/amg/graph.rs` — 3 forward exec arms + 3 arity-validator arms.
* `src/amg/builder.rs` — 3 builder helpers.
* `src/moe/sparse.rs` — `combine_selected` pure helper.
* `src/moe/mod.rs` — re-export.
* `tests/moe_primitive_ops_test.rs` — new (8 tests).
* `docs/HANDOFF_MOE_6.md` — this file.

No loader, adapter-toolkit, CLI, generation, CUDA, ROCm, Metal, or
tier-planner changes. Fused op intact.
