# HANDOFF — MOE-5: experimental graph sparse-MoE reference op

Milestone: **MOE-5** (first graph/runtime integration of MoE).
Correctness-first, CPU-only, experimental, synthetic-fixture-only. **No**
real Mixtral, no real checkpoint loading (MOE-2 fail-loud preserved), no
adapter toolkit, no CLI, no generation, no CUDA, no tier-planner, no
optimisation, no batching. Predecessors: MOE-0..4 (`b90e1a8`).

## Route chosen: **fused op (Option A)** — not primitive ops, not STOP

A single experimental graph node, `NodeType::MoeSparseReference { layer_id,
k }`, whose forward delegates to the already-certified
`MoeDenseLayer::forward_sparse` (MOE-4). Option B (separate Router / TopK /
Dispatch / Combine nodes with a dynamic scheduler) was deliberately
deferred to MOE-6 per the milestone's strong preference. The fused op is
the minimal viable graph change and was confirmed feasible — **not** the
"amplio rediseño" rule-16 would have STOPped on.

### Why it was minimal (ripple measured by the compiler)

`NodeType` derives `Eq`, so the variant carries only `Eq`-safe ids
(`layer_id: u32`, `k: u32`); the actual layer lives in a small process
registry. Adding the variant forced exactly **two** match arms in
`graph.rs` (the compiler found them):

1. `execute_single_inner` — the forward execution arm (the real work).
2. the input-arity validator (`MoeSparseReference` takes 1 input).

Backward is **tape-based**, not a `NodeType` match, so no backward arm was
needed (the op records no `BackOp` — it is inference-only). Every other
`NodeType` match in the codebase (scheduler, fusion, pattern, gpu
executor, op router, gpu planner) already has a `_ =>` wildcard, so nothing
else changed.

## What was integrated

- `NodeType::MoeSparseReference { layer_id: u32, k: u32 }` (`src/amg/nodes.rs`).
- Forward arm in `Graph::execute_single_inner` (`src/amg/graph.rs`):
  reads the single input tensor, `ensure_cpu`, calls
  `crate::moe::graph_op::execute_sparse_reference(layer_id, x, k)`, sets the
  output. No backward.
- Arity-validator arm (`src/amg/graph.rs`): `MoeSparseReference` ⇒ 1 input.
- `GraphBuilder::moe_sparse_reference(x_id, layer_id, k)` builder helper
  (`src/amg/builder.rs`).
- `src/moe/graph_op.rs`: a process-global experimental layer registry
  (`register_layer → layer_id`, `get_layer`, `execute_sparse_reference`).

## How it connects to `src/moe`

The graph op is a thin bridge: the executor arm calls
`moe::graph_op::execute_sparse_reference`, which looks up the registered
`MoeDenseLayer` and calls `forward_sparse(input, k)` — the exact MOE-4
path. No MoE math lives in the graph; the graph only routes the call.

## How it is validated against MOE-4

`tests/moe_graph_op_test.rs` builds a graph
`input → MoeSparseReference(layer_id, k) → output`, executes it on the
synthetic fixture, and asserts the graph output equals
`layer.forward_sparse(x, k).output` within `1e-5`. Since MOE-4 is itself
pinned to the MOE-3 dense oracle, the chain is:

```text
graph MoeSparseReference  ==  forward_sparse (MOE-4)  ==  dense restricted oracle (MOE-3)
```

Also verified: `k = num_experts` ⇒ graph output == dense `forward`;
determinism; bad `k` rejected at the op boundary; a normal non-MoE graph
(`input → silu → output`) still executes correctly after the variant was
added.

## Why fail-loud is still active

MOE-5 changed **nothing** in the loader. The MOE-2 guard
(`LoaderError::MoeUnsupported`) still fires on real MoE checkpoints; the
registry that backs `MoeSparseReference` is populated only by tests /
explicit experimental callers, never by a real load. Test
`moe_checkpoint_still_fails_loud` re-asserts detection is unchanged.

## What was NOT implemented

- No real Mixtral / Qwen-MoE / DeepSeek family; no real checkpoint load.
- No primitive Router / TopK / Dispatch / Combine graph nodes (MOE-6).
- No dynamic / sparse scheduler — the fused op runs synchronously like any
  other node; sparsity is internal to the reference call.
- No backward / autograd for the MoE op (inference-only).
- No batching, no optimisation, no CUDA/ROCm/Metal, no tier-planner, no
  adapter toolkit, no CLI, no generation.

## How this prepares MOE-6

MOE-6 can decompose the fused op into primitive graph nodes (Router,
TopK, gathered expert execution, Combine) with the scheduler changes that
implies, validating each step against this fused op — which is itself
pinned to the MOE-4 reference and the MOE-3 oracle. The numerical contract
is fully established before the deeper scheduler work begins.

## Tests

- `src/moe/graph_op.rs` — 2 unit tests (registry roundtrip, unknown id).
- `tests/moe_graph_op_test.rs` — 6 integration tests
  (`graph_moe_sparse_matches_reference`, `graph_moe_sparse_is_deterministic`,
  `graph_moe_sparse_k_equals_all_matches_dense`,
  `graph_moe_sparse_rejects_bad_k`,
  `graph_moe_sparse_preserves_existing_dense_graph`,
  `moe_checkpoint_still_fails_loud`).

Local validation: `cargo test --lib --release` → **669 passed / 0 failed /
1 ignored** (was 667). `moe_loader_failloud_test` still green (fail-loud
preserved).

## Files modified

* `src/amg/nodes.rs` — `NodeType::MoeSparseReference` variant.
* `src/amg/graph.rs` — forward exec arm + arity-validator arm.
* `src/amg/builder.rs` — `moe_sparse_reference` builder helper.
* `src/moe/graph_op.rs` — new (layer registry + execute bridge + 2 tests).
* `src/moe/mod.rs` — re-exports.
* `tests/moe_graph_op_test.rs` — new (6 integration tests).
* `docs/HANDOFF_MOE_5.md` — this file.

No loader, adapter-toolkit, CLI, generation, CUDA, ROCm, Metal, or
tier-planner changes.
