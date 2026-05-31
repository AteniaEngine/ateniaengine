# HANDOFF — MOE-FULL-4: MoE block as AMG graph operator

Milestone: **MOE-FULL-4** (run a whole certified `RealMoeLayer` as a single
AMG graph node — the substrate bridge MOE-FULL-1 identified as the pivotal
design choice). **Experimental, CPU-only, registry-backed.** No model
execution beyond one MoE block, no generation, no attention, no decoder, no
loader/Adapter-Toolkit/CUDA/CLI changes, no fail-loud lift. Predecessor:
MOE-FULL-3 (`21eff37`).

## What op was added

`NodeType::MoeRealLayerReference { layer_id: u32 }` (in `src/amg/nodes.rs`) —
an experimental fused op that runs an entire real MoE layer (router + routed
experts + optional shared expert, MOE-11) as one graph node. It takes a single
input (the token vector `[d_model]`) and outputs `[d_model]`.

This is the chosen design from the MOE-FULL-1 audit: **wrap the certified
imperative `RealMoeLayer` as one graph custom-op** (lowest risk, reuses the
certified math) rather than re-deriving the MoE block as native primitive
graph nodes. It follows the exact pattern of the existing
`MoeSparseReference` / `MoeDynamicDispatch` / `ConditionalExpert` ops.

## How it wraps `RealMoeLayer`

`RealMoeLayer` carries f32 weights, so it is not `Eq` and cannot live inside a
`NodeType` variant (which derives `Eq`). As with the MOE-5..8 ops, the node
carries only the `Eq`-safe `layer_id: u32`, and the real layer lives in a new
process-global registry in `src/moe/graph_op.rs`:

```rust
register_real_moe_layer(layer) -> u32         // store + get an id
get_real_moe_layer(id) -> Option<Arc<RealMoeLayer>>
execute_real_moe_layer(id, input) -> Vec<f32> // RealMoeLayer::forward_auto
execute_real_moe_layer_with(id, input, conv)  // explicit convention (MOE-17)
```

The graph executor arm (`src/amg/graph.rs`, in `execute_single_inner`)
materialises the input to CPU, calls `execute_real_moe_layer(layer_id, x)` —
which delegates to the certified `RealMoeLayer::forward_auto` (MOE-18
auto-resolved convention) — and sets the node output. Inference-only: no
backward op (not differentiable here). Builder helper:
`GraphBuilder::moe_real_layer_reference(x_id, layer_id)` in `src/amg/builder.rs`.

Ripple was minimal and identical to prior MoE ops: +1 `NodeType` variant, +1
arity-validator arm, +1 executor arm, +1 builder helper, +registry. The
compiler confirmed all other `NodeType` matches (scheduler, fusion, gpu) carry
`_ =>` wildcards — no graph/runtime redesign.

## Fixture used

The committed real Mixtral layer-0 fixture
`fixtures/moe/mixtral_layer0.{safetensors,json}` (~385 KB, from
`hf-internal-testing/tiny-random-MixtralForCausalLM`, packed experts, 4
experts, d_model 64, no shared expert). A real checkpoint slice, already in the
repo from MIXTRAL-CERT-1 — **no model downloaded, no model committed**.

## Numerical validation

The integration test builds `input → MoeRealLayerReference → output` in a real
AMG `Graph`, executes it, and compares:

- **vs `RealMoeLayer::forward_auto(input)`** (the certified reference):
  `max_abs_diff < 1e-5` ✅ (the graph op equals the certified layer).
- **vs the MOE-16 HuggingFace f64 reference** (transitively): `argmax_match`
  AND `max_abs_diff < 0.5` ✅ (Mixtral → Atenia convention; ~1e-10 in
  MIXTRAL-CERT-1).

## What is NOT activated

- Only one MoE **block** runs in the graph. No attention, norms, embeddings,
  lm_head, KV cache, decoder, multi-token, or generation.
- The op is registry/test-only; no productive path constructs the node.
- No fail-loud lift — the productive loader still refuses MoE checkpoints
  (`LoaderError::MoeUnsupported`), re-asserted by `fail_loud_still_active`.
- No Adapter Toolkit, loader load-path, CUDA, or CLI changes. No optimisation,
  no batching. No "Mixtral supported" claim.

## Why fail-loud stays active

The graph op reads its `RealMoeLayer` from the experimental registry, which is
populated only by tests / explicit opt-in callers — never by the productive
loader. The MOE-2 guard in `weight_mapper.rs` is untouched, so a real MoE
checkpoint loaded the normal way still fails loud. The op is reachable only by
code that deliberately assembles and registers a layer.

## How this prepares MOE-FULL-5

MOE-FULL-5 (one full Mixtral decoder layer) can now place a
`MoeRealLayerReference` node where a dense block's SwiGLU FFN sits, surrounded
by the reused dense attention + RMSNorm + residual graph nodes, and compare one
real decoder layer to a HF single-layer reference. The MoE block already runs
as a graph node, so the remaining work is wiring the (already-existing) dense
surround around it.

## Tests

- `src/moe/graph_op.rs` — 2 new unit tests: `real_register_and_execute_roundtrip`,
  `real_unknown_layer_id_errors` (4 total in the module).
- `tests/moe_real_layer_graph_op_test.rs` — 7 integration tests:
  `graph_real_moe_layer_matches_reference`, `graph_real_moe_layer_is_deterministic`,
  `graph_real_moe_layer_rejects_unknown_layer`, `graph_real_moe_layer_rejects_bad_input_dim`,
  `graph_real_moe_layer_with_explicit_convention`, `fail_loud_still_active`,
  `dense_models_still_load`.

Local validation (real output, exit 0): full lib suite
`cargo test --lib --release -- --test-threads=1` → **746 passed / 0 failed /
1 ignored** (was 744; +2). Integration suite 7/7. Existing MoE graph suites
still green: `moe_graph_op_test` 6/6, `moe_primitive_ops_test` 8/8,
`moe_dynamic_dispatch_test` 8/8, `moe_conditional_expert_test` 8/8.

## Files modified

* `src/amg/nodes.rs` — `NodeType::MoeRealLayerReference` variant.
* `src/amg/graph.rs` — arity-validator arm + executor arm.
* `src/amg/builder.rs` — `moe_real_layer_reference` helper.
* `src/moe/graph_op.rs` — real-MoE-layer registry + execute fns + 2 unit tests.
* `src/moe/mod.rs` — re-exports.
* `tests/moe_real_layer_graph_op_test.rs` — new (7 integration tests).
* `docs/HANDOFF_MOE_FULL_4.md` — this file.
* `docs/MOE_FULL_PATH_AUDIT.md` — progress note.

No generation, loader load-path, Adapter Toolkit, CUDA, ROCm, Metal,
tier-planner, or CLI changes. Fail-loud preserved.
