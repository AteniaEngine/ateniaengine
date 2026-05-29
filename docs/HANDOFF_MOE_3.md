# HANDOFF — MOE-3: dense MoE reference execution path

Milestone: **MOE-3** (first *executable* MoE path). Correctness-first,
CPU-only, experimental, self-contained. **No** sparse dispatch, no real
top-k pruning, no optimisation, no graph/runtime/CUDA/tier-planner/
adapter-toolkit/CLI/generation touch, no MoE family, no real-model load.
Predecessors: MOE-0 audit, MOE-1 substrate (`f36d873`), MOE-2 detection
(`4b87d31`).

## What MOE-3 is

The first MoE path that actually computes something. It is a pure-`f32`
**reference** implementation (`src/moe/dense.rs`) whose only goal is
numerical correctness — the same "correctness first" discipline AQS used.
It is **not** wired into the model, the graph, or the loader; it is a
standalone, certifiable mathematical truth that a future sparse path
(MOE-4) must reproduce.

## How the dense MoE works

```text
router_logits = W_router · x            (W_router: [num_experts, d_model])
weights       = softmax(router_logits)   (one per expert; sum to 1)
for each expert e:  y_e = expert_e.forward(x)    (ALL experts always run)
output        = Σ_e weights[e] · y_e
```

Every expert runs on every token (dense). There is no pruning. (As a pure
micro-optimisation, an expert whose routing weight is *exactly* `0.0` is
skipped — numerically identical to running it and scaling by zero; this is
not top-k.)

## How the router works

`MoeDenseLayer::route(x)` computes `W_router · x` (naive f64-accumulated
matvec) then a numerically-stable `softmax` (max-subtraction, f64
internals). Result is `MoeRouterOutput { weights }`, non-negative and
summing to 1. No top-k, no sparsity — every weight is kept.

## How the experts + combine work

Each `MoeDenseExpert` is a SwiGLU MLP:
`down( silu(gate·x) ⊙ (up·x) )`, with weights row-major
`w_gate/w_up: [d_ff, d_model]`, `w_down: [d_model, d_ff]`. `silu` and all
matvecs accumulate in f64 for reference-grade accuracy.

`MoeDenseLayer::forward(x)` routes, runs all experts, and returns the
router-weighted sum `Σ_e weights[e]·expert_e(x)` (length `d_model`).

## Fixture

`build_fixture_layer()` builds the official tiny synthetic dense-MoE
fixture made executable: **4 experts, conceptual top-k = 2, dense
execution = 4**, `d_model=8`, `d_ff=16`, deterministic seeded weights, no
external dependency. `conceptual_top_k` is metadata only — the dense path
ignores it. This is the MOE-1 "tiny synthetic MoE" turned into a runnable,
F64-trivially-certifiable reference.

## Differences vs sparse MoE (what MOE-4 will change)

| | MOE-3 (this) | Sparse MoE (MOE-4) |
|---|---|---|
| Experts run | **all N** | only top-k selected |
| Routing | softmax weights, no selection | top-k selection + (re)normalised weights |
| Cost | O(N) experts/token | O(k) experts/token |
| Goal | correctness reference | efficiency, matched to MOE-3 numerically |
| Graph/runtime | none (standalone) | new ops + gated execution |

The dense path is the **oracle**: MOE-4's sparse output (for the top-k it
selects, with the same weights) must match MOE-3 restricted to those
experts.

## Limitations

- Standalone `f32` reference — not connected to the execution graph,
  loader, or any real model (MOE-2 still fails loud on real MoE
  checkpoints).
- Single-token vectors (no batching/sequence loop); batching is a trivial
  outer loop left for integration.
- Not optimised (naive matvecs); performance is explicitly a non-goal.

## How this prepares MOE-4

MOE-4 (sparse dispatch) reuses this layer as the correctness oracle:
implement top-k selection over the router weights, run only the selected
experts, renormalise, and assert the result equals the dense reference
restricted to the selected set. The graph operators + gated execution
model are the structural work MOE-0 flagged as the principal blocker.

## Tests

`src/moe/dense.rs` — 9 certification tests:
`routing_weights_sum_to_one`, `deterministic_output`,
`combine_matches_manual_calculation`,
`identical_experts_reproduce_dense_behavior`,
`zero_weight_expert_contributes_nothing`,
`expert_ordering_does_not_change_weighted_result`,
`fixture_executes_end_to_end`, `softmax_basic_properties`,
`rejects_bad_shapes`.

Local validation: `cargo test --lib --release` → **657 passed / 0 failed /
1 ignored** (was 648). No real models, no loaders, no graph.

## Files modified

* `src/moe/dense.rs` — new (router + experts + combine + fixture + 9 tests).
* `src/moe/mod.rs` — re-exports.
* `docs/HANDOFF_MOE_3.md` — this file.

No CUDA, ROCm, Metal, tier-planner, adapter-toolkit, CLI, generation,
graph, or runtime changes.
