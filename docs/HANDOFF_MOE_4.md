# HANDOFF — MOE-4: sparse MoE reference execution path

Milestone: **MOE-4** (real top-k sparse MoE, validated against the MOE-3
dense oracle). Correctness-first, CPU-only, experimental, **still isolated
in `src/moe/`** — no graph, no runtime, no loader, no CUDA, no
adapter-toolkit, no CLI. Predecessors: MOE-0..3 (`db1aa81`).

## What MOE-4 is

The first MoE path with genuine **sparsity**: it selects the top-k experts
per token and executes **only** those, instead of MOE-3's all-experts
dense path. It lives in `src/moe/sparse.rs` and is validated bit-for-bit
(within f32 tolerance) against the MOE-3 dense path restricted to the same
experts.

```text
router_logits = W_router · x
weights       = softmax(router_logits)              (all experts)
(idx, w)      = top_k(weights, k)                    (k highest; ties → lower index)
w'            = w / Σ w                              (renormalise selected)
output        = Σ_{e ∈ idx} w'[e] · expert_e(x)      (selected experts ONLY)
```

## How top-k works

`top_k_routing(weights, k) -> TopKSelection { indices, weights }`:

- Validates `0 < k <= weights.len()`, all weights finite and non-negative.
- Ranks experts by `(weight desc, index asc)` → **deterministic
  tie-breaking by lower expert index**.
- Takes the top k, then sorts the selected `indices` ascending for stable
  output (with weights kept paired).
- Renormalises the selected weights to sum 1; errors
  (`SelectedSumNonPositive`) if their sum is ≤ 0.

Errors: `ZeroK`, `KExceedsExperts`, `NonFiniteWeight{index}`,
`NegativeWeight{index}`, `SelectedSumNonPositive`, `Dense(MoeDenseError)`.

## How weights are renormalised

After top-k, only k of the softmax weights remain and their sum is < 1.
They are divided by their own sum so the **selected** weights sum to 1
(`w' = w / Σ w`, f64 internals). This is the standard Mixtral / Qwen-MoE
convention (softmax → top-k → renormalise) and — crucially — it makes the
sparse output directly comparable to the dense reference restricted to the
same experts.

## How sparse forward works

`MoeDenseLayer::forward_sparse(input, k) -> MoeSparseForwardOutput { output,
selected_experts }`. It reuses the MOE-3 layer's router and experts (no
duplication): route → `top_k_routing` → run only the selected experts →
weighted combine. `selected_experts` is returned so callers can verify
that **only** the selected experts were executed.

## How it is validated against the dense oracle

`MoeDenseLayer::forward_dense_restricted(input, selected_indices)` is the
oracle: it computes the full softmax, keeps the given indices' weights,
renormalises them to sum 1, and combines those experts — the dense path
restricted to a subset. The test
`sparse_forward_matches_dense_restricted_oracle` asserts
`forward_sparse(x, k).output == forward_dense_restricted(x, top_k_indices)`
within `1e-5`. Two more equivalences are proven:

- `k = num_experts` ⇒ sparse output == MOE-3 `forward` (renorm over the
  full softmax is identity).
- sparse executes exactly the top-k indices of the full softmax.

This makes MOE-3 the mathematical oracle for MOE-4, exactly as planned.

## What was NOT implemented

- **No graph / runtime integration** — the sparse path is a standalone
  `f32` reference; gated execution inside the execution graph is **MOE-5**
  (the structural blocker MOE-0 flagged).
- No CUDA / ROCm / Metal, no tier-planner, no adapter-toolkit, no loader,
  no CLI, no generation, no MoE family, no real-model load.
- No batching/sequence loop (single-token reference); no optimisation.
- Token-choice routing only (per-token top-k); no expert-choice / capacity
  / load-balancing — out of scope.

## How this prepares MOE-5

MOE-5 (graph/runtime integration) can now lift `forward_sparse` into the
execution graph: add the router + top-k + gathered expert execution as
graph operators with conditional/gated execution, and assert the graph
output equals this isolated `forward_sparse` reference (which is itself
pinned to the MOE-3 dense oracle). The numerical contract is fully
established before any runtime change.

## Tests

`src/moe/sparse.rs` — 10 tests: `top_k_selects_largest_weights`,
`top_k_tie_breaks_by_lower_index`, `top_k_rejects_zero_k`,
`top_k_rejects_k_larger_than_experts`, `top_k_weights_are_renormalized`,
`top_k_rejects_non_finite_weights`,
`sparse_forward_matches_dense_restricted_oracle`,
`sparse_forward_executes_only_selected_experts`,
`sparse_forward_k_equals_all_matches_dense_forward`,
`sparse_forward_is_deterministic`.

Local validation: `cargo test --lib --release` → **667 passed / 0 failed /
1 ignored** (was 657). No real models, no loaders, no graph.

## Files modified

* `src/moe/sparse.rs` — new (top-k + sparse forward + oracle + 10 tests).
* `src/moe/mod.rs` — re-exports.
* `docs/HANDOFF_MOE_4.md` — this file.

(`src/moe/dense.rs` unchanged — `forward_sparse` / `forward_dense_restricted`
are implemented on `MoeDenseLayer` from within `sparse.rs`, same crate.)

No graph, runtime, AMG, CUDA, tier-planner, adapter-toolkit, CLI,
generation, or loader changes.
