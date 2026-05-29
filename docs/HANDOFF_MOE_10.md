# HANDOFF — MOE-10: real expert tensor binding

Milestone: **MOE-10** (bind real checkpoint tensor *data* to executable
experts and run the sparse forward over real weights). Correctness-first,
CPU-only, experimental. The MOE-2 fail-loud guard is **unchanged** — a real
MoE checkpoint still refuses to load as a model. No Mixtral / Qwen-MoE
end-to-end support is claimed. No CUDA, ROCm, Metal, tier-planner, CLI,
generation, Adapter Toolkit, batching, or optimisation. Predecessor:
MOE-0..9 (`7aaba77`).

## What was connected

MOE-9 produced `MoeWeightMap`: structured *metadata* (names, shapes, roles)
with **no tensor data**. MOE-10 adds the missing bridge in
`src/moe/binding.rs`:

```text
MoeWeightMap (metadata)  +  byte resolver (name → Vec<f32>)
  → RealExpertTensorBinding::resolve  → real MoeDenseExpert  (MOE-3)
  → build_real_layer                  → real MoeDenseLayer
  → forward_sparse                    → sparse forward over REAL weights (MOE-4)
```

This is the first time real checkpoint bytes flow into the executable MoE
path: real → real → real.

## 1. What real tensors are loaded

For a given layer of a real (small, synthetic-named) checkpoint:

- **Router**: `block_sparse_moe.gate.weight` / `mlp.gate.weight`, shape
  `[num_experts, d_model]` → `MoeDenseLayer.w_router`.
- **Per expert** (gate/up/down): Mixtral `w1/w3/w2`, Qwen-MoE
  `gate_proj/up_proj/down_proj`. gate/up are `[d_ff, d_model]`, down is
  `[d_model, d_ff]` — exactly the `MoeDenseExpert` row-major convention, so
  **no transpose** is required.

Bytes are decoded via the production `SafetensorsReader`
(`get(name).to_vec_f32()`, supporting F32/BF16/F16) and length-checked
against the declared shape.

## 2. How experts are constructed

`RealExpertTensorBinding::resolve(layer_id, expert_id, &ExpertTensors,
&resolve)`:

1. Requires all three projections present in metadata (else
   `IncompleteExpert`).
2. Infers `(d_model, d_ff)` from the gate shape and cross-checks up (same
   shape) and down (`[d_model, d_ff]`) — `ShapeInconsistency` on disagreement.
3. Resolves each tensor's bytes via the resolver; `UnresolvedTensor` if a
   listed tensor has no data, `DataLengthMismatch` if the byte count is wrong.
4. Builds + validates a `MoeDenseExpert`.

`build_real_layer(map, layer_id, conceptual_top_k, &resolve)` does this for
every expert (ascending id order, `BTreeMap`-backed), resolves the router
(`[num_experts, d_model]`, rows must equal the expert count), and assembles a
validated `MoeDenseLayer`.

## 3. How the real forward runs

`layer.forward_sparse(input, k)` (MOE-4) over the real-bound layer: route →
softmax → top-k → renormalise → execute only the selected experts → combine.
The integration tests assert the real-weight sparse output equals the
dense-restricted oracle (`forward_dense_restricted`) within `1e-5`.

## Decoupling — why no loader dependency

The binding never references the loader. It takes a **byte resolver** closure
`Fn(&str) -> Option<Vec<f32>>`, so `src/moe/` stays loader-free. A caller
wires a reader in one line:
`|name| reader.get(name).and_then(|e| e.to_vec_f32().ok())`. This keeps the
MOE-2 fail-loud guard and the productive load path entirely untouched: the
binding is an isolated path a caller drives explicitly, not a model loader.

## What is still missing for real Mixtral / Qwen-MoE

- **Full model load**: only a single MoE layer is bound from an explicit
  reader; there is no multi-layer / multi-token transformer, no attention,
  no norms, no embedding/lm_head, no KV cache.
- **Config parsing**: `num_experts` / `experts_per_token` / shared-expert
  config from `config.json` is not read (Adapter Toolkit work — deferred);
  `conceptual_top_k` is passed in by the caller.
- **Shared experts**: Qwen-MoE/DeepSeek shared-expert tensors are mapped by
  MOE-9 but the binding builds only the routed experts + router (the shared
  branch is not yet combined into the forward).
- **Fail-loud lift**: the loader still refuses real MoE checkpoints; binding
  in isolation does **not** lift the guard. A future milestone must lift it
  behind an explicit opt-in once a full MoE forward is wired.
- **Graph integration**: the MOE-5..8 graph ops still use the process-global
  layer registry; binding does not yet populate that registry from real
  data.

## What was NOT implemented

- No full model execution, no Mixtral/Qwen-MoE end-to-end, no real download.
- No fail-loud lift, no Adapter Toolkit / loader load-path change.
- No graph / runtime wiring, no config parsing, no shared-expert forward.
- No CUDA/ROCm/Metal, tier-planner, CLI, generation, batching, optimisation.

## How this prepares MOE-11

MOE-11 can: (a) bind **all** MoE layers of a small real checkpoint, (b) read
expert/top-k config from `config.json`, (c) combine the shared expert, and/or
(d) populate the MOE-5..8 graph registry from real bindings — incrementally
working toward a full MoE forward behind an explicit, validated opt-in that
lifts fail-loud only for the certified path.

## Tests

- `src/moe/binding.rs` — 7 unit tests (single-expert resolve, layer build +
  sparse forward vs oracle, missing-data, incomplete metadata, length
  mismatch, missing layer, identical-experts reproduces single expert).
- `tests/moe_real_binding_test.rs` — 7 integration tests on the production
  `SafetensorsReader`: `real_expert_tensor_resolution`,
  `real_expert_construction`, `real_expert_forward`,
  `sparse_forward_with_real_experts`, `expert_registry_resolves_real_tensors`,
  `fail_loud_still_active`, `dense_models_still_load`.

Local validation: `cargo test --lib --release -- --test-threads=1` →
**683 passed / 0 failed / 1 ignored** (was 676). Integration suite: 7/7.

## Files modified

* `src/moe/binding.rs` — new (binding + 7 unit tests).
* `src/moe/mod.rs` — re-exports.
* `tests/moe_real_binding_test.rs` — new (7 integration tests).
* `docs/HANDOFF_MOE_10.md` — this file.

No loader load-path, Adapter Toolkit, CLI, generation, CUDA, ROCm, Metal,
tier-planner, or graph changes. Fail-loud preserved.
