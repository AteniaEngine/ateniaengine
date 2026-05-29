# HANDOFF — MOE-12: multi-layer real MoE stack

Milestone: **MOE-12** (compose multiple assembled MoE layers into a
sequential stack and run them in order). Correctness-first, CPU-only,
experimental, opt-in. The MOE-2 fail-loud guard is **unchanged** — a real
MoE checkpoint still refuses to load as a model. No Mixtral / Qwen-MoE
end-to-end support is claimed. No CUDA, ROCm, Metal, tier-planner, CLI,
generation, Adapter Toolkit, batching, or optimisation. Predecessor:
MOE-0..11 (`dcd79ec`).

## What was added

`src/moe/stack.rs`:

```text
MoeStackConfig + per-layer (layer_id, MoeLayerConfig)
  + MoeWeightMap (MOE-9)  + byte resolver (MOE-10)
  → RealMoeLayer per layer   (MOE-11 assembly)
  → RealMoeStack { layers }
  → forward: x → layer0 → layer1 → ... → output
```

This is the first multi-layer MoE execution in Atenia: several real,
assembled MoE layers run back-to-back.

## 1. How the stack works

A `RealMoeStack { config: MoeStackConfig, layers: Vec<RealMoeLayer> }` runs
each layer's MOE-11 `forward` in sequence, feeding the output of layer `i`
straight into layer `i+1`:

```text
x → layers[0].forward → layers[1].forward → ... → layers[n-1].forward → out
```

`MoeStackConfig { num_layers }` is intentionally minimal. There are **no
residual connections, no norms, no attention, no embeddings** — this is a
*pure sequential composition of MoE layers*, the smallest possible
multi-layer step. It is **not** a transformer and **not** a full model.

Sequential composition is dimensionally valid because every MoE layer maps
`d_model → d_model` (SwiGLU experts preserve the model dimension), so no
projection is needed between layers.

## 2. How multiple layers are assembled

`RealMoeStack::assemble(map, stack_config, &[(layer_id, MoeLayerConfig)],
&resolve)` builds one `RealMoeLayer` per spec (in order) via the MOE-11
assembler, then calls `RealMoeStack::new`, which validates:

* the stack is non-empty (`EmptyStack`);
* `config.num_layers` equals the number of layers (`LayerCountMismatch`);
* every layer shares one `d_model` (`DModelMismatch`).

`RealMoeStack::new(config, Vec<RealMoeLayer>)` is also public, for callers
that assembled layers by other means. `forward` additionally checks the
input length equals `d_model` (`InputDimMismatch`).

## 3. How it is validated against manual execution

The tests assert the stack `forward` equals **manual layer-by-layer
chaining** within `1e-5`:

```rust
let h0 = stack.layers[0].forward(&x)?;
let h1 = stack.layers[1].forward(&h0)?;
let manual = stack.layers[2].forward(&h1)?;   // 3-layer case
assert!(stack.forward(&x) ≈ manual);           // within 1e-5
```

Covered for 2-layer and 3-layer Mixtral-style stacks, and a 2-layer
Qwen-MoE-style stack **with shared experts** in every layer. Since each layer
is itself certified against the MOE-4 sparse reference (MOE-11), the stack
adds no numerical drift over plain sequential composition.

## What is still missing for real Mixtral / Qwen-MoE

- **Transformer structure**: no residual connections, no RMSNorm/LayerNorm,
  no attention, no embeddings/lm_head, no positional handling — a real model
  interleaves attention + norms + residuals around each MoE layer. The stack
  composes MoE layers *only*.
- **Config parsing**: `num_layers` and per-layer configs are hand-built
  fixtures, not read from `config.json` (Adapter Toolkit work, deferred).
- **Multi-token / sequences**: single-vector forward only; no KV cache, no
  batching.
- **Shared-expert gating**: shared expert added ungated (some Qwen variants
  gate it by a sigmoid).
- **Fail-loud lift**: the loader still refuses real MoE checkpoints; the
  stack in isolation does **not** lift the guard.
- **Graph integration**: the MOE-5..8 graph ops still use the process-global
  registry; the stack does not populate it.

## What was NOT implemented

- No transformer, no residuals/norms/attention/embeddings, no full model.
- No Mixtral/Qwen-MoE end-to-end, no real download, no `config.json` parsing.
- No fail-loud lift, no Adapter Toolkit / loader load-path change, no graph
  wiring, no multi-token / batching, no shared-expert gating.
- No CUDA/ROCm/Metal, tier-planner, CLI, generation, optimisation.

## How this prepares MOE-13

MOE-13 can wrap each `RealMoeLayer` in the surrounding transformer-block
structure (RMSNorm + residual, and eventually attention), read `num_layers`
and per-layer configs from `config.json` (Adapter Toolkit boundary), and/or
populate the MOE-5..8 graph registry from a real stack — incrementally toward
a full MoE forward behind an explicit, validated opt-in that lifts fail-loud
only for the certified path.

## Tests

- `src/moe/stack.rs` — 9 unit tests (config validation, assembly, forward,
  2-/3-layer == manual chaining, d_model-consistency rejection, shared-expert
  stack, layer-count mismatch, wrong input dim).
- `tests/moe_real_stack_test.rs` — 8 integration tests on the production
  `SafetensorsReader`: `real_moe_stack_assembly`, `real_moe_stack_forward`,
  `two_layer_stack_matches_manual_execution`,
  `three_layer_stack_matches_manual_execution`,
  `stack_validates_d_model_consistency`, `stack_with_shared_experts`,
  `fail_loud_still_active`, `dense_models_still_load`.

Local validation: `cargo test --lib --release -- --test-threads=1` →
**699 passed / 0 failed / 1 ignored** (was 690). Integration suite: 8/8.

## Files modified

* `src/moe/stack.rs` — new (stack config + assembly + forward + 9 unit tests).
* `src/moe/mod.rs` — re-exports.
* `tests/moe_real_stack_test.rs` — new (8 integration tests).
* `docs/HANDOFF_MOE_12.md` — this file.

No loader load-path, Adapter Toolkit, CLI, generation, CUDA, ROCm, Metal,
tier-planner, or graph changes. Fail-loud preserved.
