# HANDOFF — MOE-FULL-5: decoder layer integration path

Milestone: **MOE-FULL-5** (compose one decoder layer — norm + self-attention +
residual + MoE node + residual — in the AMG graph, proving the MOE-FULL-4
`MoeRealLayerReference` node integrates with the surrounding dense pieces).
**Experimental, CPU-only, single token, single decoder layer.** No full model,
no generation, no KV cache, no multi-token, no batching, no loader / Adapter
Toolkit / CUDA / CLI changes, no fail-loud lift. Predecessor: MOE-FULL-4
(`6c4307c`).

## Architecture

`src/moe/decoder_layer.rs` assembles, in the AMG graph:

```text
x  [1, d_model]
 → RMSNorm·γ₁ → Q/K/V proj → scores(q·kᵀ) → softmax → ctx(·V) → O proj → + x   (residual 1)
 → RMSNorm·γ₂ → MoeRealLayerReference (MOE-FULL-4)                       → + r1  (residual 2)
 → output [1, d_model]
```

Built **only from existing AMG primitives** (`rms_norm`, `broadcast_mul`,
`matmul`, `matmul_rhs_transposed`, `softmax`, `reshape`, `add`) plus the
MOE-FULL-4 `moe_real_layer_reference` node. **No new graph op was added.**

### Deliberate minimalism (and why)

- **Single token (`seq = 1`), single-head, no RoPE, no GQA, no KV cache.**
  With one token, `softmax` over a one-element score row is always `1.0`, so
  the `1/sqrt(d)` score scale does not affect the output and is omitted
  (GraphBuilder has no `scale` op). The attention sub-block still runs
  structurally (Q/K/V/O projections, scores, softmax, weighted-V), so this
  validates **graph composition** of attention + residual + MoE — not
  multi-token attention dynamics (that is MOE-FULL-6).
- It does **not** reuse the productive `build_transformer_block_llama`: that
  function is private and coupled to the productive `LlamaRuntime` / causal
  mask / loader-registered parameters. Reusing it would require refactoring the
  dense path — out of scope. The experimental layer instead uses parameter
  nodes holding fixed tensors (the "constant" mechanism) and is fully isolated
  in `src/moe/`.

## Fixture used

The committed real Mixtral layer-0 fixture
`fixtures/moe/mixtral_layer0.{safetensors,json}` (~385 KB, packed experts, 4
experts, d_model 64, no shared expert) supplies the **real** MoE sub-block
(`RealMoeLayer::assemble`). The attention weights are deterministic synthetic
matrices (the milestone validates *composition*, not attention numerics). No
model downloaded, no model committed.

## Numerical results

The graph decoder layer is validated against an independent imperative
reference (`decoder_layer_reference`, f64-accumulating, same operation incl.
`RealMoeLayer::forward_auto` for the MoE sub-block):

- Unit tests (synthetic d_model=8 MoE fixture): graph vs reference
  `max_abs_diff < 1e-5` ✅.
- Integration test (real Mixtral layer-0, d_model=64): graph vs reference
  `max_abs_diff < 1e-5` ✅.
- The MoE node inside the layer still equals `RealMoeLayer::forward_auto`
  (`< 1e-5`) — MOE-FULL-4's contract holds within the composed layer.
- `residual_changes_output`: the layer output differs from a bare MoE forward
  on `x` (attention + residuals contribute), proving real composition rather
  than pass-through.

## Limitations

- One decoder layer, one token. No multi-token attention, no RoPE, no GQA, no
  KV cache, no causal mask (single token needs none).
- Attention weights are synthetic; no HF single-layer logit comparison yet
  (that needs the full multi-token attention path of MOE-FULL-6).
- Experimental / test-only: no productive path builds this layer; fail-loud
  intact.

## What is still missing for a full model

- **Multi-token attention** with RoPE + GQA + causal mask + KV cache
  (MOE-FULL-6).
- **Stacking** N layers + embeddings + final norm + lm_head, then the
  generation loop, with logits compared to HF.
- **Memory/residency** for large MoE (MOE-FULL-7).
- Productive loader wiring + the fail-loud lift behind an explicit opt-in.

## Tests

- `src/moe/decoder_layer.rs` — 4 unit tests: `weights_validate`,
  `graph_matches_reference`, `graph_is_deterministic`, `residual_changes_output`.
- `tests/moe_decoder_layer_test.rs` — 6 integration tests (real Mixtral
  fixture): `decoder_layer_matches_reference`, `decoder_layer_is_deterministic`,
  `decoder_layer_rejects_bad_dims`, `moe_graph_node_still_matches_reference`,
  `fail_loud_still_active`, `dense_models_still_load`.

Local validation (real output, exit 0): `cargo test --lib --release --
--test-threads=1` → **750 passed / 0 failed / 1 ignored** (was 746; +4).
Integration suite 6/6. Prior MoE graph suites still green
(`moe_real_layer_graph_op_test` 7/7, `moe_graph_op_test` 6/6,
`moe_conditional_expert_test` 9/9).

## Files modified

* `src/moe/decoder_layer.rs` — new (experimental decoder layer + reference + 4 tests).
* `src/moe/mod.rs` — `pub mod decoder_layer;` + re-exports.
* `tests/moe_decoder_layer_test.rs` — new (6 integration tests).
* `docs/HANDOFF_MOE_FULL_5.md` — this file.
* `docs/MOE_FULL_PATH_AUDIT.md` — progress note.

No generation, loader load-path, Adapter Toolkit, CUDA, ROCm, Metal,
tier-planner, or CLI changes. Fail-loud preserved.
