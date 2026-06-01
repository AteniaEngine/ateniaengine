# HANDOFF — MOE-FULL-6: tiny full MoE transformer forward

Milestone: **MOE-FULL-6** (a whole tiny MoE transformer forward in the AMG
graph — embeddings → decoder layers with real attention + RoPE + causal mask +
the certified MoE block → final norm → lm_head → logits — validated against an
offline HuggingFace f64 reference). **Experimental, CPU-only, multi-token
prefill, test-only.** No decode loop, no generation, no KV cache, no batching,
no loader / Adapter Toolkit / CUDA / CLI changes, no fail-loud lift.
Predecessor: MOE-FULL-5 (`67d8e1c`).

## Scope built (Ruta A — honest, bounded)

A **real full forward to logits**, not a partial path:

```text
token ids [1, seq]
 → IndexSelect(embed_tokens)                         [1, seq, hidden]
 → ×2 decoder layers:
      RMSNorm·γ → Q/K/V → RoPE → scores·(1/√d) → causal mask → softmax
                → ·V → O proj → + residual
      RMSNorm·γ → MoeRealLayerReference (position-wise) → + residual
 → final RMSNorm·γ
 → lm_head matmul                                    [1, seq, vocab]
```

- **Multi-token prefill** (seq = 5), **real causal mask**, **real RoPE**
  (`gb.rope`, the same op certified against HF for the dense Llama/Qwen path),
  **2 real decoder layers**, the **MOE-FULL-4 certified MoE block** applied
  position-wise, final norm + lm_head.
- Built **only from existing AMG primitives** (`index_select`, `rms_norm`,
  `broadcast_mul`, `matmul_rhs_transposed`, `reshape`, `rope`, `permute`,
  `transpose_last_two`, `batch_matmul`, `broadcast_add`, `softmax`, `add`) +
  the MoE node. **No new graph op.**

### Honest simplifications (documented, not faked)

- **MHA, not GQA.** The fixture is generated with
  `num_key_value_heads == num_attention_heads`, so the model is a *real*
  Mixtral configured without GQA. (In the productive dense path GQA is resolved
  by a load-time K/V tile; reusing that is a larger refactor, out of scope.)
- The `1/√head_dim` attention score scale is absorbed by **pre-scaling `w_q`**
  in Rust (GraphBuilder has no `scale` op) — numerically exact.
- The MoE node was extended (MOE-FULL-6) to apply position-wise over a flat
  `[seq·d_model]` input (chunked per `d_model` row); the single-token path is
  bit-identical to MOE-FULL-4 (re-verified: 7/7).

## Fixture used

`fixtures/moe/full_mixtral.{safetensors,json}` — a **real tiny
`MixtralForCausalLM`** generated offline by
`fixtures/moe/generate_full_forward_reference.py`:
vocab 48, hidden 32, 2 layers, 4 heads (MHA, head_dim 8), 4 experts top-2,
classic `block_sparse_moe.w1/w3/w2` experts, RoPE θ=10000, rms_eps 1e-5,
untied lm_head. Safetensors ~245 KB, JSON ~6 KB (config + input_ids + HF f64
logits). No model downloaded, no large model committed.

## Numerical results (vs HuggingFace f64 reference)

Whole-sequence logits, Atenia f32 graph vs HF Mixtral f64 forward:

| Metric | Value |
|---|---|
| max_abs_diff | **7.451e-08** |
| mean_abs_diff | 1.340e-08 |
| rmse | 1.798e-08 |
| per-position argmax | **match (all 5 positions)** |

The whole MoE transformer forward matches HuggingFace to ~7e-8 (f32 vs f64) —
essentially f32 machine precision. This validates embeddings + real attention
(RoPE, causal mask, multi-token) + the certified MoE block + final norm +
lm_head composed end-to-end.

## What is NOT implemented

- No decode loop / generation, no KV cache, no incremental decoding.
- No GQA (MHA only), no sliding-window attention, no attention/softcap quirks.
- No batching, no multi-sequence, no optimisation.
- Test-only: no productive path builds this; the loader still fails loud on MoE
  and nothing is wired into the productive runtime/Adapter Toolkit/CLI.

## What remains for real generation

- **KV cache + decode loop** (incremental single-token steps reusing cached
  K/V) — the actual generation path.
- **GQA** (load-time K/V tile or a graph repeat-kv) for real Mixtral-8x7B.
- **Productive integration**: a Mixtral family adapter on the productive load
  path, lifting fail-loud behind an explicit opt-in, config-driven topology.
- **Large-model residency** (MOE-FULL-7): experts via the tiered WeightStore
  instead of materialising all in f32.

## Tests

- `src/moe/full_forward.rs` — 3 unit tests: `builds_and_logits_shape`,
  `is_deterministic`, `causal_mask_hides_future` (synthetic weights).
- `tests/moe_full_forward_test.rs` — 7 integration tests (real fixture):
  `tiny_full_forward_builds`, `tiny_full_forward_logits_shape`,
  `tiny_full_forward_is_deterministic`, `tiny_full_forward_matches_hf_reference`,
  `causal_mask_changes_future_visibility`, `fail_loud_still_active`,
  `dense_models_still_load`.

Local validation (real output, exit 0): full lib suite
`cargo test --lib --release -- --test-threads=1` → **753 passed / 0 failed /
1 ignored** (was 750; +3). Integration 7/7 (`matches_hf_reference`
max_abs_diff 7.451e-08). No regressions: `moe_real_layer_graph_op_test` 7/7,
`moe_decoder_layer_test` 6/6.

## Files modified

* `src/moe/full_forward.rs` — new (tiny full transformer graph + 3 unit tests).
* `src/moe/graph_op.rs` — `execute_real_moe_layer` extended to multi-token
  (position-wise); single-token path unchanged.
* `src/moe/mod.rs` — `pub mod full_forward;`.
* `tests/moe_full_forward_test.rs` — new (7 integration tests).
* `fixtures/moe/generate_full_forward_reference.py` — new (offline generator).
* `fixtures/moe/full_mixtral.{safetensors,json}` — new fixture (~251 KB total).
* `docs/HANDOFF_MOE_FULL_6.md` — this file.
* `docs/MOE_FULL_PATH_AUDIT.md` — progress note.

No generation, loader load-path, Adapter Toolkit, CUDA, ROCm, Metal,
tier-planner, or CLI changes. Fail-loud preserved.
