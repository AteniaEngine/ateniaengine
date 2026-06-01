# HANDOFF — MOE-FULL-7: MoE generation (prefill + KV cache + decode)

Milestone: **MOE-FULL-7** (the first end-to-end *generation* path for the
experimental tiny MoE transformer — multi-token prefill seeding a per-layer KV
cache, then an incremental greedy decode loop reusing that cache — validated
both against the MOE-FULL-6 full-recompute oracle and an offline HuggingFace
f64 greedy reference). **Experimental, CPU-only, greedy-only, test-only.**
No productive loader / runtime / Adapter Toolkit / CLI / WeightStore / CUDA
change, no fail-loud lift. Predecessor: MOE-FULL-6 (`ebd1ffe`).

## Scope built (honest, bounded)

A real **prefill → KV cache → incremental decode → multi-token** loop:

```text
prompt ──► prefill (seq = prompt_len, cached_len = 0)
           │  harvest per-layer post-RoPE-permute K, V  ──► KV cache (seed)
           ▼
greedy argmax ─► token₀
           │
decode step s (seq = 1, cached_len = prompt_len + s), per layer:
   RMSNorm·γ → Q/K/V → RoPE(offset = cached_len)
             → concat(cache_K, K_new) / concat(cache_V, V_new)   (axis 2)
             → scores = Q·Kᵀ_full  (1 query vs cached_len+1 keys, NO mask)
             → softmax → ·V_full → O → + residual
   RMSNorm·γ → MoE block (MOE-FULL-4 node, 1 row) → + residual
   → final RMSNorm·γ → lm_head → logits[1,1,vocab]
greedy argmax ─► tokenₛ₊₁ ;  harvest K_full/V_full ─► next step's cache
```

- **Prefill** reuses the MOE-FULL-6 attention wiring and additionally exposes
  the per-layer post-RoPE-permute K and V nodes (`[1, n_heads, seq, head_dim]`)
  to seed the cache.
- **Decode** runs at `seq = 1`. The cache enters as two parameter slots per
  layer, patched each step via `Graph::overwrite_parameter`; the post-concat
  `K_full`/`V_full` (length `cached_len + 1`) are harvested for the next step.
  RoPE uses `rope_with_offset(cached_len)` so the new token rotates at its
  absolute position. **No causal mask is needed** — the single new query is the
  last position and legitimately attends to every cached key.
- Built **only from existing AMG primitives** (`rope_with_offset`, `concat`,
  `batch_matmul`, `transpose_last_two`, `softmax`, `permute`, `index_select`,
  `rms_norm`, `broadcast_mul`, `matmul_rhs_transposed`, `add`) + the MOE-FULL-4
  MoE node. **No new graph op.**
- The KV-cache state machine (harvest node outputs → `overwrite_parameter`) is
  exactly the pattern the productive dense generator uses, but reimplemented
  entirely inside the experimental MoE module — nothing productive is touched.

### Honest simplifications (documented, not faked)

- **Greedy only** (argmax). No sampling / temperature / top-k / top-p.
- **MHA, no GQA** — same tiny Mixtral fixture as MOE-FULL-6
  (`n_kv == n_heads`). `1/√head_dim` absorbed by pre-scaling `w_q`.
- **Per-step graph rebuild** (same policy as the dense `generate_greedy`): each
  decode step builds a fresh `seq=1` graph at the current `cached_len`. Fine for
  a tiny experimental model; production decode-graph reuse is later work.
- Cache lives in plain `Tensor`s (f32), not the tiered `WeightStore`.

## Fixtures used

- **Weights**: the already-committed MOE-FULL-6 tiny Mixtral
  `fixtures/moe/full_mixtral.safetensors` (real `MixtralForCausalLM`; vocab 48,
  hidden 32, 2 layers, 4 heads MHA, 4 experts top-2). Not re-committed.
- **Greedy reference**: `fixtures/moe/full_mixtral_gen.json` (new, ~4 KB),
  produced by `fixtures/moe/generate_decode_reference.py`. The script **loads
  the committed weights** into a real HF Mixtral, goes f64, and computes greedy
  decoding by **full recompute every step** (the KV-cache oracle): prompt
  `[22, 25, 29]`, 4 new tokens, recording each step's f64 vocab logit row. No
  model downloaded, no large model committed.

## Numerical results

Two independent locks.

**(1) KV-cache correctness (no HF needed)** — prefill + incremental decode must
equal recomputing the full prefix every step (the MOE-FULL-6
`build_tiny_mixtral_graph` oracle). Unit test `kv_cache_matches_full_recompute`
passes within 1e-4 (f32). This is the R2 falsifier for the cache + RoPE-offset
logic.

**(2) HuggingFace f64 greedy parity** — Atenia's prefill+decode loop vs the
offline HF greedy reference:

| Metric | Value |
|---|---|
| generated ids | **exact match** (`[17, 20, 10, 17]`) |
| per-step logits max_abs_diff | **4.470e-08** |
| per-step logits mean_abs_diff | 1.139e-08 |
| per-step argmax | **match (all 4 steps)** |

~4e-8 is f32 machine precision vs the f64 reference. The whole generation path
(prefill, KV cache, RoPE position offset, decode attention over cached keys,
MoE block, lm_head) reproduces HuggingFace greedy decoding exactly.

## What is NOT implemented

- No sampling (greedy only); no GQA (MHA only); no sliding-window.
- No batching, no multi-sequence, no decode-graph reuse, no perf work.
- Cache is f32 in-RAM, not routed through the tiered `WeightStore`.
- Test-only: no productive path builds this; the loader still fails loud on MoE;
  nothing is wired into the productive runtime / Adapter Toolkit / CLI / CUDA.

## What remains (MOE-FULL-8)

- **GQA** (load-time K/V tile or a graph repeat-kv) for real Mixtral-8x7B.
- **Large-model residency**: experts via the tiered `WeightStore` instead of
  materialising all in f32.
- **Productive integration**: a Mixtral family adapter on the productive load
  path, lifting fail-loud behind an explicit opt-in, config-driven topology,
  and folding this loop into the productive generator.
- Sampling, decode-graph reuse, longer-context validation.

## Tests

- `src/moe/generate.rs` — 3 unit tests (synthetic weights):
  `generation_is_deterministic`, `kv_cache_matches_full_recompute`,
  `argmax_tokens_match_full_recompute_greedy`.
- `tests/moe_decode_generation_test.rs` — 5 integration tests (real fixture):
  `decode_generation_runs_and_shapes`, `decode_generation_is_deterministic`,
  `decode_generated_ids_match_hf_greedy`, `decode_step_logits_match_hf_reference`
  (max_abs_diff 4.470e-08), `fail_loud_still_active`.

Local validation (real output, exit 0): full lib suite
`cargo test --lib --release -- --test-threads=1` → **756 passed / 0 failed /
1 ignored** (was 753; +3). Integration: decode 5/5, plus no regressions —
`moe_full_forward_test` 7/7, `moe_decoder_layer_test` 6/6,
`moe_real_layer_graph_op_test` 7/7.

## Files modified

* `src/moe/generate.rs` — new (prefill-with-harvest + decode graph + greedy
  loop + 3 unit tests).
* `src/moe/mod.rs` — `pub mod generate;`.
* `tests/moe_decode_generation_test.rs` — new (5 integration tests).
* `fixtures/moe/generate_decode_reference.py` — new (offline greedy generator).
* `fixtures/moe/full_mixtral_gen.json` — new greedy reference (~4 KB; reuses the
  committed `full_mixtral.safetensors` weights).
* `docs/HANDOFF_MOE_FULL_7.md` — this file.
* `docs/MOE_FULL_PATH_AUDIT.md` — MOE-FULL-7 marked DONE; remaining work renamed
  MOE-FULL-8.

No generation-productive, loader load-path, Adapter Toolkit, WeightStore, CUDA,
ROCm, Metal, tier-planner, or CLI changes. Fail-loud preserved.
