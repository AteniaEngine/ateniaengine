# HANDOFF — MOE-FULL-9: GQA + expert cache + controlled productive preparation

Milestone: **MOE-FULL-9** — the first **controlled** productive preparation for
MoE. It closes the last *correctness* gap with real Mixtral (GQA), makes the
residency path practical (expert cache), and prepares the productive loader to
recognise + validate MoE checkpoints and **report precisely why it still
refuses them** — without lifting fail-loud or declaring support.
**Experimental, CPU(+NVMe), test/opt-in only.** No CLI / runtime / Adapter
Toolkit / pipeline change, no generation activation, no Mixtral/Qwen-MoE support
declared, **fail-loud intact**. Predecessor: MOE-FULL-8 (`4fb1cf5`).

## 1. GQA (grouped-query attention)

Real Mixtral uses GQA (`num_key_value_heads < num_attention_heads`). The
productive dense path resolves GQA "by a load-time K/V tile"; `src/moe/gqa.rs`
does the same for the MoE path:

- `tile_kv_weight(w_kv, n_kv, n_heads, head_dim, d_model)` tiles the K/V
  projection weights from `[n_kv*head_dim, d_model]` to `[n_heads*head_dim,
  d_model]`, replicating HF `repeat_kv` (query head `h` ← K/V head
  `h / kv_groups`). `to_mha_kv` is the identity when `n_kv == n_heads`.
- Because the weights are pre-tiled to MHA shape, the **certified MOE-FULL-6/7
  attention graph is reused unchanged** — no new graph op, no attention-topology
  change.
- **Validated** on a real tiny GQA Mixtral (`n_heads=4`, `n_kv=2`, kv_groups=2;
  `fixtures/moe/gqa_mixtral.*`) against an offline HF f64 reference:
  **max_abs_diff 5.960e-08**, per-position argmax matches. So `num_heads !=
  num_kv_heads` is now handled to f32 machine precision.

## 2. Expert cache (LRU + prefetch + reuse)

`src/moe/residency.rs::ExpertCache` sits over the MOE-FULL-8 tiers so the NVMe
tier is not re-read on every token:

- **LRU** bounded by `capacity` (0 = caching disabled); evicts least-recently-
  used experts.
- **prefetch(indices)** warms the cache (hot-set / profiled experts) — one tier
  read each, then resident.
- **reuse**: `forward_cached(&self, cache, x)` resolves routed experts through
  the cache; a hit skips the tier read entirely. `CacheStats` exposes
  `hits / misses / evictions / prefetched / tier_bytes_read`.
- **Bit-identical** to the uncached `forward`. Evidence: the same token routed
  twice reads the tier only once (2 misses, then 2 hits); a fully-prefetched
  cache serves subsequent forwards with **0 misses**; a capacity-1 cache evicts
  under churn; capacity-0 disables caching.

## 3. Family recognition (metadata + wiring, no activation)

`src/moe/family.rs` gives the loader a family identity:

- `MoeFamily { Mixtral, QwenMoe }` + `classify_family(names)` (block_sparse →
  Mixtral; shared_expert / mlp.router / per-expert gate_proj → Qwen-MoE; packed
  → Mixtral). `MoeFamilyDescriptor` carries router naming, expert layout,
  shared-expert presence, top-k renormalisation.
- `validate_family_config(names, &MoeConfig)` cross-checks the declared config
  (MOE-FULL-2) against the tensors: expert count, experts-per-token bound,
  shared-expert agreement — returns notes, never executes.
- This is **wiring/metadata only**. No productive Adapter-Toolkit registry entry
  is added and **no load is activated** — recognising a family does not enable
  loading it.

## 4. Loader preparation (still fails loud, precisely)

The loader's MoE guard (`weight_mapper.rs`, both shard paths) now builds its
message via `family::moe_failloud_report` and **still returns
`LoaderError::MoeUnsupported`**:

```text
MoE detected
Family: Mixtral (experts=8, router_tensors=1, expert_tensors=4, shared_expert_tensors=0)
Productive support not enabled (the experimental MoE path is opt-in / test-only; …)
```

instead of the old generic "MoE unsupported". The fail-loud guard is
**unchanged** — MoE never loads here; dense models are entirely unaffected (the
guard does not fire). Verified end-to-end through the real `load_into` path.

## What is NOT done (out of scope / MOE-FULL-10)

- **No fail-loud lift, no activation.** Real MoE checkpoints still refuse to
  load. No generation-productive path, no CLI change, no Mixtral/Qwen-MoE
  support declared.
- No productive Adapter-Toolkit registry entry (recognition lives in
  `src/moe/family.rs`, not `src/adapter_toolkit/`); wiring the residency+cache
  block into the productive runtime is MOE-FULL-10.
- No VRAM expert tier, no real-checkpoint full-model certification, no
  config-driven topology on the productive path.

## Tests

- `src/moe/gqa.rs` — 5 unit (`kv_groups_math`, `tile_replicates_kv_heads_in_hf_order`,
  `mha_tiling_is_identity`, `rejects_bad_shape`, `mixtral_8x7b_shape_tiling`).
- `src/moe/residency.rs` — +5 cache unit (`cached_forward_matches_uncached`,
  `cache_reuse_avoids_tier_reads`, `prefetch_warms_cache_to_zero_misses`,
  `lru_evicts_under_budget`, `capacity_zero_disables_caching`).
- `src/moe/family.rs` — 8 unit (classify Mixtral/Qwen/dense, descriptors,
  config validation ×3, failloud report).
- `tests/moe_gqa_test.rs` — 3 integration (real GQA fixture vs HF, 5.960e-08).
- `tests/moe_family_loader_test.rs` — 4 integration (end-to-end loader fail-loud
  with family message for Mixtral + Qwen-MoE, dense unaffected, real fixture
  classification).

Local validation (real output, exit 0): full lib suite
`cargo test --lib --release -- --test-threads=1` → **780 passed / 0 failed /
1 ignored** (was 762; +18). Integration: GQA 3/3, family-loader 4/4. No
regressions: residency 4/4, decode 5/5, full_forward 7/7, **moe_loader_failloud
3/3** (the legacy `unsupported_message` helper is untouched).

## Files modified

* `src/moe/gqa.rs` — new (K/V tiling for GQA + 5 unit tests).
* `src/moe/residency.rs` — `ExpertCache` (LRU/prefetch/reuse), `CacheStats`,
  `forward_cached`, `prefetch`, `resolve_cached` + 5 cache unit tests.
* `src/moe/family.rs` — new (family classification, config validation,
  fail-loud report + 8 unit tests).
* `src/moe/mod.rs` — `pub mod gqa; pub mod family;` + family re-exports.
* `src/v17/loader/weight_mapper.rs` — both MoE guards now use the family-aware
  `moe_failloud_report` (still returns `MoeUnsupported`; fail-loud unchanged).
* `tests/moe_gqa_test.rs`, `tests/moe_family_loader_test.rs` — new integration.
* `fixtures/moe/generate_gqa_reference.py`, `fixtures/moe/gqa_mixtral.{safetensors,json}`
  — new GQA fixture (~237 KB).
* `docs/HANDOFF_MOE_FULL_9.md` — this file.
* `docs/MOE_FULL_PATH_AUDIT.md`, `docs/MOE_OVERVIEW.md` — MOE-FULL-9 marked DONE;
  remaining activation work renamed MOE-FULL-10.

The only productive-code touch is the loader's fail-loud **message** (still
`MoeUnsupported`). No fail-loud lift, no runtime / Adapter Toolkit / CLI / CUDA
change, no MoE support declared.
