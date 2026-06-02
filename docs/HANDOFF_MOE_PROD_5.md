# HANDOFF — MOE-PROD-5: resident layer reconstruction (scope C)

Milestone: **MOE-PROD-5** — on a warm load, rebuild the MoE backend **directly
from the persistent tier**, without re-reading the shards or re-assembling
experts. MOE-PROD-4 (scope A) skipped the tier *write* but still read shards +
assembled experts on warm. User-approved **scope C**: persist + reconstruct the
**whole backend** (experts + attention + embed + lm_head + router + gate).
Predecessor: `20d2d55`.

## FASE 1 — Audit

The warm load (scope A) still, per layer: `RealMoeLayer::assemble` (reads experts
from shards + f32 decode), a RAM validation copy + self-validate, then tiers
(reusing files). And `tensor_metas()` opens all 8 shards (~28.6 GB) + `build_graph`
reads attention/embed/lm_head from shards. **Material finding (reported, scope
decision):** skipping only expert assembly does **not** avoid the shard I/O —
attention + metadata still need the shards. To truly avoid shards, the whole
backend must be persisted and reconstructed. The user chose scope C.

## FASE 2-4 — Design + reconstruction

- `disk_tier.rs`: `read_f32_named(path, numel)` (read a tier file back, size-
  validated).
- `residency.rs`: `from_real_layer_at` now also persists **router + shared-gate**;
  `ResidentExpertLayer::from_tier` / `ResidentExpert::from_tier` rebuild a
  disk-tier layer from existing tier files (experts wrapped as disk handles,
  router/gate read from tier) — **no `RealMoeLayer`, no shards**.
- `runtime.rs`:
  - `MoeWeightSource::identity_signature()` — a cheap model id input (sharded:
    index `weight_map` + base dir + total_size; single-file: names+shapes) that
    needs **no shard data**, so the warm decision itself doesn't open shards.
  - `persist_backend` — cold writes embed / final_norm / lm_head / per-layer
    attention (+biases) to the tier (experts/router/gate already persisted).
  - `try_warm_reconstruct` — validates the manifest, then rebuilds the entire
    `TinyMixtralWeights` + residency from the tier and returns the runtime,
    **without `tensor_metas`, `assemble`, `build_graph`, or any shard read**.
  - Manifest **v2**: adds `convention`, `has_shared`, `attn_has_bias` and records
    every tensor (experts + backend). Version bump invalidates v1 tiers safely.
  - load_core: after config parse, if a valid tier exists → `try_warm_reconstruct`
    returns early; otherwise the cold path runs and persists everything.

## FASE 3/5 — Manifest validation + fallback (safety)

`try_warm_reconstruct` returns `None` (→ caller falls back to the **certified
shard path**, never corrupting the load) on **any** doubt:
- missing / unparseable `tier_manifest.json`;
- `version` ≠ 2, `model_id` mismatch, or layer/expert/top-k mismatch;
- any recorded tier file missing or with a wrong byte length;
- an unknown convention / non-graph family.

Default off (`ATENIA_MOE_TIER_PERSIST` unset) → unchanged ephemeral tier.

## FASE 6-7 — Bit-exactness + tests

- `tests/moe_tier_reconstruct_test.rs` (new) — **the scope-C proof**: build a
  2-shard checkpoint, cold-load (persist), **delete the shard files**, warm-load
  → reconstructs the whole backend from the tier alone and generates
  **bit-identically**. Proves "without reading shards".
- `tests/moe_tier_persist_test.rs` — still green; the second load now takes the
  warm reconstruction path (mtimes unchanged, identical generation, regenerates
  on file loss → fallback).
- Full MoE regression (residency-tier, expert-cache, mixtral, qwen, deepseek,
  sharded, production, robustness, cli) + lib suite — green. Default path
  unchanged.

## FASE 8 — Benchmark (real Qwen1.5-MoE-A2.7B, max-new 2)

| | MOE-PROD-4 warm (scope A) | MOE-PROD-5 cold | MOE-PROD-5 warm (scope C) |
|---|---|---|---|
| Total wall (load + 2 tokens) | 2445 s | 3757 s | **701 s** |
| Shards read | yes (28.6 GB) | yes | **no (tier only — "reconstructing 24 layers")** |
| Expert assembly | yes | yes | **no** |
| Output (token ids) | `374, 6188` | `374, 6188` | `374, 6188` — **identical (bit-exact)** |
| Tier on disk | ~50 GiB | ~53.3 GiB (4659 tensors, +backend) | reused |

**Warm scope C: 701 s vs scope-A's 2445 s → ~1744 s (~29 min, ~71 %) faster**,
bit-identical, with **no shard read and no expert assembly** (the whole backend
is rebuilt from the tier: "tier warm: manifest valid, reconstructing 24 layers").

## Problem found + fix (caught by the real benchmark, fixed in-milestone)

The first real warm load **fell back to the cold path** ("4659 reused, 0
written"). Diagnostics (gated by `ATENIA_MOE_CACHE_STATS=1`) pinpointed it:
`from_tier layer 0: L0.shared.gate.bin: 46137344 bytes, expected 11534336`.
**Root cause:** Qwen-MoE's **shared expert** uses
`shared_expert_intermediate_size` (5632) which **differs from the routed
`moe_intermediate_size`** (1408); `from_tier` sized the shared expert with the
routed d_ff. **Fix:** recover the shared expert's FFN width from its tier file
size (`numel / hidden`) and size it accordingly. The tiny **Mixtral** fixture has
no shared expert, so it could not catch this — added a tiny **Qwen** (shared
expert, different d_ff) reconstruction test that reproduces + locks the fix.
(Also surfaced a Windows build gotcha: a running `atenia.exe` locks the binary so
`cargo build` silently kept a stale exe — rebuilt clean.)

## FASE 9 — Review

- **Saving:** warm load+gen **2445 s → 701 s (~71 %, ~29 min)** on the real
  14.3 B MoE; the warm load itself (reconstruction, ~reading ~9 GB of backend
  f32 + wrapping expert disk handles) is now a few minutes vs the ~30+ min
  shard-read+assemble it replaced.
- **What still costs:** the **per-token disk-tier generation** (the MOE-PROD-3
  domain: top-k expert NVMe reads per token, no cache benefit) and reading the
  ~9 GB of backend f32 from the tier.
- **New bottleneck:** generation (disk-tier per-token expert reads) — load is no
  longer the dominant cost on a warm run. Further wins would target the
  generation path (e.g. bf16 tier to halve read volume), not the loader.

## Files modified

- `src/tensor/disk_tier.rs` — `read_f32_named`.
- `src/moe/residency.rs` — persist router/gate, `from_tier` reconstruction.
- `src/moe/runtime.rs` — `identity_signature`, `try_warm_reconstruct`,
  `persist_backend`, manifest v2, warm fast path + fallback.
- `tests/moe_tier_reconstruct_test.rs` — new.
- `docs/HANDOFF_MOE_PROD_5.md` (this) + `docs/STATUS.md`.

No new architecture/family/math/graph ops; outputs bit-exact; default path
unchanged.

## Deliverable answers (filled after benchmark)

1. **What implemented:** warm reconstruction of the whole MoE backend from the
   persistent tier, with a safe fallback to the certified shard path.
2. **How it reconstructs layers:** `try_warm_reconstruct` reads the manifest +
   tier files: experts → disk handles, router/gate/attention/embed/lm_head →
   f32 from tier; builds `TinyMixtralWeights` + residency; no shards.
3. **How the manifest is validated:** model_id + version + counts + every file
   present with the exact byte length; any failure → fallback.
4. **Fallbacks:** missing/invalid manifest, model_id/version mismatch, missing
   file, size mismatch, unknown convention/family → certified shard path.
5. **Tests:** reconstruct-without-shards (new) + persist/reuse/regenerate + full
   MoE regression + lib suite.
6. **Time before:** 2445 s (scope-A warm, load + 2 tokens).
7. **Time after:** 701 s (scope-C warm).
8. **Real saving:** ~1744 s (~29 min, **~71 %**), bit-identical output.
9. **New bottleneck:** per-token disk-tier generation (MOE-PROD-3 domain) +
   reading ~9 GB backend f32 from tier; the load is no longer dominant.
10. **Commit:** see git log.
11. **CI:** see push.
