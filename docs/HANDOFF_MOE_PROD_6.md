# HANDOFF — MOE-PROD-6: BF16 tier + shared-expert cache

Milestone: **MOE-PROD-6** — reduce the **per-token generation I/O** that
MOE-PROD-5 left as the bottleneck (warm load is no longer dominant), **without
touching the MoE math, routing, or outputs** (bit-exactness mandatory, safe
fallback mandatory). Two levers: (a) persist experts in **bf16** (half the NVMe
read volume), keeping f32 at execution; (b) a bounded **shared-expert cache**
so the shared expert (read every token) is materialised once per layer.
Predecessor: `db59803`.

## FASE 1 — Audit (what still costs time)

Per MOE-PROD-5 the warm load is fast (reconstructs from the tier, no shards).
The remaining cost is the **per-token disk-tier generation**:

- **Routed experts** (Qwen1.5-MoE: hidden 2048, `moe_intermediate_size` 1408,
  60 experts, top-4, 24 layers): each expert gate+up+down ≈ 8.65 M elems × 4 B =
  **34.6 MB**; top-4 × 24 layers = 96 reads/token ≈ **3.3 GB/token** (the
  measured run read ~19.2 GiB total; the MOE-PROD-3 LRU cache recovers ~22 %).
- **Shared expert** (`shared_expert_intermediate_size` 5632 = **4× routed**):
  gate+up+down ≈ 34.6 M elems × 4 B = **138 MB per layer × 24 = 3.3 GB/token,
  every token, not cached** — one shared expert costs as much I/O as the whole
  top-4 routed set. This is the single highest-leverage target.

**Risks:** bf16 truncation is lossless **only** when a value's low 16 mantissa
bits are zero (true for any f32 decoded from a bf16 source via `(bits)<<16`;
**not** true for arbitrary f32). So bf16 must be **auto-detected per tensor** and
fall back to f32 when not representable, or bit-exactness breaks. The read path
(`ensure_cpu` on a `Disk` handle) already dispatches on dtype and upcasts
bf16→f32 with the same `bf16_decode_bulk` kernel — so the round-trip is provably
the identity for bf16-source weights.

## FASE 2 — Design

- **TierStorageFormat** = the existing `disk_tier::DiskDtype` (`F32` 4 B/elem |
  `BF16` 2 B/elem). No new on-disk format/header — the dtype lives on the handle
  and in the manifest entry.
- **BF16TierTensor** (write): `residency::bf16_truncate_lossless(&[f32]) ->
  Option<Vec<u16>>` returns the truncated bf16 bits iff every value is
  representable, else `None` (→ keep f32). `write_or_reuse_expert` writes bf16
  when allowed + representable, f32 otherwise, and **reuses** an existing file of
  either dtype (size match) without rewriting.
- **TierReadMode** (read): warm reconstruction detects the dtype **by file size**
  (`numel*4`→F32, `numel*2`→BF16, anything else → error → fallback); the handle
  is opened with that dtype and materialises to f32 losslessly. No manifest
  round-trip needed for reconstruction, but the manifest also records the dtype
  for integrity.
- **SharedExpertCache** (FASE 5): a pinned slot inside the per-layer
  `ExpertCache` (`shared: Option<MoeDenseExpert>`), bounded by construction to
  **one expert per layer**. Populated on the first token, reused thereafter.

Only routed + shared **experts** go bf16; the backend tensors (router,
shared-gate, attention, embed, lm_head, norms) stay f32.

## FASE 3-4 — BF16 write + read

- `tensor/disk_tier.rs`: `write_bf16_tensor_named(path, &[u16])` (persistent,
  deterministic) + `open_existing(path, numel, dtype)` (generalises
  `open_existing_f32`). The existing `ensure_cpu` `Disk` arm already upcasts
  BF16→F32, so no execution-path change was needed — experts run in f32 exactly
  as before.
- `moe/residency.rs`: `TierEntry.dtype`; `write_or_reuse_expert` (bf16/f32
  decision + reuse); `place_at` / `from_dense_at` / `from_real_layer_at` thread
  `allow_bf16`; `from_tier`'s `wrap` detects dtype by size.
- `moe/runtime.rs`: `tier_bf16_from_env()` (default **on** when the persistent
  tier is active; `ATENIA_MOE_TIER_BF16=0` forces f32). Manifest **v3**: per-entry
  `dtype`, dtype-aware byte validation + `total_bytes`. The version bump
  invalidates v2 (f32-only) tiers so a warm load rebuilds them once (and may
  persist bf16).

## FASE 5 — Shared-expert cache

`ExpertCache` gains a pinned `shared` slot + `set_shared_enabled` +
`CacheStats { shared_hits, shared_misses }`. `forward_cached` resolves the shared
expert from the slot (hit) or materialises + pins it (miss); with the slot
disabled it resolves from the tier every token (the certified MOE-PROD-5
behaviour). `register_resident_moe_layer` enables the slot by default, disabled
by `ATENIA_MOE_SHARED_CACHE=0`. Bounded: one shared expert per layer.

## FASE 6 — Fallbacks (safety)

- bf16 not representable → tensor stays **f32** (per-tensor, automatic).
- `ATENIA_MOE_TIER_BF16=0` → all experts f32; `ATENIA_MOE_SHARED_CACHE=0` →
  shared resolved per token. Both revert to certified behaviour.
- Warm read: a tier file whose size is neither `numel*4` nor `numel*2`, a
  manifest dtype/size mismatch, a missing file, or a version/model_id mismatch →
  `try_warm_reconstruct` returns `None` → **certified shard path** (never a
  silent wrong answer).
- Default with the persistent tier off = byte-identical to MOE-PROD-5.

## FASE 7-8 — Bit-exactness + tests

- `src/moe/residency.rs` units: `bf16_truncate_lossless_detects_representability`
  (Some for bf16-source, None for arbitrary); `shared_cache_pins_and_is_bit_exact`
  (shared read from tier **exactly once** across 5 tokens — `shared_misses==1,
  shared_hits==4` — output == uncached `forward`); `shared_cache_disabled_resolves_every_token`.
- `tests/moe_bf16_tier_test.rs` (new): on a tiny Qwen-MoE checkpoint **masked to
  bf16-source**, a cold load writes the experts as **bf16** (manifest `dtype:
  "bf16"`, half size), and a warm reconstruction **after deleting the shards**
  generates **bit-identically**; the non-masked checkpoint **auto-falls back to
  f32**; `ATENIA_MOE_TIER_BF16=0` forces f32; `ATENIA_MOE_SHARED_CACHE=0`
  matches the default output.
- Full MoE regression (reconstruct, tier-persist, expert-cache, residency-tier,
  sharded) + the lib suite — green. Fixed a **pre-existing env-var race** in
  `tests/moe_tier_reconstruct_test.rs` (two tests mutate global env without a
  lock; added an `ENV_LOCK`), exposed by the v3 manifest timing change.

## FASE 9 — Benchmark (real Qwen1.5-MoE-A2.7B, prompt `22,25,29`, max-new 2)

A **rigorous same-prompt A/B** measured in one session (the MOE-PROD-5 handoff's
`701 s` figure used a **longer** prompt — 8 MoE rows vs 4 here — so it is **not**
like-for-like; the like-for-like MOE-PROD-5 behaviour is re-measured here as
**B**). All four runs emit the **identical** token ids `16, 15` → bit-exact on
the real 14.3 B model across the bf16 tier and the shared cache.

| Run | warm wall | tier on disk | gen bytes¹ |
|---|---|---|---|
| **B** — f32 tier, shared cache OFF (= MOE-PROD-5) | **350 s** | 53.3 GiB | 21.95 GiB |
| **A** — bf16 tier, shared cache OFF | **230 s** | **28.6 GiB** | 21.95 GiB |
| **C** — bf16 tier, shared cache ON (**MOE-PROD-6**) | **204 s** | 28.6 GiB | **12.67 GiB** |

¹ `tier_bytes_read` counts **f32-equivalent** element bytes (the size after
upcast), not raw disk bytes — so A reads the same *logical* volume as B but
**half** the *actual* NVMe bytes (bf16). C's drop to 12.67 GiB is the shared
expert no longer re-read every token.

Cold (writes the tier, load + 2 tokens): **bf16 1942 s / 28.6 GiB** vs
**f32 2587 s / 53.3 GiB** — bf16 also speeds the cold write (half the bytes).

**Headline:** like-for-like warm **350 s → 204 s = −146 s (~42 % faster)**,
bit-exact (`16, 15`), decomposing into **bf16 tier −120 s (~34 %)** +
**shared cache −26 s (~11 %)**.

## FASE 10 — Audit

- **Real gain:** warm load+gen **350 s → 204 s (~42 %, −146 s)** at identical
  work (same prompt, bit-exact `16, 15`); the on-disk tier shrinks **53.3 → 28.6
  GiB** (bf16) and the per-token generation reads **half** the NVMe bytes plus
  the shared expert only once per layer.
- **New bottleneck:** the warm **reconstruction** (reading ~the f32 backend +
  opening 4659 expert handles) and the **CPU matmul** of the routed + shared
  FFNs now dominate — the per-token NVMe read is no longer the largest term.
  Further wins would target the backend read (bf16 the backend too) or the CPU
  expert matmul (GPU offload), not the tier read volume.
- **Shared-cache impact:** **−26 s (~11 %)** on top of bf16 (A 230 → C 204);
  it removes the shared expert's per-token re-read (gen bytes 21.95 → 12.67 GiB).
  Larger relative effect at longer generations (the shared read is paid once,
  not per token).
- **BF16-tier impact:** the **bigger** lever — **−120 s (~34 %)** (B 350 → A
  230), halving the expert NVMe volume, plus a 24.7 GiB smaller on-disk tier and
  a faster cold write (2587 → 1942 s). Lossless: bit-exact output.

## Files modified

- `src/tensor/disk_tier.rs` — `open_existing(dtype)`, `write_bf16_tensor_named`.
- `src/moe/residency.rs` — `bf16_truncate_lossless`, `TierEntry.dtype`,
  `write_or_reuse_expert`, `allow_bf16` threading, `from_tier` dtype-by-size,
  `ExpertCache` pinned shared slot + stats, `forward_cached` shared caching,
  unit tests + `build_real_with_shared`.
- `src/moe/graph_op.rs` — `ATENIA_MOE_SHARED_CACHE` wiring, shared stats agg.
- `src/moe/runtime.rs` — `tier_bf16_from_env`, manifest **v3** (per-entry dtype),
  dtype-aware validation/total_bytes, `from_real_layer_at(allow_bf16)`.
- `tests/moe_bf16_tier_test.rs` — new. `tests/moe_tier_reconstruct_test.rs` —
  env-race lock.
- `docs/HANDOFF_MOE_PROD_6.md` (this) + `docs/STATUS.md`.

No new architecture/family/math/graph ops; routing and outputs unchanged; the
default path (persistent tier off) is unchanged.

## Deliverable answers

1. **What implemented:** a bf16 expert tier (auto-detected lossless, f32
   fallback) + a pinned per-layer shared-expert cache, both bit-exact with safe
   fallbacks; manifest v3.
2. **How the bf16 tier works:** at cold write each expert tensor is truncated to
   bf16 iff every value is bf16-representable (else kept f32); the on-disk dtype
   is detected by file size on warm read and upcast to f32 by the existing
   `ensure_cpu` disk arm — experts execute in f32 exactly as before.
3. **How the shared cache works:** the shared expert is materialised once into a
   pinned slot in the per-layer `ExpertCache` and reused every subsequent token
   (it never changes during a generation); disable with `ATENIA_MOE_SHARED_CACHE=0`.
4. **Tests:** bf16 detect + shared-cache pin/disable units, bf16-tier
   write/read/fallback/forced-f32 integration (warm == cold, shards deleted),
   full MoE regression + lib suite.
5. **Time before:** 350 s (like-for-like MOE-PROD-5 warm: f32 tier, shared off,
   same prompt — re-measured this session; the handoff's 701 s used a longer
   prompt and is not comparable).
6. **Time after:** 204 s (MOE-PROD-6 warm: bf16 tier + shared cache).
7. **Real gain:** −146 s (~42 %) warm, bit-exact (`16, 15`); tier 53.3 → 28.6
   GiB; decomposes into bf16 −120 s + shared cache −26 s.
8. **New bottleneck:** warm reconstruction (f32 backend read + 4659 handle
   opens) + CPU expert matmul; per-token NVMe read is no longer dominant.
9. **Commit:** see git log.
10. **CI:** see push.
