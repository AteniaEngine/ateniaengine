# HANDOFF — MOE-PERF-2 (expert cache): auto-sized cache + bf16-resident experts

> **Naming note.** `docs/HANDOFF_MOE_PERF_2.md` already exists and documents a *different*,
> earlier effort that reused the "MOE-PERF-2" label (a GPU expert-FFN attempt that was
> abandoned as slower). To avoid destroying that history this handoff uses a distinct
> filename. This is the current **MOE-PERF-2 — Auto-Sized Expert Cache + BF16 Resident
> Experts** milestone.

Milestone: **MOE-PERF-2** — cut expert-cache RAM pressure and rematerialization, the
highest-ROI item from the MOE-PERF-1 audit, **without changing numerics, certification
results, manifests, ADRs, routing, MLA, attention, or generation logic**.

## What it does

Two changes to the expert residency cache (`src/moe/residency.rs`) + its sizing
(`src/moe/runtime.rs`):

- **PERF-2A — auto-sized cache.** The per-layer cache capacity is no longer a fixed
  `2·top_k` (which commits ~90 GB on Mixtral → OOM). It is **auto-sized** so the total
  resident footprint (`n_layers × capacity × per_expert_bytes`) fits a RAM budget
  (default 50% of available RAM; `ATENIA_MOE_CACHE_RAM_FRACTION`). The explicit
  `ATENIA_MOE_EXPERT_CACHE` override (integer / `all`) still wins — backward compatible and
  fully reproducible. Clamped to `[1, num_experts]` (never 0).
- **PERF-2B — bf16-resident experts.** Cached experts are stored **bf16** when their f32
  values are bf16-representable (low 16 mantissa bits zero — exactly the bf16-tier case),
  else verbatim f32. Decoding back to f32 on a cache hit is **bit-exact** (the bf16
  truncate/decode round-trip is the identity for such values). Default on; opt-out
  `ATENIA_MOE_CACHE_BF16=0`. The pinned shared-expert slot stays f32 (one per layer,
  negligible).

## Numerical safety (critical)

bf16 residency is **lossless by construction** — a projection is kept bf16 **only** when
`bf16_truncate_lossless` succeeds (the f32 came from a bf16 source), otherwise it stays f32.
Decoding is the exact inverse. So a cache hit reproduces the **bit-identical** f32 weights a
fresh tier resolve would, on any tier dtype, with or without the int8 sim. The forward is
bit-exact at any cache capacity (proven by `cached_forward_matches_uncached`), so model
output is **unchanged and deterministic** — capacity affects only RAM / I-O. **No numerics,
no thresholds, no manifests, no ADRs, no routing/MLA/attention/generation changed.**

## Measurements (real)

- **RAM per cached expert: −50% (exact).** `perf2_bf16_cache_is_bit_exact_and_smaller`
  asserts `resident_bytes_f32_equiv == 2 × resident_bytes` and `resident_bytes_saved > 0`
  for bf16-representable experts — a measured **2× reduction** in cache residency, bit-exact.
- **Auto-size math (deterministic).** `auto_expert_cache_capacity(budget, n_layers,
  per_expert_bytes, num_experts)` unit-tested. For the real **Mixtral-8x7B** (32 layers,
  per-expert 176 M params): f32 = 704 MB/expert, **bf16 = 352 MB/expert**. The fixed
  `2·top_k = 4` cache = 32 × 4 × 704 MB = **90 GB (f32) → OOM**; in bf16 the same capacity
  = **45 GB** (fits on a ≥~50 GB-free host where f32 OOM'd). On this 32 GB host
  (~16 GB free → ~8 GB budget) auto-sizing selects **capacity 1 automatically** (no manual
  `ATENIA_MOE_EXPERT_CACHE=1`, no OOM by construction); on larger hosts it selects the
  **largest cache that fits — doubled by bf16** → fewer re-reads → less of the MOE-PERF-1
  402.7 s forward thrash.
- **Compute unaffected:** `tests/moe_perf_scale_bench.rs` (reduced-dim) timings are
  unchanged (the bf16 decode-on-hit is SIMD, far cheaper than the NVMe read it still avoids).

> The real-weight 87 GB Mixtral forward delta was **not** re-measured here (a ~1 h heavy
> load that previously stressed the host — out of scope for this change, against the
> "no unexpected heavy run" guard). The benefit is the measured 2× cache-RAM reduction + the
> auto-size that removes the OOM and lets a larger cache fit; the forward-time gain is
> proportional to the additional experts that now stay resident.

## Before / after

| | Before (PERF-1) | After (PERF-2) |
|---|---|---|
| Default cache capacity | fixed `2·top_k` | **auto-sized to a RAM budget** (override still wins) |
| Mixtral default behaviour | `2·top_k=4` → ~90 GB → **OOM** (manual `=1` workaround) | **auto cap=1 on a 32 GB host (no OOM, no manual tuning)** |
| Cached expert storage | F32 (704 MB) | **bf16 when lossless (352 MB) — ½ RAM** |
| Same-capacity cache RAM (Mixtral cap=4) | 90 GB | **45 GB** |
| Numerics / certs | — | **bit-identical (unchanged)** |

## Instrumentation (PERF-1 inventory reused + extended)

`CacheStats` (hits/misses/evictions/tier_bytes_read/resolve_nanos/…) unchanged; added cache
snapshots: `resident_bytes()` (now bf16-aware), `resident_bytes_f32_equiv()`,
`resident_bytes_saved()`, `bf16_entries()`, `bf16_resident()`. No new telemetry framework.

## Validation

- `moe::residency` unit tests: **18 pass** (incl. 4 new — `perf2_auto_capacity_fits_budget`,
  `perf2_cached_weight_bf16_is_lossless`, `perf2_bf16_cache_is_bit_exact_and_smaller`,
  `perf2_bf16_disabled_is_f32_and_still_bit_exact`).
- **Certifications unaffected:** `moe_scale_cert_test` (Mixtral/Qwen/DeepSeek) **3/3** — same
  `max_abs_diff`, argmax exact, greedy→EOS, deterministic, with the new cache path.
- Full `cargo test --lib` green; perf bench green.

## Validation checklist (proved)

- **No certification regression:** scale-cert 3/3 unchanged.
- **No numerical regression:** bf16 residency lossless by construction; cache bit-exact tests.
- **No API regression:** `ExpertCache::new`/`forward_cached`/`prefetch` signatures unchanged
  (internal `put` is private); added methods only.
- **Deterministic behaviour:** output bit-exact at any capacity; capacity override available.

## Files

- `src/moe/residency.rs` (bf16-resident `CachedExpert`/`CachedWeight` + `ExpertCache`
  storage/instrumentation + `auto_expert_cache_capacity` + tests).
- `src/moe/runtime.rs` (`expert_cache_capacity` auto-sizing + `cache_ram_budget_bytes`).
- docs: this handoff + `STATUS.md`.

## Risks / caveats

- Auto-sized capacity depends on available RAM at load (varies by host/run) → the *cache
  size* is not fixed across hosts, but the *model output* is bit-exact and deterministic at
  any capacity. Use `ATENIA_MOE_EXPERT_CACHE=<n>` for a fully reproducible capacity.
- bf16 residency only saves RAM when experts are bf16-representable (the default bf16 tier);
  an f32 tier falls back to f32 storage (still bit-exact, no savings) — by design.
- The 87 GB real-weight forward-time delta is not re-measured (heavy run); the RAM win is
  measured exactly.

## Future PERF roadmap (from PERF-1)

PERF-3 expert prefetch / async tier reads · PERF-4 qint8 default tier (gated on a numeric
cert) · PERF-5 MoE-generate instrumentation parity. MLA latent cache + GPU expert offload
deferred (high risk). **Do not start PERF-3.**
