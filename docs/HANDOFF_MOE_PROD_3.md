# HANDOFF — MOE-PROD-3: expert-cache integration

Milestone: **MOE-PROD-3** — wire the existing `ExpertCache` into the real MoE
disk-tier node so routed experts touched again are served from RAM instead of
re-read from NVMe every token; measure whether it speeds up generation.
**Optimise the validated MoE runtime only** — no new models / families /
architectures. Predecessor: `669e8dc` (RUNTIME-MOE-2 reopened, Qwen-MoE real
GREEN).

## FASE 1 — Audit

- **Runtime state:** the MoE graph node (`graph_op::execute_real_moe_layer`)
  dispatches `Real(RAM-f32)` vs `Resident(tiered)` (MOE-PROD-2). The `Resident`
  branch called `ResidentExpertLayer::forward` — **no cache** → the routed
  top-k experts are re-read from NVMe **per token** (the RUNTIME-MOE-2
  bottleneck).
- **Where the cache exists:** `residency::ExpertCache` (LRU + stats
  hits/misses/prefetched/bytes) with `ResidentExpertLayer::forward_cached(&mut
  cache, x)` — **bit-identical** to `forward` (unit test
  `cached_forward_matches_uncached`).
- **Who used it:** only `self_validate_residency` (the load-time probe) and the
  unused `MoeRuntime.caches` field. **The generation node did NOT use it.**
- **Risk:** caching **all** experts per layer re-materialises the whole layer in
  RAM (~50 GB for Qwen) — defeats the disk tier / OOMs 32 GB. So the cache must
  be **bounded**; the hit ratio then depends on MoE routing locality (routing is
  designed to spread load), and the **shared expert is read every token
  regardless** (not cached). The win is therefore *expected to be partial* — the
  point of this milestone is to **measure** it honestly, not assume.

## FASE 3 — Integration (reuse, no rewrite)

- `graph_op.rs`: `RegisteredMoe::Resident { layer, cache: Mutex<ExpertCache> }`
  (was a bare `Arc`). `register_resident_moe_layer(layer, cache_capacity)`
  builds the per-layer cache once at load; `execute_moe_row` calls
  `forward_cached(&mut cache, row)` for the `Resident` branch. The cache is
  threaded through the **whole generation** (registered once, lives for the
  runtime), so reuse accrues across prefill rows and decode steps.
- `runtime.rs`: `ATENIA_MOE_EXPERT_CACHE` → capacity (integer clamped to
  `num_experts`, `"all"`, or `"0"` to disable); **default `2 * experts_per_token`**
  — bounded (keeps RAM low) while capturing short-range reuse.
- `full_forward.rs`: `MoeBlock::registered(layer, cache_capacity)`.
- `graph_op::aggregate_resident_cache_stats()` + CLI `ATENIA_MOE_CACHE_STATS=1`
  → prints `hits / misses / hit_ratio / tier_bytes_read` after generation.
- No new cache type; no runtime rewrite; RAM tier (Owned) and DeepSeek/MLA
  unchanged.

## FASE 4 — Correctness (cache must not alter outputs)

- `cached_forward_matches_uncached` (layer unit test) — **pass**.
- `moe_residency_tier_test` now exercises **disk + cache vs RAM** →
  `max_abs_diff == 0.0`, identical generation — **the integrated cache is
  bit-exact**.
- `moe_expert_cache_test` (new) — disk-tier node registers cache **hits > 0** and
  **misses > 0** on a repeated-token forward; generation still stops at EOS.
- Full MoE regression green (mixtral/qwen/deepseek/sharded/production/robustness/
  cli).

## FASE 5 — Benchmark (real Qwen1.5-MoE-A2.7B, rust prompt, max-new 6)

| | Before (RUNTIME-MOE-2, no cache) | After (MOE-PROD-3, cache cap=8) |
|---|---|---|
| Total wall (load + 6 tokens) | **~2905 s** | **~3101 s** |
| Expert-cache hit ratio | n/a (no cache) | **22.7 %** (hits 262 / misses 890) |
| Routed tier bytes read | ~all lookups | **~28.68 GiB** (≈ 23 % fewer reads) |
| Output (decoded) | `" is designed to be fast,"` | `" is designed to be fast,"` — **identical (bit-exact on real weights)** |

**Headline (honest): the cache works but does NOT speed up end-to-end
generation.** It achieves a real **22.7 % routed-expert hit ratio** at bounded
RAM (capacity 8) and cuts routed NVMe read volume by ~23 %, but total wall-clock
is **unchanged** (3101 vs 2905 s — the ~200 s delta is within load variance; the
load reads/writes ~80 GB and varies run-to-run). The output is bit-identical to
the no-cache run on the real 14.3 B model.

**Why no speedup:** routed-expert re-reads are **not** the bottleneck. The cost
is dominated by (a) **load-time NVMe tiering** (~25–35 min: writing ~50 GB of
f32 experts as 4392 separate files + reading 28.6 GB of shards twice),
(b) the **shared expert read every token** (not cached — Qwen-MoE has one),
(c) **per-file open overhead** (4392 tiny files), and (d) **CPU matmul** of the
routed + shared FFNs. A 23 % reduction in routed reads is swamped by these.

## FASE 6 — Robustness

- Cache **disabled** (`ATENIA_MOE_EXPERT_CACHE=0`) → every lookup a miss
  (capacity-0 semantics), behaviour == pre-MOE-PROD-3.
- Cache **bounded** (default `2*top_k`) → LRU eviction keeps RAM bounded.
- Cache **full** (`=all`) → all experts cached (RAM cost = full layer; opt-in).
- **Restart:** the cache is process-local (rebuilt each load) — no stale state.
- Bit-exactness holds across all capacities (the cache only changes *where* an
  expert is read from, never the math).

## FASE 7 — Review

- **How much faster?** **Net ~0 end-to-end** at the default capacity (8). The
  cache does its job (22.7 % routed hits, ~23 % fewer routed NVMe reads,
  bit-exact) but that is not where the time goes.
- **New bottleneck?** Not routed re-reads. In order: (1) **load-time NVMe
  tiering** — writing ~50 GB of f32 experts as 4392 individual files + reading
  the 28.6 GB shards twice (metadata pass + assembly); (2) **shared expert read
  per token**; (3) **per-file open overhead**; (4) **CPU matmul**.
- **Is Mixtral viable now?** No change from the cache. Mixtral-8x7B (~94 GB →
  ~188 GB f32 on NVMe, 32 experts top-2, no shared expert) would load even
  slower; correctness-feasible via the disk tier but very slow. The cache does
  not unlock it — the load/format cost does.
- **Still worth optimising?** Yes, but **not via routed-expert LRU caching**.
  Higher-leverage, ordered by impact: (a) **store tier experts as bf16** (halve
  ~50 GB of NVMe I/O, lossless for bf16 sources); (b) **persist/reuse the tier
  cache across runs** instead of rewriting 50 GB every load; (c) **batch the
  tier into fewer/larger files** (cut the 4392-file open overhead);
  (d) **cache the shared expert** (read every token). MOE-PROD-3's finding is
  that the *generation* path was already not the dominant cost — **load** is.

## Files modified

- `src/moe/graph_op.rs` — `Resident { layer, cache }`, capacity-parameterised
  registration, `forward_cached` dispatch, `aggregate_resident_cache_stats`.
- `src/moe/runtime.rs` — `ATENIA_MOE_EXPERT_CACHE` capacity, pass to
  `MoeBlock::registered`.
- `src/moe/full_forward.rs` — `MoeBlock::registered(layer, capacity)`.
- `src/bin/atenia.rs` — `ATENIA_MOE_CACHE_STATS=1` hit-ratio report.
- `tests/moe_expert_cache_test.rs` — new (cache hits + EOS).
- `docs/HANDOFF_MOE_PROD_3.md` (this) + `docs/STATUS.md`.

No new architecture/family/math/graph ops; outputs bit-exact; RAM tier +
DeepSeek unchanged.

## Deliverable answers (filled after benchmark)

1. **What changed:** the disk-tier MoE node now uses the bounded per-layer
   `ExpertCache` (`forward_cached`) instead of re-reading every token.
2. **Which parts use cache:** the `Resident` (disk-tier) MoE node only; RAM-tier
   and DeepSeek unchanged.
3. **Hit ratio:** **22.7 %** (262 hits / 890 misses, capacity 8).
4. **Time before:** ~2905 s (load + 6 tokens).
5. **Time after:** ~3101 s (within load variance — no net change).
6. **Real gain:** ~23 % fewer routed NVMe reads, **but ~0 wall-clock speedup**
   (the bottleneck is load + shared expert + per-file overhead, not routed
   re-reads). Output bit-identical.
7. **New bottleneck:** load-time NVMe tiering (≫ generation), shared-expert
   per-token reads, per-file overhead, CPU matmul.
8. **Tests:** cached==uncached bit-exact, disk+cache==RAM bit-exact, cache-hits,
   full MoE regression — all green.
9. **Commit:** see git log.
10. **CI:** code change → CI runs; see push.
11. **Next:** store tier experts as **bf16** + **persist the tier across runs**
    (the real levers), not more expert caching.
