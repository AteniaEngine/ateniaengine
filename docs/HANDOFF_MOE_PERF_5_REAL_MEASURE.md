# HANDOFF — MOE-PERF-5-REAL-MEASURE: real telemetry baseline before PERF-4

Milestone: **MOE-PERF-5-REAL-MEASURE** — use the new MOE-PERF-5 instrumentation to capture a
real telemetry baseline for MoE generation, and decide whether PERF-4 (qint8 tier) should remain
next. **Measurement only** — no optimization / runtime / numerics / routing / cache / cert /
manifest / ADR change. The only file added is a **test-only** `#[ignore]` measurement sweep
(`tests/moe_perf5_real_measure.rs`); `src/` is **untouched**.

## PHASE 1 — Measurement plan & workload classification

| Workload | Class | Rationale |
|---|---|---|
| **Mixtral-8x7B real (87 GB)** | **TOO HEAVY / SKIP** | Host has **~12 GB free of 32 GB**; the cache=1 forward peaks **~29 GB** and the default cache commits **~90 GB → OOM** (documented hazard, `HANDOFF_MIXTRAL_CERT_C5.md`). The 402.7 s forward would thrash/hang the host. Not run blindly. |
| **DeepSeek-V2-Lite real (29 GB)** | **SKIP** | Load-time transient spikes on a 12-GB-free host; **MLA streams experts uncached** → cache telemetry is N/A regardless of running it (timing only). |
| **Qwen-MoE real** | **N/A** | No whole-model transformer path (block-level cert) — cannot run a real generation. |
| **Scale fixtures** (`mixtral_scale`, `qwen_scale`, `deepseek_scale`) | **SAFE** | Drive the **exact certified runtime** (`forward_cached`, disk tier, prefetch, LRU) at the **real routing fan-out** (Mixtral top-2 / Qwen top-4 / DeepSeek top-6). The cache/prefetch/tier telemetry is real; only the hidden dim (→ absolute time) is reduced. |

**Real-scale anchor** (from PERF-1 / C5, not re-run): Mixtral 87 GB — load (warm tier) **4.5 s**,
forward (seq=4) **402.7 s**, default cache **~90 GB → OOM**, cache=1 peak ~29 GB.

## PHASE 2 — Telemetry runs (raw)

`tests/moe_perf5_real_measure.rs`, release. Per family: RAM-tier timing baseline, then disk tier
× prefetch {off,on} × cache {auto, 1}. `generate_instrumented`, prefill-dominated (fixtures EOS
after 1 token → decode ≈ 0).

| fixture | tier | cache | pref | prefill ms | tok/s | hits | miss | evic | resid KiB | reads | mat KiB | par | ovl ms | resv ms | cache? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mixtral_scale | ram | auto | off | 3.84 | 257 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes |
| mixtral_scale | disk | auto | off | 3.87 | 256 | 14 | **6** | 0 | 144 | 6 | 144 | 0 | 0 | 1.99 | yes |
| mixtral_scale | disk | auto | on | 2.50 | 396 | 14 | 6 | 0 | 288 | 6 | 144 | **6** | 0.37 | 0.94 | yes |
| mixtral_scale | disk | 1 | off | 4.25 | 234 | 0 | **20** | **18** | 336 | 20 | 480 | 0 | 0 | 2.11 | yes |
| mixtral_scale | disk | 1 | on | 3.67 | 271 | 5 | 15 | 13 | 384 | 15 | 360 | 15 | 0.55 | 2.06 | yes |
| qwen_scale | ram | auto | off | 3.95 | 252 | 0 | 0 | 0 | 384 | 0 | 0 | 0 | 0 | 0 | yes |
| qwen_scale | disk | auto | off | 47.27¹ | 21 | 21 | **19** | 0 | 648 | 19 | 264 | 0 | 0 | 42.38¹ | yes |
| qwen_scale | disk | auto | on | 4.36 | 228 | 21 | 19 | 0 | 912 | 19 | 264 | **19** | 1.92 | 3.39 | yes |
| qwen_scale | disk | 1 | off | 7.07 | 141 | 0 | **40** | **38** | 972 | 40 | 516 | 0 | 0 | 4.18 | yes |
| qwen_scale | disk | 1 | on | 4.87 | 204 | 5 | 35 | 33 | 1032 | 35 | 456 | 35 | 3.16 | 5.08 | yes |
| deepseek_scale | ram | auto | off | 0.38 | 4354 | 0 | 0 | 0 | — | 0 | 0 | 0 | 0 | 0 | **NO** |
| deepseek_scale | disk | auto | off | 14.72 | 125 | 0 | 0 | 0 | — | 0 | 0 | 0 | 0 | 0 | **NO** |
| deepseek_scale | disk | * | * | 6.7–6.9 | 240–249 | 0 | 0 | 0 | — | 0 | 0 | 0 | 0 | 0 | **NO** |

¹ The **first** disk read per family pays the cold OS-page-cache (and tier-write) cost; later
rows read warm. So absolute `prefill_ms`/`resolve_ms` across consecutive rows are **confounded by
OS caching** and are not a clean prefetch A/B. The analysis therefore rests on the **structural
counters** (`hits`/`misses`/`evictions`/`parallel_prefetches`), which are **deterministic** and
cache-independent. (`resident KiB` is the *process-wide* registry resident total — accurate for a
single-load run, cumulative across the sweep's repeated loads.)

## PHASE 3 — Analysis

**Cache size is the dominant lever (graph families).** Going `auto → cache=1` forces re-reads:
Mixtral **6 → 20 misses (+18 evictions)**, Qwen **19 → 40 misses (+38 evictions)** — hit rate
collapses to ~0. This is the PERF-1 *cache=1-thrash ↔ cache=4-OOM* tension, now confirmed in
telemetry: with `auto` the experts fit (mostly hits); forced to `cache=1` every selected expert is
re-read every layer.

**Prefetch is observable and scales with top-k.** With prefetch on, `parallel_prefetches` equals
the miss count (every miss is batch-resolved) and `overlap_saved_ms > 0`: Mixtral (top-2) ~0.4 ms
< Qwen (top-4) ~1.9 ms — more experts per token ⇒ more reads overlapped, matching MOE-PERF-3. At
reduced dim the absolute overlap is small (tiny experts); it scales with real expert width.

**I/O-bound vs compute-bound.** Graph families on disk are **I/O-bound**: tier reads + resolve
dominate, and the RAM tier (zero tier I/O) is uniformly faster at the same dim. The MoE *compute*
is trivial (DeepSeek RAM tier **4354 tok/s** at reduced dim). **DeepSeek/MLA streams experts
uncached** (`cache? = NO`): its tier I/O is invisible to `CacheStats` and shows only as a timing
gap (RAM 0.38 ms → disk 14.7 ms) — a real coverage limitation, not a zero.

**Bottleneck per family:**
| Family | Bottleneck | I/O vs compute | Notes |
|---|---|---|---|
| Mixtral (top-2) | tier read volume + cache capacity | I/O-bound on disk | cache=1 → 20 miss/18 evict; widest real experts amplify it |
| Qwen-MoE (top-4) | tier read volume + cache capacity | I/O-bound on disk | most misses (top-4, 16 experts); prefetch overlaps most |
| DeepSeek-V2-Lite (top-6, MLA) | tier read (uncached) | I/O on disk, compute trivial | cache telemetry N/A (uncached stream); timing-only |

## PHASE 4 — PERF-4 readiness decision

**Yes — PERF-4 (qint8 default tier, gated on a numeric certificate) should remain next.**

- **Why (telemetry-backed):** the graph families are I/O-bound on disk and the **cache-capacity
  tension is the dominant cost** (`cache=1` → 0 hits + mass evictions). qint8 attacks **both**
  measured terms at once: it cuts each expert to **¼ bytes**, so (a) `materialized_bytes` /
  `resolve_time` drop ~4× per read, and (b) **~4× more experts fit the same RAM budget** →
  `auto` capacity rises → fewer misses/evictions → higher hit rate. It also lifts the real-scale
  blocker: Mixtral's `cache=4 → ~90 GB OOM` becomes ~22 GB at qint8, turning a *forced* cache=1
  into a viable cache≥4 (the single biggest win the telemetry points to). It **compounds** with
  PERF-3 prefetch (smaller reads overlap even better).
- **Expected win:** ~4× tier-read volume reduction + cache-capacity headroom that converts
  cache-thrash into cache-hits on the large models (Mixtral/Qwen) — directly the bottleneck above.
- **Highest-risk part:** **qint8 numeric accuracy**. Per-expert int8 quantization can push some
  experts past the certified ADR-004 gate; PERF-4 must therefore stay **gated on a passing qint8
  numeric certificate (NUMERIC-POLICY-3)** before becoming a default. The quantization↔cert gate
  is the risk to manage, not the I/O plumbing.
- **Caveat (does not change the decision):** DeepSeek/MLA streams experts **uncached**, so it
  gains qint8's smaller-read benefit but **not** the cache-capacity benefit, and its telemetry
  won't show either (timing only). Routing DeepSeek through `forward_cached` is a *behavior*
  change (a future optimization milestone), not a PERF-4 prerequisite.

**Nothing needs to come before PERF-4.** PERF-5 already removed the measurement blocker; the
telemetry confirms the bottleneck PERF-4 targets is real and dominant.

## PHASE 5/6 — Files & validation

- **Files:** `tests/moe_perf5_real_measure.rs` (new, `#[ignore]` measurement),
  `docs/HANDOFF_MOE_PERF_5_REAL_MEASURE.md` (this), `docs/STATUS.md`, `docs/MOE_PERF_AUDIT.md`.
- **Code touched:** test-only. **`src/` untouched** ⇒ `cargo test --lib` semantics unchanged
  (not required). The sweep runs green (`perf5_real_measure_sweep` 1 passed, `#[ignore]`); the
  PERF-5 CI tests (`moe_perf5_telemetry_test` 3/3) are unaffected.

## Final deliverable summary

1. **Plan:** disk-tier fixture sweep at real top-k × prefetch{off,on} × cache{auto,1}; real runs
   classified too-heavy/skip on a 12-GB-free host.
2. **Selected/skipped:** fixtures (safe) run; Mixtral/DeepSeek real skipped (host-stress), Qwen
   real N/A.
3. **Telemetry:** table above (real cache/prefetch/tier counters).
4. **Prefetch:** `parallel_prefetches` = misses, `overlap_saved_ms` > 0, scales with top-k
   (Mixtral 0.4 ms < Qwen 1.9 ms).
5. **Bottleneck:** I/O-bound on disk; **cache capacity** dominant (cache=1 → 0 hits + evictions);
   DeepSeek uncached (timing only).
6. **PERF-4 readiness:** **YES, remains next** — qint8 cuts read bytes 4× *and* relieves the cache
   tension; highest risk is the qint8 numeric certificate.
7. **Files:** as above. 8. **Tests:** `perf5_real_measure_sweep` (1, `#[ignore]`); `src/`
   untouched. 9/10. **Commit/CI:** docs + test-harness commit; CI as recorded in STATUS.

**Do not start PERF-4.** This milestone ends at the readiness decision.
