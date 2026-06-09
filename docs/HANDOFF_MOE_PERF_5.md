# HANDOFF — MOE-PERF-5: MoE-generation telemetry (observability parity with dense)

Milestone: **MOE-PERF-5** — bring MoE-generation instrumentation to **parity with dense
generation**: expose reliable timing + expert-cache / prefetch / tier statistics for real MoE
workloads, so PERF-3/PERF-4 can be measured end-to-end instead of on surrogates (the gap
MOE-PERF-3-VALIDATION hit). **Instrumentation only** — no optimization, no behavior change, no
numerics / routing / MLA / cache / loader / certification / manifest / ADR change. The default
output is byte-for-byte unchanged; telemetry is **opt-in**.

## PHASE 1 — Audit (dense vs MoE)

| Metric | Dense (`cli_generate.rs`) | MoE before PERF-5 |
|---|---|---|
| load time | ✅ `load_secs` | ❌ (not surfaced) |
| total gen / tok-s | ✅ `total_secs`, `tps` | ❌ (`generate` returns only ids) |
| first-token / prefill heartbeat | ✅ visible heartbeat | ❌ |
| prefill vs decode split | ⚠️ partial (heartbeat) | ❌ (internal only, behind `ATENIA_MOE_CACHE_STATS`) |
| loader/matmul counters | ✅ `CounterSnapshot` deltas | ❌ |
| expert-cache hit/miss | n/a | ⚠️ `CacheStats` existed but **never surfaced** to a generation |
| prefetch overlap | n/a | ⚠️ `parallel_prefetches`/`prefetch_wall_nanos` existed, unsurfaced |
| tier bytes / reads | n/a | ⚠️ `tier_bytes_read` existed, unsurfaced |

**Gap:** the MoE generate path (`controlled_moe_generate → MoeRuntime::generate`) returned only
token ids. All the raw signal already existed in `CacheStats` + `aggregate_resident_cache_stats`;
it was simply never gathered around a generation or timed at stage granularity.

## PHASE 2/3 — Telemetry design + implementation (additive, read-only)

- **`src/moe/telemetry.rs` (new):** `MoeGenTelemetry` (load / prefill / decode / first-token /
  total / tokens / tok-s; cache hits / misses / evictions / resident_bytes; parallel_prefetches /
  overlap_saved_ms / resolve_time_ms; materialized_bytes / tier_reads; `cache_telemetry_available`)
  + `StageTimings` + `cache_stats_delta` (field-wise `after − before`, saturating) + `render()`
  (human-readable block mirroring the dense layout). `Serialize` for JSON.
- **`src/moe/graph_op.rs`:** `aggregate_resident_cache_stats` extended to also sum `evictions`,
  `parallel_prefetches`, `prefetch_wall_nanos`; new `aggregate_resident_cache_resident_bytes()`.
- **`src/moe/generate.rs` + `src/moe/mla.rs`:** added `*_timed` variants
  (`generate_greedy_tiny_eos_timed`, `DeepseekWeights::generate_greedy_eos_timed`) returning
  `StageTimings`; the existing public fns are now thin wrappers over them → **bit-identical
  generation** (guarded by the existing `moe::generate`/`moe::mla` unit tests, still green).
- **`src/moe/runtime.rs`:** `MoeRuntime::generate_instrumented(prompt, max) -> (Vec<u32>,
  MoeGenTelemetry)` — wraps the unchanged backend call in read-only timers + a before/after
  `aggregate_resident_cache_stats` **snapshot diff** (isolates this generation from the
  process-global cumulative registry). `load_ms` left 0 for the caller to fill.
- **`src/moe/production.rs`:** `controlled_moe_generate_instrumented(dir, prompt, max)` — the same
  gated/certified/opt-in path as `controlled_moe_generate`, identical guards, times the load
  (`load_ms`) + generation, returns telemetry. Token output **bit-identical**.

## PHASE 4 — CLI surface (opt-in, default unchanged)

`src/cli_generate.rs::run_moe_text`: with **`ATENIA_MOE_TELEMETRY=1`** the instrumented entry is
used and `tele.render()` is printed to stderr after generation. Without the env var the original
`controlled_moe_generate` path runs — **default behaviour byte-for-byte unchanged**.

## PHASE 5/6 — Validation + demonstration

`tests/moe_perf5_telemetry_test.rs` (new):

- `perf5_{mixtral,qwen,deepseek}_*` (CI, RAM tier): instrumented tokens **== `generate()`**
  (bit-identical), timing populates + is self-consistent (`tok/s == tokens / total`,
  `first_token == prefill`), deterministic, and `cache_telemetry_available` is correct per family
  (Mixtral/Qwen `true`, **DeepSeek `false`** — it streams experts uncached). **3/3 pass.**
- `perf5_disk_tier_cache_metrics_demo` (`#[ignore]`, mutates env): graph fixtures on the **disk
  tier** with prefetch on — proves the cache/prefetch/tier metrics populate. Example output:

```
[ATENIA] MoE generation telemetry              [ATENIA] MoE generation telemetry
  load        :       0.0 ms                     load        :       0.0 ms
  prefill     :      33.6 ms (first token 33.6)   prefill     :     112.0 ms (first token 112.0)
  decode      :       0.0 ms                     decode      :       0.0 ms
  total gen   :      33.6 ms                     total gen   :     112.1 ms
  tokens      :         1 (29.72 tok/s)          tokens      :         1 (8.92 tok/s)
  expert cache: hits 14 / misses 6 (70.0% hit)   expert cache: hits 21 / misses 19 (52.5% hit)
  tier I/O    : 6 reads, 0.1 MiB, resolve 40.8ms  tier I/O    : 19 reads, 0.3 MiB, resolve 222.8ms
  prefetch    : 6 parallel, 11.8 ms overlapped    prefetch    : 19 parallel, 125.2 ms overlapped
       (mixtral_scale, disk tier)                      (qwen_scale, disk tier)
```

(The scale fixtures hit EOS on the first greedy token, so `decode = 0`; the point is **all
metric families appear** — timing, cache hit/miss + hit-rate, prefetch overlap, tier reads/bytes.)

## PHASE 8 — Test results

- `moe::telemetry` unit: **4/4** (delta saturating, tps/overlap, overlap-zero-without-prefetch,
  DeepSeek-flag render). `moe::generate` 3/3, `moe::mla` 4/4 (bit-identical guards green).
- `tests/moe_perf5_telemetry_test`: **3/3** + 1 `#[ignore]` demo.
- **`cargo test --lib`: 886 passed / 0 failed.** `moe_scale_cert_test`: **3/3** (no certification
  or generation-behavior regression — instrumented output equals the certified path).

## Coverage & limitations (honest)

- **Timing** (load / prefill / decode / first-token / total / tok-s): **all families**.
- **Expert-cache / prefetch / tier** metrics: the **graph families (Mixtral, Qwen-MoE) on the disk
  tier**, where experts stream through the registered `ExpertCache` (`forward_cached`). On the RAM
  tier those experts are resident (registered `Real`) → metrics are a **true zero** (no tier I/O).
- **DeepSeek (MLA)** runs its experts through the *uncached* `ResidentExpertLayer::forward` (it is
  not in the graph cache registry), so it reports **timing only**; `cache_telemetry_available =
  false` flags this so a zero is never read as a measurement. Routing DeepSeek through
  `forward_cached` would be a **behavior change** (out of scope for an instrumentation milestone) —
  noted as a candidate for a future optimization milestone.

## Files

- `src/moe/telemetry.rs` (new), `src/moe/graph_op.rs`, `src/moe/generate.rs`, `src/moe/mla.rs`,
  `src/moe/runtime.rs`, `src/moe/production.rs`, `src/moe/mod.rs`, `src/cli_generate.rs`,
  `tests/moe_perf5_telemetry_test.rs` (new), `docs/HANDOFF_MOE_PERF_5.md`, `docs/STATUS.md`,
  `docs/MOE_PERF_AUDIT.md`.

## Outcome / next

Observability now reaches **parity with dense generation** for MoE: a single `ATENIA_MOE_TELEMETRY=1`
run reports load/prefill/decode/first-token/tok-s for any family, plus real expert-cache /
prefetch / tier I/O for the disk-tier graph families. This **unblocks measuring PERF-3/PERF-4 on
real certified runs** (the MOE-PERF-3-VALIDATION blocker). **Next: PERF-4** (qint8 default tier,
gated on a numeric cert), now measurable end-to-end via this telemetry. **Do not start PERF-4.**
The milestone ends at observability parity.
