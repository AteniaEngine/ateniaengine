# HANDOFF — MOE-PERF-3 (prefetch): async (parallel) expert prefetch for the disk tier

> **Naming note.** `docs/HANDOFF_MOE_PERF_3.md` already exists and documents a *different*,
> earlier effort that reused the "MOE-PERF-3" label (a shared-expert VRAM-residency attempt,
> stopped as no-ROI). To avoid destroying that history this handoff uses a distinct filename.
> This is the current **MOE-PERF-3 — Async Expert Prefetch for Disk Tier** milestone.

Milestone: **MOE-PERF-3** — overlap the disk-tier expert reads of a token's selected experts
so the NVMe read latency is hidden under the existing thread pool, **without changing
numerics, routing, MLA, attention, generation, certification thresholds, manifests, or
ADRs**. Attacks the bottleneck MOE-PERF-2-VALIDATION identified: expert re-read **latency**
at `cache cap=1`, which neither bf16 nor auto-size can remove (they need more RAM).

## PHASE 1 — Access pattern (audited)

In `ResidentExpertLayer::forward_cached`, a token's selected routed-expert IDs are known
**immediately after routing, before the FFN loop** (`top_k_routing_with(...).indices`). The
old loop resolved them **one at a time** (`resolve_cached`) — each disk-tier miss a serial
NVMe read. The **next layer's** experts depend on the current layer's output (a hard
sequential chain), so next-layer prefetch is **not** possible without speculation. The only
safe overlap window is **within a forward**: read the token's selected experts concurrently.

## PHASE 2 — Design (chosen: option D, lowest risk)

**Parallel batch-resolve of the selected experts before the FFN loop**, via the existing
**rayon** pool (no async runtime, no uncontrolled threads):

- **Bit-exact + deterministic:** resolved weights are pure functions of the tier
  (order-independent); the FFN loop combines in the **same** `selection.indices` order
  regardless. Output is identical to the serial path.
- **Bounded:** at most `top_k` experts transiently — the forward already held one at a time;
  the cache LRU bound is unchanged.
- **Works at cap=1:** the parallel read does not depend on the LRU (it feeds the FFN loop
  directly); cache reuse across forwards is preserved.
- **Opt-in + safe fallback:** `ATENIA_MOE_PREFETCH=1` (default **off**). When off, the exact
  old serial path runs. Resolve errors propagate as before.

Rejected: (A) prefetch-into-LRU before the loop — at cap=1 the LRU evicts → re-reads; (B)
next-layer prefetch — impossible (sequential dependency); (C) background read worker —
uncontrolled threads / async, forbidden.

## PHASE 3 — Implementation

`src/moe/residency.rs` only (the runtime auto-picks it up — `ExpertCache::new` reads the
env, and the productive path already calls `forward_cached`, `graph_op.rs:290`):

- `prefetch_enabled()` (env `ATENIA_MOE_PREFETCH=1`, default off); `ExpertCache.prefetch`
  field + `set_prefetch`/`prefetch_on` (test/A-B hook).
- `ResidentExpertLayer::resolve_selected(cache, indices)` — serves cache hits from RAM,
  **parallel-resolves the misses** (`rayon par_iter`), records stats, inserts into the LRU
  (subject to capacity), returns experts in `indices` order.
- `forward_cached` routed loop: when `cache.prefetch`, batch-resolve via `resolve_selected`
  and run the FFN from the result; else the unchanged serial loop. Shared-expert path
  unchanged.
- `CacheStats` += `parallel_prefetches` and `prefetch_wall_nanos` (`resolve_nanos` keeps
  summing per-expert read time → overlap saved = `resolve_nanos − prefetch_wall_nanos`).

No `runtime.rs` change; no numerics; no routing/MLA/attention/generation change.

## PHASE 4 — Tests (`moe::residency`)

- `perf3_prefetch_is_bit_exact_deterministic_and_bounded` — caps {1,4,8}: prefetch ==
  non-prefetch == uncached forward (bit-exact), deterministic on repeat, `parallel_prefetches
  > 0` (on) / `0` (off), `cache.len() <= cap`.
- `perf3_prefetch_with_shared_expert_is_bit_exact` — cap=1 + shared expert, bit-exact.

## PHASE 5 — Measurement (real)

`tests/moe_perf3_prefetch_bench.rs` (`#[ignore]`), disk-tier, 16 experts, **top-6**, **cap=1**,
16-token decode, per-expert 1.5 MiB:

| Config | decode wall | misses | parallel reads | Σ read | wall read | overlap saved |
|---|---|---|---|---|---|---|
| prefetch **OFF** | **191.74 ms** | 96 | 0 | 116.5 ms | — | — |
| prefetch **ON** | **95.32 ms** | 89 | 89 | 98.8 ms | **22.4 ms** | **76.4 ms** |

**~2× faster decode at cap=1** (191.74 → 95.32 ms). misses ~equal (prefetch changes read
**order**, not count — the cap=1 thrash is irreducible). The win is overlapped NVMe read
latency: the 6 within-forward reads run concurrently (`wall read` 22.4 ms vs `Σ read`
98.8 ms). The gain scales with **read latency** (large real experts) and **top-k**
(DeepSeek top-6 / Qwen top-4 benefit most; Mixtral top-2 less).

> Maps to the MOE-PERF-1 baseline: the Mixtral 402.7 s forward is dominated by serial expert
> re-reads at cap=1; overlapping them (here measured ~2× on top-6) is the lever that helps
> **without** more RAM. The 87 GB real forward was not re-run (heavy); the mechanism speedup
> is measured on a representative disk-tier layer.

## PHASE 6 — Validation

- `moe::residency`: **20 tests pass** (2 new PERF-3).
- **No certification regression:** `moe_scale_cert_test` (Mixtral/Qwen/DeepSeek) **3/3** both
  default **and** with `ATENIA_MOE_PREFETCH=1` (bit-exact, argmax exact, greedy→EOS, det.).
- Full `cargo test --lib` green. No numerical / certification / API regression
  (`forward_cached`/`prefetch`/`ExpertCache::new` signatures unchanged; methods added only).

## Safety model

Default off → certified serial path unchanged. On → only read order/concurrency changes;
experts + math identical → output bit-exact + deterministic (tests + prefetch-on scale-cert).
Existing **rayon** pool (bounded, no new async runtime, no uncontrolled threads). Resolve
errors propagate as before.

## Deliverable summary

1. **Design:** parallel within-forward batch-resolve of selected experts (option D).
2. **Implemented:** `resolve_selected` + opt-in `forward_cached` branch + stats; no runtime change.
3. **Guardrails:** opt-in (`ATENIA_MOE_PREFETCH=1`), default off, bit-exact, deterministic,
   bounded, rayon-only, safe fallback.
4. **Before/after:** decode **191.74 → 95.32 ms (~2×)** at cap=1, top-6 (measured).
5. **Tests:** residency 20/20 (2 new); scale-cert 3/3 default + prefetch-on; full lib green.
6. **Files:** `src/moe/residency.rs`, `tests/moe_perf3_prefetch_bench.rs`,
   `docs/HANDOFF_MOE_PERF_3_PREFETCH.md`, `STATUS.md`, `MOE_PERF_AUDIT.md`.
7. **Roadmap:** PERF-4 remains next (qint8 default tier, gated on a numeric cert).

## Risks / caveats

- The win needs slow reads + top-k > 1; top-2 (Mixtral) overlaps ~2 reads (smaller gain).
- rayon is timing-non-deterministic but **result-deterministic** (output bit-exact); the
  bench wall varies run-to-run (`#[ignore]`, not a CI gate).
- Within-forward overlap only; next-layer prefetch is impossible (sequential dependency).

## Next PERF step

**PERF-4 — qint8 default tier for huge models (gated on a numeric certificate)** — ¼ bytes
on disk → smaller tier, faster cold build, less AV scan; compounds with PERF-2/3. Then PERF-5
(MoE-generate instrumentation parity). MLA latent cache + GPU expert offload deferred (high
risk). **Do not start PERF-4.**
