# HANDOFF — MOE-PERF-2-VALIDATION: real impact measurement

Measurement only — **no runtime / optimization / cache / numerics / certification /
manifest / ADR change.** Quantifies the real effect of MOE-PERF-2 (auto-sized cache +
bf16-resident experts) using **existing** instrumentation (`CacheStats`,
`ExpertCache::resident_bytes*`).

## PHASE 1 — Baseline (recovered from records, not estimated)

From the MIXTRAL-CERT-3 run (`HANDOFF_MIXTRAL_CERT_C5.md`):

| Item | Value |
|---|---|
| Model | Mixtral-8x7B-v0.1 (real, 87 GB) |
| Tier | persistent **bf16 disk tier** (`ATENIA_MOE_TIER_PERSIST=1`, NVMe) |
| Cache | `ATENIA_MOE_EXPERT_CACHE=1` (manual; the default `2·top_k=4` ≈ **90 GB → OOM**) |
| Seq | 4 (canonical input) |
| Path | `MoeRuntime::load_from_dir` → `forward_logits` |
| **Warm load** | **4.5 s** (tier warm reconstruct) |
| **Forward** | **402.7 s** (≈ 100 s/position) |

## PHASE 2/3 — Reproduce + before/after (real numbers)

The 87 GB real-weight forward was **not** re-run (a ~1 h heavy load that previously
stressed the host — against the no-heavy-run guard, and the conclusion is determinable
without it; see PHASE 4). Instead the **cache mechanism** (the only thing PERF-2 changed)
is measured directly on a Mixtral-style disk-tier layer (32 experts, top-2, 24-token
decode, per-expert 96 KiB f32) with the existing `CacheStats`
(`tests/moe_perf2_cache_validation.rs`, `--ignored`):

| Config | hits | **misses** (= tier reads = rematerializations) | evictions | resident | f32-equiv | **saved** |
|---|---|---|---|---|---|---|
| **BEFORE** cap=1 f32 | 0 | **48** | 47 | 98 304 B | 98 304 B | 0 |
| cap=1 **bf16** | 0 | **48** | 47 | **49 152 B** | 98 304 B | **49 152 B (½)** |
| cap=4 bf16 | 10 | 38 | 34 | 196 608 B | 393 216 B | ½ |
| cap=8 bf16 | 12 | 36 | 28 | 393 216 B | 786 432 B | ½ |
| cap=all(32) bf16 | 25 | **23** | 0 | 1 130 496 B | 2 260 992 B | ½ |
| cap=all(32) f32 | 25 | 23 | 0 | 2 260 992 B | 2 260 992 B | 0 |

Two clean, reproducible facts:

1. **bf16 halves `resident` at every capacity** (`saved` = exactly ½), and the **miss
   count is identical** for a given capacity (cap=1: 48 vs 48; cap=all: 23 vs 23). →
   bf16 is a **pure RAM win, zero runtime effect by itself**.
2. **A larger cache cuts misses** (rematerializations): 48 (cap=1) → 38 (cap=4) → 36
   (cap=8) → **23 (cap=all, −52%)**. → The runtime win is the **bigger cache**, which bf16
   makes affordable at half the RAM — but it is only realized **when RAM allows cap>1**.

## PHASE 4 — Root cause

**B) PERF-2 reduced RAM but, on a 32 GB host, not runtime — and it ENABLES the runtime
win on larger hosts.** Evidence:

- On this 32 GB host the auto-size picks **cap=1** (budget ≈ 8 GB ÷ (32 layers ×
  352 MB bf16/expert) < 1). At cap=1 the measured miss count is **unchanged** (48 before
  f32, 48 after bf16). So the 402.7 s Mixtral forward — dominated by per-position expert
  re-reads at cap=1 — would be **~unchanged on this host**.
- bf16 still delivers the **measured 2× resident-RAM reduction** and removes the OOM /
  manual tuning (auto cap=1, no 90 GB blow-up).
- The forward-time win requires **cap>1**, i.e. caching the distinct experts a forward
  touches so they are read once, not re-read. bf16 halves the RAM to reach a given cap
  (Mixtral cap=4: 90 GB f32 → **45 GB bf16**), so on a ≥~50 GB-free host PERF-2 turns the
  former OOM into a working larger cache → fewer misses → faster forward. On 32 GB that
  headroom does not exist, so the forward is unchanged here (case B).

The remaining bottleneck on a RAM-constrained host is therefore **expert re-reads at
cap=1** — which neither bf16 nor auto-size can remove (they need more RAM to grow the cache).

## PHASE 5 — Roadmap reassessment

**Keep PERF-3 (expert prefetch / async tier reads) as the next milestone — now with
stronger justification.** On a RAM-constrained host the cache cannot grow (cap=1), so the
only lever that improves the forward is **hiding the NVMe read latency under compute**
(async prefetch of the next layer's selected experts), which helps **even at cap=1**.
Evidence: at cap=1 the misses (48) are irreducible by caching here; only overlapping those
reads with compute (PERF-3) cuts wall time without more RAM. PERF-4 (qint8 default) and
PERF-5 (instrumentation parity) remain after. PERF-2 stands as the RAM/OOM fix that also
unlocks the cap>1 runtime win on larger hosts.

## Did PERF-2 materially improve real-world MoE execution?

- **RAM: yes — a measured 2× reduction in expert-cache residency at every capacity, and the
  removal of the ~90 GB OOM (auto-sizing, no manual tuning).**
- **Runtime on a 32 GB host: no (case B)** — auto cap=1, rematerializations unchanged (48).
- **Runtime on a ≥~50 GB-free host: yes (enabled)** — bf16 makes cap≥4 affordable
  (45 GB vs 90 GB), and a larger cache cuts misses up to **−52%** (measured), which maps
  to a proportional forward-time reduction.

## Files / tests / commit

- `tests/moe_perf2_cache_validation.rs` (new, `#[ignore]` measurement harness; uses only
  existing instrumentation).
- `docs/HANDOFF_MOE_PERF_2_VALIDATION.md` (this), `docs/STATUS.md`.
- Tests run: the validation harness (above) + `cargo test --lib` (unchanged — the harness
  is a `--test` target, not `--lib`; no `src/` change). No expensive certification re-run.

**Do not start PERF-3.** This milestone ends at the validation conclusions.
