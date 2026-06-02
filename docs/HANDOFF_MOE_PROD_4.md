# HANDOFF — MOE-PROD-4: persistent expert tier (scope A)

Milestone: **MOE-PROD-4** — stop regenerating the disk tier on every run; reuse a
valid existing tier. MOE-PROD-3 found that **load** dominates (the ~50 GB NVMe
tier write + reads), not generation. Scope **A** (user-approved): deterministic
tier names + manifest + **skip the write** when a valid tier exists — bit-exact
by construction, low risk. Predecessor: `76f3f32`.

## FASE 1 — Audit

- **How the tier is generated:** load_core (disk) → per layer
  `ResidentExpertLayer::from_real_layer(&moe, Disk)` → `place(Disk)` →
  `disk_tier::write_f32_tensor(dir, data)` → **`tensor_<uuid>.bin`** (random
  name). Worse: `DiskTensorHandle::Drop` **deletes** the file on drop — the tier
  is ephemeral by design. Every run re-reads the shards, re-assembles f32
  experts, and re-writes ~50 GB.
- **What identifies a tier today:** nothing (random UUIDs, no manifest).
- **Risk:** the write is coupled to assembly in the certified residency path;
  threading identity + persistence must not change the bytes (it doesn't — same
  f32 bytes whether written or reused).

## FASE 2-3 — Design + implementation (scope A)

- `disk_tier.rs`: `InnerDiskFile` gains a `persistent` flag — when set, `Drop`
  does **not** delete the file. New `write_f32_tensor_named(path, data)`
  (deterministic, persistent) and `open_existing_f32(path, numel)` (wrap an
  existing file without writing). Ephemeral UUID writes keep `persistent=false`
  (unchanged).
- `residency.rs`: `TierContext { dir, entries, reused, written }`; `place_at`
  writes to `dir/<key>.bin`, **reusing** the file if it exists with the expected
  byte length (else writing); `ResidentExpertLayer::from_real_layer_at(layer,
  ctx, layer_id)` tiers experts under deterministic keys `L{l}.e{i}.{gate,up,
  down}` / `L{l}.shared.*`. Router + shared-gate stay RAM (re-read each load).
- `runtime.rs`: `ATENIA_MOE_TIER_PERSIST=1` opts in. `compute_model_id` =
  stable hash of config + **sorted** (name, shape) metadata (sorted because
  `tensor_metas()` iteration order is non-deterministic — see the bug below).
  Tier dir = `<base>/moe_tier/<model_id>/`. After the load, a `TierManifest`
  (version, model_id, family, counts, total_bytes, per-tensor entries) is
  written; `ATENIA_MOE_CACHE_STATS=1` reports `reused/written`.
- **Default off** → ephemeral UUID tier, byte-identical to MOE-PROD-2/3. DeepSeek
  (RAM) and the RAM tier are untouched.

## FASE 4-5 — Validation + integrity (tiny fixture, fast)

`tests/moe_tier_persist_test.rs` (new):
1. First load writes the tier + `tier_manifest.json`; the `.bin` files **survive
   the runtime being dropped** (persistent handles).
2. Second load **reuses** them — `.bin` mtimes **unchanged** (no rewrite) — and
   generation is **identical** (bit-exact).
3. Integrity: deleting a tier file makes the next load **regenerate** it (size
   mismatch / missing → rewrite), still bit-exact.

Plus: full MoE regression (residency-tier disk==RAM, expert-cache, mixtral, qwen,
deepseek, sharded, production, robustness, cli) and `disk_tier` unit tests
(23/23) — all green with the default (ephemeral) path unchanged.

## Problem found + fix (self-introduced, caught by the test)

First test run: the tier was rewritten every load (3 different `model_id`s in one
process). Root cause: `compute_model_id` hashed the `(name, shape)` metadata in
**`tensor_metas()` iteration order**, which is HashMap-backed and
non-deterministic. **Fix:** sort by tensor name before hashing → stable id →
stable tier dir → reuse works. Caught and fixed in-milestone.

## FASE 6 — Benchmark (real Qwen1.5-MoE-A2.7B, cold vs warm, max-new 2)

| | Cold (writes tier) | Warm (reuses tier) |
|---|---|---|
| Total wall (load + 2 tokens) | **3757 s** | **2445 s** |
| Tier reused / written | **0 / 4392** | **4392 / 0** |
| Output (token ids) | `374, 6188` | `374, 6188` — **identical (bit-exact)** |
| NVMe tier | ~49.5 GiB written | ~49.5 GiB reused (0 written) |

**Saving: ~1312 s (~22 min, ~35 % of total wall)** — purely from **skipping the
50 GB tier write** on the warm load. Same `model_id` (`24109ed8f03a8b8a`) both
runs → all 4392 expert files reused, output identical. This is the real lever
MOE-PROD-3 pointed at (vs the expert cache's ~0 % gain).

## FASE 7 — Robustness

- First load → writes tier + manifest.
- Second load → reuses (no rewrite), bit-exact.
- Tier file deleted → regenerated next load (per-file size check).
- Partial tier → missing files written, present files reused.
- Different checkpoint → different `model_id` → different dir (no false reuse).
- Default (persist off) → ephemeral UUID tier, unchanged.

## FASE 8 — Review

- **Time saved:** ~1312 s (~22 min, **~35 %**) on the warm load vs cold — the
  full cost of writing 4392 expert files (~49.5 GiB) to NVMe, eliminated.
- **New bottleneck:** the warm load (2445 s) still **reads the shards** (28.6 GB
  metadata + attention) and **assembles the f32 experts** (scope A keeps
  `RealMoeLayer` assembly + the self-validation), plus per-file opens of the
  4392 reused tier files and the generation itself. The tier *write* is gone; the
  tier *read/assembly* is now dominant.
- **Impact:** real and significant — a returning user of the same MoE checkpoint
  loads ~35 % faster, and the 50 GB tier is written **once**, not every run.
  Combined with MOE-PROD-1/2/3 the controlled MoE path now: loads sharded,
  fits RAM via disk tiering, caches routed experts, and **reuses** the tier.
- **Is scope B worth it?** The remaining 2445 s is dominated by shard read + f32
  expert assembly. **Scope B** (reconstruct `ResidentExpertLayer` directly from
  the tier + manifest, skipping shard re-read and f32 assembly entirely) would
  cut most of that — it is now the clear next lever. Recommended as a follow-up.

## Files modified

- `src/tensor/disk_tier.rs` — `persistent` flag + `write_f32_tensor_named` +
  `open_existing_f32`.
- `src/moe/residency.rs` — `TierContext` / `TierEntry`, `place_at`,
  `from_dense_at`, `from_real_layer_at`.
- `src/moe/runtime.rs` — `ATENIA_MOE_TIER_PERSIST`, `compute_model_id` (sorted),
  tier dir + manifest write + reuse wiring.
- `tests/moe_tier_persist_test.rs` — new.
- `docs/HANDOFF_MOE_PROD_4.md` (this) + `docs/STATUS.md`.

No new architecture/family/math/graph ops; outputs bit-exact; default path
unchanged.

## Deliverable answers (filled after benchmark)

1. **What implemented:** opt-in persistent expert tier — deterministic names +
   manifest + skip-write-if-valid + integrity (regenerate on loss).
2. **How the tier is validated:** per-model `model_id` (config+sorted metadata
   hash) selects the tier dir; per-file presence + byte-length decides reuse vs
   rewrite; a `tier_manifest.json` records identity/version/entries.
3. **Time before:** 3757 s cold (load + 2 tokens, writes tier).
4. **Time after:** 2445 s warm (reuses tier).
5. **Real saving:** ~1312 s (~22 min, **~35 %**) — the 50 GB tier write,
   eliminated; output bit-identical.
6. **New bottleneck:** shard read (28.6 GB) + f32 expert assembly + per-file
   opens of the 4392 reused files (scope A still assembles `RealMoeLayer`).
7. **Tests:** persist/reuse/regenerate (new) + full MoE regression + disk_tier
   units — all green; default ephemeral path unchanged.
8. **Commit:** see git log.
9. **CI:** code change → CI runs; see push.
10. **Next:** scope B — reconstruct `ResidentExpertLayer` directly from the tier
    + manifest (skip shard re-read + f32 assembly) to cut the remaining ~2445 s.
