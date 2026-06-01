# HANDOFF — MOE-PROD-1: Sharded MoE loading

Milestone: **MOE-PROD-1** — unblock loading of **sharded** real MoE checkpoints
(the first of the two RUNTIME-MOE-2 blockers). **Engine** milestone (loader code
changes allowed); **not** a real-weight validation. No model downloaded; no new
architecture / family / math / graph op. Predecessor: `c6c9473`
(RUNTIME-MOE-2 BLOCKED).

## FASE 1 — Audit (what was wrong)

- **MoE loader was single-file.** `MoeRuntime::load_from_files` opened one
  `SafetensorsReader`; `load_from_dir` picked the **first** `*.safetensors` in
  the directory. A sharded checkpoint (real Mixtral = 19 shards, Qwen1.5-MoE =
  8 shards) silently lost all but one shard → assembly failed.
- **Reusable dense infra:** `ShardIndex` (parses `model.safetensors.index.json`,
  `weight_map: name→shard`, `shard_path`) and `SafetensorsReader` (per-file,
  `iter()` + `get(name).to_vec_f32()`, lossless BF16→F32). The dense
  `ShardedSafetensorsReader` is bulk/streaming (no by-name random access), so it
  is **not** directly reusable for the MoE path's `get(name)` pattern.
- **Why it failed with shards:** only one shard was read; the other tensors were
  absent → "missing tensor" / family-recognition failures.
- **Why it explodes RAM (separate blocker):** the compute backend
  (`TinyMixtralWeights` / `DeepseekWeights`) holds **every tensor as `Vec<f32>`
  in RAM** and the graph path even `clone()`s them per forward. Qwen1.5-MoE-A2.7B
  (14.3 B) ≈ **57 GB f32** → exceeds 32 GB RAM. The `ExpertTier::Disk` residency
  substrate exists but is a **parallel** structure the compute backend does not
  use.
- **Minimal change that unblocks the *sharded* dimension:** a weight-source
  abstraction that exposes `(name, shape)` metadata and a by-name `f32`
  resolver, backed by either one file or many shards. The rest of the loader is
  unchanged.

## FASE 2-5 — What was implemented

### Sharded loading (FASE 4) — done

New **`MoeWeightSource`** enum in `src/moe/runtime.rs`:

```text
enum MoeWeightSource {
    Single(SafetensorsReader),
    Sharded { index: ShardIndex, cache: RefCell<Option<(String, SafetensorsReader)>> },
}
```

- `open_dir(dir)` → **Sharded** when `model.safetensors.index.json` is present,
  else **Single** (first `*.safetensors`) — back-compat with MOE-FULL-14.
- `open_file(path)` → Single.
- `tensor_metas()` → `(name, shape)` across all shards (drives family
  classification, config cross-check, `MixtralAdapter::recognize`, `MoeWeightMap`).
- `get_f32(name)` → by-name `Vec<f32>`; the sharded arm resolves the tensor's
  shard via `weight_map`, decoding through the **same** `TensorEntry::to_vec_f32`
  as single-file (so bytes decode identically). A **single-shard cache**
  (`RefCell`) avoids re-reading a multi-GB shard per tensor — consecutive
  by-name lookups are layer-ordered and shard-local, so peak loader RAM is
  ~one shard, not all shards.

`load_from_files` and `load_from_dir` now build a `MoeWeightSource` and call a
shared **`load_core(config_text, source)`**; `build_graph` / `build_deepseek`
take `&MoeWeightSource` instead of `&SafetensorsReader`. The opt-in gate fires
**first**, before any filesystem I/O, in both public entry points (fail-loud
contract preserved — see "problem found").

**Result:** sharded MoE checkpoints load, and a sharded load is **bit-for-bit
identical** to the equivalent single-file load (proven by test).

### RAM footprint / residency (FASE 3 + 5) — STOP, documented

Reducing the f32-in-RAM footprint is **not** done here, by design (FASE 3:
"if it implies a big refactor, STOP and report"). The compute backend
(`TinyMixtralWeights` / `DeepseekWeights`) materialises **all** weights as f32
and the graph path clones them per forward; the certified `full_forward` /
MLA compute consumes f32 buffers. Making the backend bf16- or disk-backed means
refactoring the **certified compute path** — a large, risky change outside this
milestone. The `ExpertTier::Disk` substrate exists but is parallel to compute,
so flipping the residency tier alone would not reduce the backend footprint.

**Honest status:** sharding is solved; the **57 GB f32 RAM** ceiling for
Qwen1.5-MoE is **not**. Both are required to load it on a 32 GB host.

## FASE 6-7 — Tests

New `tests/moe_sharded_loader_test.rs` (5 tests), fixtures built **at test
time** by splitting the committed tiny `full_mixtral.safetensors` into two
shards + an index (no download, no committed large fixture):

| Test | What it proves |
|---|---|
| `sharded_equals_single_file_logits_and_generation` | **bit-exact**: sharded logits `max_abs_diff == 0.0` vs single-file; identical generation |
| `single_file_directory_without_index_still_loads` | back-compat: a dir with one `.safetensors` and no index still loads → generates to EOS |
| `missing_shard_file_errors_clearly` | `weight_map` → nonexistent shard → clear `Load` error |
| `missing_tensor_errors_clearly` | a required tensor absent from all shards → clear failure |
| `corrupt_index_errors_clearly` | invalid `index.json` → clear `Load` error mentioning the index |

FASE 7 (real-metadata smoke without download): the bit-exact split test
exercises the exact sharded layout (`model-00001-of-00002.safetensors` +
`model.safetensors.index.json` + `weight_map`) a real Qwen1.5-MoE uses; family
recognition + config cross-check run over the multi-shard metadata. Real Qwen
shards were **not** downloaded.

## FASE 9 — Final validation (real, exit 0)

- `tests/moe_sharded_loader_test.rs` — **5/5 pass** (incl. bit-exact).
- MoE regression: `moe_mixtral_runtime_test` 3/3, `moe_qwen_runtime_test` 3/3,
  `moe_deepseek_runtime_test` 4/4 (MLA/`build_deepseek`), `moe_family_loader_test`
  3/3, `moe_loader_failloud_test` 3/3, `cli_moe_generate_test` 2/2,
  `moe_production_test` 4/4, `moe_runtime_robustness_test` 6/6,
  `moe_scale_cert_test` 3/3, `moe_partial_cert_test` 4/4.
- Full lib suite: see commit/CI (run single-threaded).

## Problem found + fix (self-introduced, caught by tests)

Moving the opt-in gate into `load_core` initially placed it **after** the
config read / file open, so `opt_in_disabled_refuses` (which passes nonexistent
paths expecting `OptInDisabled`) failed with a `Config` error instead. **Fix:**
restored the opt-in gate as the **first** statement of both `load_from_files`
and `load_from_dir`, before any filesystem I/O — preserving the fail-loud
contract ("refuse before touching the disk when the flag is unset"). Re-validated
green. No external STOP needed (self-introduced, caught and fixed in-milestone).

## What still blocks RUNTIME-MOE-2 (Qwen1.5-MoE real)

1. **f32 RAM footprint** — the remaining blocker. Qwen1.5-MoE-A2.7B ≈ 57 GB f32
   in the compute backend > 32 GB RAM. Needs bf16/disk-backed compute weights
   (a certified-compute refactor) **or** a ≥64 GB RAM host.
2. **Download** — 28.6 GB real weights (disk OK on the 830 GB NVMe), only after
   (1) so the load can actually complete.

Sharding (this milestone) is necessary but **not sufficient** on its own for the
32 GB host.

## Files modified

- `src/moe/runtime.rs` — `MoeWeightSource` (single/sharded), `load_core`,
  sharded `load_from_dir`, `build_graph`/`build_deepseek` take the source;
  opt-in-gate-first preserved.
- `tests/moe_sharded_loader_test.rs` — new (5 tests).
- `docs/HANDOFF_MOE_PROD_1.md` — this file; `docs/STATUS.md`,
  `docs/MOE_OVERVIEW.md`, `docs/MOE_FULL_PATH_AUDIT.md` — updated.

No new architecture / family / math / graph ops. Single-file path, dense path,
and all certified MoE outputs unchanged (bit-exact).

## Deliverable answers

1. **What implemented?** `MoeWeightSource` (single + sharded) + `load_core`
   refactor + sharded `load_from_dir`; 5 tests incl. bit-exact sharded==single.
2. **Sharded supported now?** **Yes.** Sharded checkpoints load, bit-identical
   to single-file.
3. **f32 RAM reduced?** **No** (STOP-documented) — compute backend holds f32;
   reducing it is a certified-compute refactor, out of scope.
4. **What's missing for Qwen1.5-MoE real?** The f32-RAM/residency reduction
   (still blocking on 32 GB) + the 28.6 GB download.
5. **Tests run?** sharded 5/5 + MoE regression (all green) + full lib suite.
6. **Risks?** Sharded `get_f32` re-opens a shard on shard-boundary crossing
   (slow but correct; correctness-first). RefCell single-shard cache is not
   thread-shared (loads are single-threaded). No effect on certified outputs.
7. **Files modified:** see above.
8. **Commit:** see git log (MOE-PROD: add sharded MoE loading support).
9. **CI:** code change → CI runs (not docs-only); see push result.
10. **When to reopen RUNTIME-MOE-2?** After a follow-up engine milestone adds
    bf16/disk-backed compute residency (so 28.6 GB fits under 32 GB RAM), or on
    a ≥64 GB RAM host — then download Qwen1.5-MoE-A2.7B-Chat and validate.

## Recommendation

Sharding is unblocked. The **next** engine milestone should tackle the f32-RAM
ceiling — bf16-resident or disk-backed compute weights in the MoE backend — as
that is now the sole remaining blocker for real small-MoE validation. Only then
reopen RUNTIME-MOE-2 with the real Qwen1.5-MoE-A2.7B-Chat download.
