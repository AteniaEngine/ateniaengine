# HANDOFF — MIXTRAL-CERT-3 (C5 active-path): Mixtral-8x7B-v0.1 → MoE-certified L3

Milestone: **MIXTRAL-CERT-3 / C5 → L3** — certify **C5 (active-path)** on the **real
Mixtral-8x7B-v0.1** weights: Atenia's real full forward vs an external float64
reference over the **active subgraph**, reaching **L3 (active-path-certified)**.
Predecessor: MIXTRAL-CERT-2 (C4 fold → L2). **No runtime / loader / MoeRuntime /
numerics / Adapter-Toolkit change** — a reference generator + a test-only harness that
only *calls* the runtime.

## Result

**C5 PASS on the real weights → Mixtral-8x7B-v0.1 is MoE-certified L3 (active-path-certified).**

| Metric | Value |
|---|---|
| Methodology | Atenia real full forward (GQA attn + RoPE + top-2 MoE, all 32 layers, **disk expert-tier**) vs **float64** reference computed **one decoder layer at a time** (HF attention in f64 + manual MoE **one active expert at a time** in f64) — the F64 form of C5 over the **active subgraph**, never the whole model in F64 |
| Canonical input | `input_ids = [1, 100, 200, 300]` (seq 4) |
| Gate | end-to-end `max_abs_diff < 0.5` (ADR-004 bar, **unchanged**) |
| **Worst `max_abs_diff`** | **3.185e-4** at **position 0** (~1570× inside the gate) |
| **Argmax** | **exact 4/4** — `[422, 327, 160, 327]`, zero mismatches |
| **Determinism** | **yes** — two consecutive Atenia forwards bit-identical (`assert_eq`) |
| Timing | load (warm tier reconstruct) 4.5 s; forward 402.7 s; test wall 955.74 s |
| Reference driver validation | 2.021e-8 vs HF f64 on the `mixtral_scale` tiny fixture, before use |

→ **Mixtral-8x7B-v0.1: MoE-certified L3 = L2 + C5 = active-path-certified.** **Not** the
dense ADR-004 `CERTIFIED`; **not L4** (global F64 ~374 GB, reserved/unreachable).

## Methodology — why this is L3 and not L4

A global `model.double()` F64 forward of Mixtral-8x7B is ~374 GB and infeasible. ADR-007's
**primary** C5 method is **F64 over the active subgraph**: the reference computes, **one
decoder layer at a time**, HF's own `MixtralAttention` in float64 (q/k/v/o only) for
attention, then the MoE **one active expert at a time** in float64 (router softmax in
float32 → top-2 → renorm → f64; expert = `w2(silu(w1·x) * (w3·x))`). Peak RAM ~2 GB — the
whole model is **never** materialised in F64, so this is **L3, not L4**. The reference
driver is validated against the tiny `mixtral_scale` fixture (2.021e-8 vs HF f64) before
being trusted on the real weights.

## Tooling

- **Reference generator** (`fixtures/moe/generate_mixtral_c5_reference.py`): modes
  `tiny` (validate the driver vs `mixtral_scale` HF f64) and `real <dir> <out>`. Reads
  the real bf16 weights → float64, **one layer at a time, one active expert at a time**;
  emits `fixtures/moe/mixtral_c5_ref.{safetensors,json}` (logits + `argmax_per_position`
  + metadata). Handles both expert layouts (classic `block_sparse_moe.experts.{e}.{w1,w3,w2}`
  / packed `mlp.experts.gate_up_proj`+`down_proj`).
- **Harness** (`tests/moe_mixtral_c5_active_path_test.rs`, `mixtral_real_c5_active_path`,
  `#[ignore]` + env `MIXTRAL_DIR`): `MoeRuntime::load_from_dir` → `forward_logits` →
  gates worst `max_abs_diff < 0.5` **+** exact per-position argmax **+** determinism. If
  any gate fails it asserts loudly and does **not** certify L3.

## Resumability & the reaping/memory story (operational, important)

- **Reference**: per-layer **atomic** hidden-state checkpoints
  (`fixtures/moe/.mixtral_c5_ckpt/`, write-tmp + `os.replace`); a reaped run loses at most
  the in-flight layer, a re-run skips done layers. Completed across windows (32/32).
- **Model on SSD + persistent disk tier**: the checkpoint was copied to an NVMe SSD and a
  **persistent bf16 disk expert-tier** built once (`ATENIA_MOE_TIER_PERSIST=1`,
  `ATENIA_DISK_TIER_DIR=<nvme>`, ~88 GB). After that, Atenia's load is a **~4.5 s warm
  reconstruct** (no shard re-read) — so harness re-runs are cheap and resumable.
- **Memory gate (the one real surprise — diagnosed, not worked around):** on a 32 GB host
  the **default** per-layer expert cache of `(2·top_k)=4` reconstructed-**F32** experts ×
  32 layers commits **~90 GB** and exceeds the system commit limit → the forward aborts
  with an OOM (`memory allocation … failed`). This is **not** a numeric FAIL. Fix is a
  **test-only env knob, numerically identical** (caching is a perf optimisation only):
  **`ATENIA_MOE_EXPERT_CACHE=1`** → peak ~29 GB (22 GB experts + ~6.5 GB backend), fits.
  The PASS above was produced with `ATENIA_MOE_EXPERT_CACHE=1`, Idle priority, 4-core
  affinity, with a memory-guard watcher. No threshold lowered; the f64 reference and the
  `< 0.5` gate are unchanged.

## Reproduce

```
python fixtures/moe/generate_mixtral_c5_reference.py real models/Mixtral-8x7B-v0.1 fixtures/moe
$env:MIXTRAL_DIR        = "models/Mixtral-8x7B-v0.1"
$env:ATENIA_MOE_EXPERT_TIER = "disk"; $env:ATENIA_MOE_TIER_PERSIST = "1"
$env:ATENIA_DISK_TIER_DIR   = "<nvme>\mixtral_tier"
$env:ATENIA_MOE_EXPERT_CACHE = "1"     # 32 GB host: bound the cache (numerically identical)
cargo test --release --test moe_mixtral_c5_active_path_test -- --ignored mixtral_real_c5_active_path --nocapture
# expect: MIXTRAL-C5 RESULT ... C5 PASS, worst 3.185e-4 < 0.5, argmax exact 4/4, deterministic -> L3
```

## Regression / scope

**No `src/` change.** The harness only *calls* `MoeRuntime::load_from_dir` +
`forward_logits` (the certified runtime). It is `#[ignore]` and not run by CI
(`cargo test --lib` only); the ADR-004 gate is not lowered. Checkpoint/tier dirs are
git-ignored scratch; weights are git-ignored. Qwen/DeepSeek and the Adapter Toolkit were
not touched.

## Files

- `fixtures/moe/generate_mixtral_c5_reference.py` (new, resumable, active-subgraph f64).
- `fixtures/moe/mixtral_c5_ref.{safetensors,json}` (new, the committed C5 reference).
- `tests/moe_mixtral_c5_active_path_test.rs` (new, `#[ignore]`).
- `docs/numcert/mixtral-8x7b-v0.1.moecert.json` (updated → L3, C5 PASS).
- `docs/STATUS.md`, `docs/MODEL_FAMILY_VALIDATION.md`, `docs/FAMILY_COVERAGE_AUDIT.md`
  (updated → L3), this handoff.

## Remaining risks / caveats

- C5 is certified over the **active subgraph** in f64 (the experts/positions exercised by
  the canonical probe), not a global f64 forward — ADR-007's primary C5 method; **L3, not L4**.
- Single canonical probe (same standard as Qwen/DeepSeek C5).
- C3 (GQA attention) + C4 (topology) remain at the **mechanism** level (reduced-dim fixture),
  not real-weight per-layer; L3 carries that caveat. Not dense ADR-004 `CERTIFIED`.
- **L4** (global `model.double()` F64, ~374 GB) reserved / unreachable; never fabricated.
