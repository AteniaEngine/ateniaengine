# HANDOFF — MOE-14: opt-in real local MoE smoke harness

Milestone: **MOE-14** (point the MOE-13 validation pipeline at a *real* MoE
checkpoint sitting in a local directory, behind an opt-in, `#[ignore]`d,
out-of-CI smoke test). Correctness-first, CPU-only, experimental. The MOE-2
fail-loud guard is **unchanged**. No automatic download, no model in the repo,
nothing new in CI. No CUDA, ROCm, Metal, tier-planner, CLI, generation,
Adapter Toolkit, batching, or optimisation. Predecessor: MOE-0..13 (`d334d6f`).

## How to run the real smoke

```bash
ATENIA_MOE_REAL_MODEL=/path/to/qwen-moe \
  cargo test --release --test moe_real_model_smoke_test -- --ignored --nocapture
```

- **Env var**: `ATENIA_MOE_REAL_MODEL` — absolute path to a local model
  directory. If unset, the ignored test prints `real smoke not run:
  ATENIA_MOE_REAL_MODEL not set` and returns (no panic, no failure).
- **Directory layout expected**: one or more `*.safetensors` files (shards
  like `model-00001-of-00003.safetensors` are supported and read in sorted
  order); optionally a `config.json`.
- The test is **model-agnostic** — no model is hardcoded.

### Expected first targets (documented, not hardcoded)

- Qwen1.5-MoE-A2.7B
- small Qwen-MoE checkpoints
- Mixtral only if the host has enough RAM

## What the smoke validates

1. **Discovery** — `discover_safetensors_files(dir)`: lists `.safetensors`
   (incl. shards), sorted; errors loud on missing dir / no files.
2. **Minimal config** — `MinimalMoeConfig::from_dir(dir)`: reads only
   `num_hidden_layers`, experts (`num_experts` / `num_local_experts`), top-k
   (`num_experts_per_tok` / `num_experts_per_token`), `hidden_size`,
   `intermediate_size`. Absent file or fields → caller falls back
   (`experts_per_token` default = 2). **Not** a full config parser.
3. **Open + merge** — `LocalMoeCheckpoint::open(files)` opens one
   `SafetensorsReader` per shard; `weight_map()` merges every shard's
   `(name, shape)` into one `MoeWeightMap`; `resolve(name)` decodes f32 bytes
   across shards (the MOE-10/13 byte resolver).
4. **Validate** — feeds the merged map + resolver to the MOE-13
   `RealMoeCheckpointValidation`: derive per-layer config from shapes →
   assemble stack → minimal forward → `ValidationReport`.
5. **Report** — prints `layers_detected`, `experts_detected`,
   `shared_experts`, `d_model`, `forward_pass_ok`, and any `errors`.

## What a PASS means / does NOT mean

A smoke **PASS** means: "the experimental MoE path discovered the shards,
built a `MoeWeightMap`, assembled a stack, and ran a *finite* forward over the
real checkpoint." It does **NOT** mean the model is numerically correct,
supported, or production-ready. There is no F64 reference comparison here, no
transformer structure, no multi-token decode. No Mixtral / Qwen-MoE full
support is claimed.

If something is missing (no env var, bad dir, no shards, a tensor's bytes
unresolved), the test prints a clear skip/error and does **not** panic; the
report's `errors` capture the reason and `forward_pass_ok` stays false.

## Why it does not run in CI

The test is `#[ignore]`d and gated on `ATENIA_MOE_REAL_MODEL`. Real MoE
checkpoints (Qwen-MoE ~14B, Mixtral ~47B) are far too large to download or
hold in CI, and must not enter the repo. CI only runs the **lightweight**
unit tests (discovery on temp dirs, tiny JSON config parsing) and a gate test
(`real_smoke_requires_env_var`) that asserts the unset-env contract.

## Fail-loud confirmation

- The **productive loader still fails loud** on MoE checkpoints
  (`LoaderError::MoeUnsupported`) — MOE-14 changes none of that path.
- The smoke uses the **experimental direct path** (`src/moe/` consuming the
  reader's public read API), exactly as MOE-9..13 do. No productive behaviour
  changes.

## Did it run against a real model?

In this development environment `ATENIA_MOE_REAL_MODEL` was **not set**, so the
ignored real smoke was **not executed** (`real smoke not run: ...`). The
harness, discovery, config parsing, and gating are all unit-tested; the real
run is a manual, opt-in step for whoever has a local checkpoint.

## What is still missing for real Mixtral / Qwen-MoE

- An actual real-model run (manual, opt-in, on a host with enough RAM).
- Transformer structure (residuals, norms, attention, embeddings/lm_head,
  multi-token / KV cache).
- A full `config.json` parser + topology cross-check (Adapter Toolkit).
- Shared-expert gating; fail-loud lift behind an explicit opt-in; graph
  registry population.

## What was NOT implemented

- No automatic download, no model committed, no CI execution of the real path.
- No fail-loud lift, no Adapter Toolkit / loader load-path / CLI / generation
  change, no CUDA/ROCm/Metal, tier-planner, batching, optimisation.

## How this prepares MOE-15

With a one-command way to point the pipeline at a real local checkpoint,
MOE-15 can iterate on whatever the first real run surfaces (dtype coverage,
naming edge cases, dimension quirks), then layer in transformer structure and
`config.json`-driven topology — toward a full MoE forward behind a validated,
explicit opt-in that lifts fail-loud only for the certified path.

## Tests

- `src/moe/smoke.rs` — 9 unit tests (discovery: empty/missing/single/shards-
  sorted; config: Qwen fields, Mixtral spellings, missing fields, invalid
  JSON, missing-file).
- `tests/moe_real_model_smoke_test.rs` —
  `moe_real_local_checkpoint_smoke_test` (`#[ignore]`, env-gated, real run)
  and `real_smoke_requires_env_var` (CI-run gate assertion).

Local validation: `cargo test --lib --release -- --test-threads=1` →
**715 passed / 0 failed / 1 ignored** (was 706). Real smoke: not run
(`ATENIA_MOE_REAL_MODEL` unset).

## Files modified

* `src/moe/smoke.rs` — new (discovery + minimal config + local checkpoint + 9 unit tests).
* `src/moe/mod.rs` — re-exports.
* `tests/moe_real_model_smoke_test.rs` — new (ignored real smoke + gate test).
* `docs/HANDOFF_MOE_14.md` — this file.

No loader load-path, Adapter Toolkit, CLI, generation, CUDA, ROCm, Metal,
tier-planner, or graph changes. Fail-loud preserved.

---

## Real Local Smoke Runs (append-only, post-MOE-14)

After MOE-14 landed, the opt-in smoke harness was run **manually, out of CI**
against three real, tiny MoE checkpoints downloaded into a local directory
(no model committed to the repo, no automatic download in CI). These are the
verbatim results.

| Checkpoint | Expert format | Result | Notes |
|---|---|---|---|
| `hf-internal-testing/tiny-random-Qwen2MoeForCausalLM` | packed/fused | **forward INCOMPLETE** | MoE detected (2 layers, shared experts); `config.json` parsed (experts=4, per_tok=2, hidden=64); `experts_detected=0`, `forward_pass_ok=false`, errors: "layer N: no experts" |
| `hf-internal-testing/tiny-random-MixtralForCausalLM` | packed/fused | **forward INCOMPLETE** | same cause; experts stored as packed 3-D tensors the per-expert binding does not recognise |
| `katuni4ka/tiny-random-qwen1.5-moe` | classic per-expert | **SMOKE PASS** | 4 layers, 32 experts, 4 shared experts, `d_model=32`, `forward_pass_ok=true` |

### Expert formats observed

- **Classic per-expert** (works today): one tensor per `(expert, projection)`:
  - `model.layers.{L}.mlp.experts.{E}.gate_proj.weight`  `[d_ff, d_model]`
  - `model.layers.{L}.mlp.experts.{E}.up_proj.weight`    `[d_ff, d_model]`
  - `model.layers.{L}.mlp.experts.{E}.down_proj.weight`  `[d_model, d_ff]`
- **Packed/fused** (detected but NOT yet assembled): all experts stacked into
  3-D tensors, gate+up fused:
  - `model.layers.{L}.mlp.experts.gate_up_proj`  e.g. `[num_experts, 2*d_ff, d_model]`
  - `model.layers.{L}.mlp.experts.down_proj`     e.g. `[num_experts, d_model, d_ff]`
  (observed on a tiny Mixtral as `gate_up_proj [4, 256, 64]`, `down_proj [4, 64, 128]`).

### What this means

- The classic per-expert path works against a **real** checkpoint end-to-end
  (metadata → derived config → stack → finite forward).
- Modern small MoE checkpoints from recent `transformers` versions use the
  **packed/fused** expert representation. Atenia **detects** them as MoE
  (the MOE-2 guard still fires) but **cannot assemble** them yet — the report
  honestly records `forward INCOMPLETE` with a clear "no experts" error and
  does not panic.
- A `SMOKE PASS` means "the experimental MoE path read this checkpoint,
  assembled a stack, and ran a finite forward". It does **NOT** mean full MoE
  model support, numerical certification, or production readiness.
- The **productive loader fail-loud is unchanged** — every one of these
  checkpoints still refuses to load as a model through the normal path.

### Conclusion

**MOE-15 should target packed/fused experts**: extract per-expert
gate/up/down projections from the 3-D `gate_up_proj` / `down_proj` tensors and
feed them into the existing `MoeDenseExpert` / `RealMoeLayer` path, without
disturbing the classic per-expert path that already passes.

> **Update:** the packed-expert limitation was addressed in **MOE-15** — see
> `docs/HANDOFF_MOE_15.md`. All three checkpoints above now SMOKE PASS.
