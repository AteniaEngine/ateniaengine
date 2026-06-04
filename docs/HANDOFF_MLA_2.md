# HANDOFF — MLA-2: disk/bf16 expert-tier for the experimental DeepSeek MLA forward

Milestone: **MLA-2** — let the experimental DeepSeek (`Backend::Mla`) forward
**stream its experts from a residency tier** (RAM default, NVMe via
`ATENIA_MOE_EXPERT_TIER=disk`) instead of holding every layer's experts in RAM as
f32. This unblocks the C5 active-path forward of the **real DeepSeek-V2-Lite** on a
32 GB host. Predecessor + design: `docs/MLA_2_DISK_TIER_AUDIT.md` (the approved
audit). **C5 was NOT run; DeepSeek-V2-Lite stays MoE-certified L2.**

## Root cause (recap)

C5 needs Atenia's real full forward on DeepSeek-V2-Lite. The MLA forward consumed a
RAM-f32 `RealMoeLayer` per layer (`DeepseekFfn::Moe(RealMoeLayer)`,
`mla.rs::layer_step`), and `runtime.rs` hard-pinned DeepSeek to `ExpertTier::Ram`.
One f32 copy of the real model is ~58.5 GB on a 31.7 GB host → infeasible. The block
was **purely residency** — not numerics, model, YaRN, or dense-first. The certified
disk/bf16 tier (`ResidentExpertLayer` + `disk_tier` + `ExpertCache`, used by Qwen)
already existed; it was simply not wired into the `Backend::Mla` branch.

## What changed (minimal, isolated to the DeepSeek/MLA path)

- **`src/moe/mla.rs`**
  - `DeepseekFfn::Moe(RealMoeLayer)` → `DeepseekFfn::Moe(Arc<ResidentExpertLayer>)`.
  - `layer_step` FFN call: `m.forward_with(&h2, cfg.moe_convention())` →
    `m.forward(&h2).map(|(y,_)| y)` — the **uncached** residency forward (no cache,
    no perf work). The residency layer carries the convention internally.
  - Removed the now-unused `DeepseekConfig::moe_convention()` and the
    `MoeExecutionConvention`/`RealMoeLayer` imports; `renorm_topk` stays as an
    informational config field.
- **`src/moe/runtime.rs`**
  - `expert_tier` no longer forced to `Ram` for DeepSeek — honours
    `expert_tier_from_env()` like Qwen/Mixtral.
  - New `deepseek_convention` computed from `norm_topk_prob` (default `true` →
    `Atenia`; `false` → `HuggingFaceQwen`). **Why:** DeepSeek's shared expert is
    *ungated*, so `RealMoeLayer::resolve_convention()` (which keys off a shared-gate)
    would mis-classify V2-Lite (`norm_topk_prob=false`) as `Atenia`. We set the
    residency layer's `convention` explicitly to reproduce the pre-MLA-2
    `forward_with(_, cfg.moe_convention())` behaviour bit-for-bit.
  - DeepSeek assembly branch: build `ResidentExpertLayer::from_real_layer(&moe,
    expert_tier)`, set `resident.convention = deepseek_convention`, self-validate,
    then **drop the f32 `RealMoeLayer`** before the next layer. The resident `Arc`
    feeds `build_deepseek` (signature now `Vec<Arc<ResidentExpertLayer>>`).
  - `self_validate_residency` now compares the resident's `forward_cached` against
    `moe.forward_with(&probe, resident.convention)` (was `forward_auto`). For the
    graph families `resident.convention == resolve_convention()`, so this is
    **identical** to before (Qwen/Mixtral unchanged); for DeepSeek it validates
    under the correct convention.
- **`tests/moe_mla0_deepseek_v2lite_test.rs`** — two MLA-2 parity tests +
  `load_disk_tier()` helper.

No latent cache, no Q-LoRA, no DeepSeek-V3, no Adapter Toolkit, no numerics change,
no Qwen/Mixtral logic change, no threshold change.

## Memory

- **Before (RAM tier, default):** experts + backend in RAM-f32 ≈ **58.5 GB** for the
  real model → does not fit 31.7 GB.
- **After, `ATENIA_MOE_EXPERT_TIER=disk`:** experts on NVMe at ~0 host RAM; RAM holds
  the backend (embed/lm_head/MLA-attention/dense-FFN ≈ 3.4 GB) + one layer's
  transient resolved experts (~0.3 GB) → peak **≈ 4 GB**.
- **NVMe footprint:** the minimal path uses the **ephemeral** disk tier, which writes
  experts as **f32** (UUID-named, deleted on drop) → ~59 GB on NVMe (`F:` has 726 GB;
  point `ATENIA_DISK_TIER_DIR` at it). **bf16-on-disk** (~30 GB, lossless for
  bf16-source) is available only via the *persistent* tier (`from_real_layer_at` +
  `Bf16Auto`), which stays disabled for DeepSeek — a later optimisation, out of MLA-2
  scope. RAM is ~4 GB either way (disk dtype does not affect host RAM).
- **Default (no env):** DeepSeek stays `Ram` — bit-identical to pre-MLA-2.

## Parity / tests (measured)

| Check | Result |
|---|---|
| MLA-0 full forward, **disk tier == RAM tier** | **bit-identical** (`assert_eq!`) |
| MLA-0 disk-tier vs HF f64 | **9.072e-5** (== RAM), argmax exact, `< 0.5` gate |
| MLA-0 greedy/EOS, disk vs RAM | **identical** + deterministic |
| `self_validate_residency` (load-time, every DeepSeek layer) | PASS under the V2-Lite convention |
| `moe_mla0_deepseek_v2lite_test` (5 tests) | **5 passed** |
| DeepSeek/Qwen/Mixtral/scale/residency regressions | green |
| **Full lib suite, single-threaded (CI mirror)** | **838 passed, 0 failed, 1 ignored** |

Commands:
```
cargo test --release --test moe_mla0_deepseek_v2lite_test -- --nocapture
cargo test --release --lib -- --test-threads=1
```

Notes:
- A parallel-only flake (`gpu::safety::resource_check::…legacy_probe_…`, a live
  free-VRAM probe drifting ~8 MB between two reads) appeared once under the default
  parallel run; it **passes in isolation and single-threaded** (the CI mode). Not
  MLA-2-related (no GPU/moe-runtime code touched).
- A pre-existing non-exhaustive `match` on `DiskDtype` (missing `QInt8`) in
  `tests/m4_7_4_e_tinyllama_disk_spill_smoke_test.rs` fails to **compile** under
  `cargo build --tests`. It predates MLA-2 (the `QInt8` variant came from
  NUMERIC-POLICY-2) and is in a file MLA-2 never touches. **CI is unaffected** — CI
  runs only `cargo test --lib` + `--test tinyllama_config_test`, never that target.

## Status after MLA-2

- DeepSeek-V2-Lite is **MoE-certified L2** (unchanged). **C5 was NOT executed.**
- **L3 is technically unblocked** (the real active-path forward now fits ~4 GB RAM)
  but **NOT certified** — that requires the C5 F64 reference + the real harness run.
- L4 (global F64) remains reserved/unreachable.

## Next (toward L3)

1. Generate the C5 F64 reference for DeepSeek-V2-Lite (python, one decoder layer at
   a time, `gc` between layers — peak ~one F64 layer, never the whole model → not L4).
   Validate the driver against the tiny MLA-0 fixture first.
2. Add the real C5 harness (`#[ignore]` + `DEEPSEEK_V2_LITE_DIR`), mirroring
   `moe_cert4_qwen_active_path_test.rs`, run under `ATENIA_MOE_EXPERT_TIER=disk` +
   `ATENIA_DISK_TIER_DIR=F:\…`. Gate: end-to-end `max_abs_diff < 0.5` (ADR-004,
   unchanged) + exact per-position argmax + deterministic.
3. On PASS → fold C5 into the manifest → **MoE-certified L3**. On fail → do NOT
   certify, document, STOP.
