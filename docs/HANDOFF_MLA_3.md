# HANDOFF — MLA-3: YaRN `mscale` fix → DeepSeek-V2-Lite reaches MoE-certified L3

Milestone: **MLA-3** — fix the YaRN attention scaling in the experimental MLA path
(root-caused in `docs/HANDOFF_MLA1_C5_ROOT_CAUSE.md`), then re-run C5 on the real
DeepSeek-V2-Lite. **Result: C5 PASS → DeepSeek-V2-Lite is now MoE-certified L3
(active-path-certified).** Not the dense ADR-004 `CERTIFIED`, not L4.

## What was wrong

`DeepseekConfig::attn_scale()` multiplied the **whole** `q·k` softmax scale by
`mscale² = yarn_get_mscale(factor, mscale_all_dim)²` (= 1.5896 for V2-Lite). HF does
**not** scale the score by mscale: `self.scaling = qk_head_dim^-0.5`, and the YaRN
`attention_scaling = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)`
multiplies the **cos/sin** — i.e. it attaches to the decoupled-RoPE `q_pe`/`k_pe`
only. For DeepSeek-V2-Lite `mscale == mscale_all_dim` ⇒ `attention_scaling = 1.0`
(no mscale anywhere). Atenia therefore over-scaled `q_nope·k_nope` by 1.59×, a
systematic per-layer error that compounded across 27 layers into the C5 logit gap.

## The fix (`src/moe/mla.rs` only)

1. **`attn_scale()`** → `qk_head_dim^-0.5` only (dropped `* m * m`).
2. New **`attention_scaling()`** = `1.0` without YaRN, else
   `yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)`
   (HF `_compute_yarn_parameters`; **exactly 1.0** when `mscale == mscale_all_dim`).
3. **`project_token`** multiplies the RoPE'd `q_pe` and `k_pe_roped` by
   `attention_scaling` (so the rope dot carries `attention_scaling²`), mirroring HF
   folding it into the cos/sin. Guarded by `if asc != 1.0` → exact no-op for V2-Lite
   and any non-YaRN config.

No other family, loader, runtime, residency, Adapter Toolkit, threshold, or ADR-007
change. The doc comment on `YarnParams` (which had encoded the wrong model) was
corrected.

## Before / after (measured)

| metric | before (MLA-2) | after (MLA-3) |
|---|---|---|
| **C5 real** whole-model `max_abs_diff` vs HF f64 | **2.032** | **2.587e-5** |
| C5 per-position argmax | 3/4 (flip at pos 2) | **4/4 exact** |
| C5 determinism | yes | yes |
| MLA-0 tiny full-forward vs HF | 9.072e-5 | **5.306e-5** (improved) |
| mscale micro-probe (layer 0, vs HF) | 5.51e-2 | matches `hf_exact` 5.37e-8 |

`2.587e-5` is the f32-precision floor: the HF f32-vs-f64 control on this model is
`3.111e-5` with identical argmax, so the active path is now correct to f32 precision.

## C5 run (the L3 certification)

- Harness: `tests/moe_mla1_deepseek_c5_active_path_test.rs` (`#[ignore]`,
  `DEEPSEEK_V2_LITE_DIR` + `ATENIA_MOE_EXPERT_TIER=disk` + `ATENIA_DISK_TIER_DIR`).
- Reference: `fixtures/moe/deepseek_v2lite_c5_ref.{safetensors,json}` — HF f64, **one
  decoder layer at a time** (never the whole model in F64 → **not L4**), driver
  validated on the tiny fixture (`8.126e-9`) first.
- Input: canonical `[1, 100, 200, 300]` (seq 4).
- **Result: `max_abs_diff = 2.587e-5 < 0.5`, argmax exact 4/4, deterministic.**
  Load ~1299 s (writes the ephemeral f32 expert tier to NVMe) + forward ~228 s,
  ~4 GB RAM (MLA-2 disk tier).

> Note: an earlier re-run died mid-load (~40 of ~59 GB tier written) with no verdict —
> a transient crash, not a C5 result. Re-running cleanly produced the PASS above.

## Validation

- MLA-0 (tiny) full forward + greedy/EOS: PASS (`5.306e-5`, argmax exact).
- MLA-2 disk-tier == RAM-tier: still **bit-identical** (`5.306e-5` both).
- DeepSeek runtime / block / scale-cert / residency / residency-tier: green
  (non-YaRN configs unchanged — `attn_scale` unchanged when `yarn = None`).
- Full lib suite (single-threaded, CI mirror): green.

## Status

- **DeepSeek-V2-Lite — MoE-certified L3 (whole model)** = L1 (C1 real 1664 experts +
  C2 real top-6) + C4 (topology) + **C5 (active-path-certified, real weights,
  2.587e-5)**. C3 fixture stays mechanism-level but C5 now validates the real-weight
  MLA attention end-to-end.
- **Not** dense ADR-004 `CERTIFIED`. **L4** (global `model.double()` F64 ~126 GB)
  remains reserved/unreachable.
- Manifest: `docs/numcert/deepseek-v2-lite.moecert.json`
  (`ladder_level_whole_model: L3`).

## Risks / follow-ups

- Single canonical C5 input (same standard as Qwen-MoE C5); other inputs uncovered.
- The disk-tier load is slow (~22 min, writes ~59 GB ephemeral f32 to NVMe); a bf16
  persistent tier for DeepSeek would cut this (out of scope, MLA-2 follow-up).
- The MLA-3 fix is general (handles `mscale != mscale_all_dim` for full DeepSeek-V2),
  but only the `attention_scaling = 1.0` branch is exercised by a real model here.
