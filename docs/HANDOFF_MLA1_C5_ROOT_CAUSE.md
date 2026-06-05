# HANDOFF — MLA-1 / C5 root cause: YaRN `mscale` wrongly applied to the MLA attention scale

**Diagnosis milestone (no fix implemented).** C5 (active-path) failed for the real
DeepSeek-V2-Lite (`max_abs_diff = 2.032`, argmax 3/4, flip at pos 2). This document
proves the root cause with experiments, discards the alternatives, and specifies the
minimal fix. **DeepSeek-V2-Lite stays MoE-certified L2. No manifest change, no L3, no
threshold change.**

## TL;DR

The MLA attention applies the YaRN **`mscale`** to the **softmax scale of the whole
`q·k` dot product**. HuggingFace (transformers 5.6.2) does **not**: for
DeepSeek-V2-Lite the effective `attention_scaling` is **exactly 1.0** (because
`mscale == mscale_all_dim`), and any YaRN scaling would attach to the decoupled-RoPE
part only — never to the full score. Atenia therefore over-scales every attention
score by `mscale² = 1.2608² = 1.5896`, a systematic error that compounds across 27
layers into the 2.032 logit gap. **Feeding the correct `attention_scaling = 1.0`
reproduces HF's layer-0 attention to `5.37e-8`.** Confirmed Atenia bug; the reference
is reliable.

## Chronology

1. **C5 run** (real, disk tier): worst `max_abs_diff = 2.032` (pos 3), argmax flip at
   pos 2 (Atenia 207 vs ref 200), deterministic. → FAIL, did not certify L3.
2. **Per-layer diagnostic** (`moe_mla1_deepseek_c5_diag_test`, vs an F64 per-layer
   reference): embeddings exact (0.0), lm_head-on-ref-hidden exact (3.8e-6),
   accumulation monotonic 0.048→32.75, **no isolation spike**.
3. **Precision control** (`generate_…c5_reference.py precision`): HF's own forward in
   **f32 vs f64** → final logits differ by only **3.111e-5**, **argmax identical**.
   ⇒ the model is f32-stable; "precision drift" is **refuted**; Atenia (2.032) is
   ~65 000× worse than naive f32 ⇒ a real bug, not the precision regime.
4. **mscale micro-probe** (`diag_mscale_probe.py`, one layer, f64): pinned the bug to
   the YaRN attention scaling (below).

## Evidence

### Precision control — the model is f32-stable (reference is reliable)

HF one-layer-at-a-time, f32 vs f64, per-layer post-FFN `|f32−f64|` stays flat at
`~2e-7 … 6e-4` (absolute, on activations up to **1130**), **final logits
`max_abs_diff = 3.111e-5`, argmax match**. A pure-f32 forward of this model is
argmax-identical to f64 — so the C5 failure is not f32 accumulation, and the F64
reference is trustworthy (its f32 and f64 forms agree).

### Per-layer diagnostic — no localized component fault; pure accumulation

| view | layer 0 | layer 7 | layer 20 | layer 26 |
|---|---|---|---|---|
| ISOLATION post_attn (intrinsic) | 5.5e-2 | 2.8e-1 | 1.34e0 | 1.12e0 |
| ISOLATION post_ffn (intrinsic) | 4.8e-2 | 1.7e-1 | 7.8e-1 | 4.01e0 |
| ACCUMULATION (real C5 growth) | 4.8e-2 | 9.4e-1 | 2.46e0 | **3.28e1** |

- **Embeddings** `0.0`, **lm_head** on the reference final hidden `3.8e-6` → I/O,
  embedding, final-norm and head are correct.
- **Layer 0 is dense-first** (no MoE/router/shared) yet already drifts `5.5e-2` in
  `post_attn` → the fault is in the **MLA attention**, not MoE/router/shared/combine.
- No layer where the isolation jumps ~10× → not a single broken layer; the *same*
  per-layer error compounds. The final hidden gap `32.75` on a ~1130-magnitude
  activation (~2.9 % relative) is normalised by the final RMSNorm down to the
  `2.032` logit gap.

### mscale micro-probe — the exact culprit (one layer, f64, vs HF `post_attn[0]`)

```
HF attention_scaling = 1.000000   vs   Atenia mscale = 1.260804   (mscale² = 1.589626)
inv_freq  max_abs_diff(HF, Atenia) = 1.05e-8           (YaRN reparametrisation OK)

mode = atenia    (mscale² on WHOLE score)        post_attn |Δ| vs HF = 5.51e-2   ← current bug
mode = hf_fix    (mscale² on ROPE part only)     post_attn |Δ| vs HF = 5.08e-2   ← also wrong
mode = hf_exact  (attention_scaling = 1.0)       post_attn |Δ| vs HF = 5.37e-8   ← matches HF
```

The numpy replica reproduces Atenia exactly (`5.51e-2` == the harness isolation
`0.055`). Switching the YaRN attention scaling to HF's actual value (**1.0**)
collapses the error to **5.37e-8**. `inv_freq` already matches HF to `1e-8`, so the
YaRN frequency reparametrisation is correct — only the **mscale on the score** is wrong.

## Root cause (`src/moe/mla.rs`)

`DeepseekConfig::attn_scale()`:

```rust
Some(y) => {
    let m = yarn_get_mscale(y.factor, y.mscale_all_dim);
    base * m * m            // <-- applies mscale² to the WHOLE q·k softmax scale
}
```

HuggingFace `DeepseekV2`:
- `self.scaling = qk_head_dim^-0.5` — **no mscale on the score**;
- the YaRN `attention_scaling = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)`
  multiplies the **cos/sin** (so it scales the decoupled-RoPE `q_pe`/`k_pe` only);
- for DeepSeek-V2-Lite `mscale == mscale_all_dim` ⇒ `attention_scaling = 1.0` ⇒ the
  mscale **cancels entirely** (none on the score, none on the RoPE part).

Atenia instead multiplies every score by `mscale² = 1.5896`. The MLA-0 comment
("only inv_freq + the softmax-scale change") encodes this incorrect model: HF does
**not** keep a softmax-scale mscale when `mscale == mscale_all_dim`.

**Why MLA-0 (tiny) still passed.** The tiny fixture has the same bug
(`mscale² = 1.147² = 1.316`) but only 3 layers, small scores and a loose `< 5e-3`
bound, so the error stayed at `9.07e-5` — below the bound. The real model
(`mscale² = 1.59`, 27 layers, 128-dim nope, larger scores) amplifies it to `2.03`.
The tiny test lacked the sensitivity to surface the latent bug.

## Classification

**A) confirmed Atenia bug** — specifically **D) wrong YaRN convention** in the MLA
attention scale. NOT B (accumulation as the cause), NOT C (expected f32 drift —
refuted by the f32==f64 control), NOT E (reference defective — the reference's f32
and f64 forms agree to 3e-5). Components **ruled out** by experiment: embeddings, I/O,
dense-first FFN, router, shared experts, MoE combine, RMSNorm, lm_head, and the YaRN
`inv_freq` reparametrisation (matches HF to 1e-8).

## Proposed fix (DESIGN ONLY — not implemented)

Minimal, isolated to the DeepSeek MLA YaRN path (`src/moe/mla.rs`):

1. **`attn_scale()`** → return `qk_head_dim^-0.5` **only** (drop `* m * m`).
2. Add **`attention_scaling()`** = `1.0` without YaRN, else
   `yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)`
   (HF's `_compute_yarn_parameters`; `= 1.0` for V2-Lite since `mscale == mscale_all_dim`).
3. In **`project_token`**, multiply the RoPE parts `q_pe` and `k_pe_roped` by
   `attention_scaling` (so only the decoupled-RoPE dot carries it — exactly like HF
   folding it into cos/sin). For V2-Lite this is a no-op (×1.0).

- **Files:** `src/moe/mla.rs` only. **No** Qwen/Mixtral, loader, runtime, Adapter
  Toolkit, or threshold change.
- **Risk:** low. YaRN-only; `attention_scaling = 1.0` for V2-Lite and the MLA-0 tiny.
  Non-YaRN DeepSeek fixtures (`deepseek_scale`, `deepseek_block`) have no `rope_scaling`
  → `attn_scale` unchanged → unaffected. MLA-2 disk==RAM parity is preserved (same
  scale on both tiers). MLA-0 tiny will match HF **better** (≈1e-7 vs the current
  9.07e-5) and still pass its asserts.
- **Validation after the fix:** re-run MLA-0 (tiny), the DeepSeek runtime/scale/block
  tests, MLA-2 parity, the mscale probe (expect `atenia` mode to match `hf_exact`),
  then regenerate nothing and re-run the real C5 harness.
- **Estimated probability C5 passes after the fix: very high (>95%).** With the fix,
  layer-0 attention matches HF to `5.4e-8`; the same scalar correction applies to
  every layer, so the per-layer attention error drops from ~2e-2 to ~5e-8. The only
  residual would then be f32 precision, which the control showed is `3.1e-5` in logits
  with **identical argmax** — comfortably inside the `< 0.5` gate.

## Diagnostic artifacts (committed with this milestone)

- `fixtures/moe/generate_deepseek_v2lite_c5_reference.py` — modes `tiny` / `real` /
  `diag` (per-layer F64) / `precision` (HF f32-vs-f64 control).
- `fixtures/moe/diag_mscale_probe.py` — the one-layer mscale root-cause probe.
- `tests/moe_mla1_deepseek_c5_active_path_test.rs` — C5 cert harness (`#[ignore]`).
- `tests/moe_mla1_deepseek_c5_diag_test.rs` — per-layer diagnostic harness (`#[ignore]`).
- `src/moe/mla.rs` + `src/moe/runtime.rs` — read-only `debug_*` capture methods
  (no numeric change; used only by the diagnostic harness).
- Reference fixtures `deepseek_v2lite_c5_{ref,diag}.{safetensors,json}`.
