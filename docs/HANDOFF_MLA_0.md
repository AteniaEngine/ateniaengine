# HANDOFF — MLA-0: YaRN + dense-first layer + DeepSeek routing convention

Milestone: **MLA-0** — make Atenia's experimental MLA path sufficient to *start* a
faithful ADR-007 certification of **DeepSeek-V2-Lite**, by closing the three
config-confirmed prerequisites the `DEEPSEEK_V2_LITE_FEASIBILITY` audit surfaced.
**Experimental, CPU-only, opt-in.** No real model downloaded, no certification, no
latent KV cache, no Q-LoRA, no productive-loader lift, no dense/runtime/ATK change.
Predecessor: MOE-FULL-12 (the MLA math), MOE-CERT-4 (the ADR-007 ladder).

## What was added (and why)

The exact DeepSeek-V2-Lite config (`q_lora_rank=null`, which already matched
Atenia) revealed that the MLA *math* was the right variant but **three things were
missing for the real-weight forward to match HuggingFace even on a short input**:

1. **YaRN RoPE scaling** — V2-Lite ships `rope_scaling.type=yarn` (factor 40,
   `original_max_position_embeddings=4096`, `mscale=0.707`). YaRN is **active at
   every position**, not just long context: it (a) reparametrises `inv_freq`
   (NTK-by-parts blend of interpolated/extrapolated frequencies) and (b) multiplies
   the attention **softmax scale by `mscale²`** (`mscale =
   yarn_get_mscale(factor, mscale_all_dim)`). Atenia used plain RoPE + plain scale.
   - Added (`src/moe/mla.rs`): `YarnParams`, `yarn_get_mscale`,
     `yarn_find_correction_dim/range`, `rope_inv_freqs(dim, base, yarn)` (a faithful
     port of HF `DeepseekV2YarnRotaryEmbedding`), and `DeepseekConfig::attn_scale()`.
     `rope_interleaved` now takes a precomputed YaRN-aware `inv_freqs`.
   - For V2-Lite `mscale == mscale_all_dim`, so HF's cos/sin `_mscale` cancels to
     1.0 — only `inv_freq` + the softmax scale change (exactly what is modelled).
2. **Dense-first layer** (`first_k_dense_replace=1`) — V2-Lite's layer 0 is a plain
   dense SwiGLU MLP, not MoE. `build_deepseek` previously assembled a MoE block for
   *every* layer (would fail on the dense layer).
   - Added: `DenseFfn` + `DeepseekFfn { Moe | Dense }`; `DeepseekLayer.ffn` replaces
     `.moe`. `load_core` skips MoE assembly for DeepSeek layers `l < first_k_dense`;
     `build_deepseek` builds the dense FFN (`mlp.{gate,up,down}_proj`) for those.
3. **Routing convention** — V2-Lite has `norm_topk_prob=false` (no top-k renorm;
   shared expert ungated). The old code always resolved to the renormalising
   convention.
   - Added: `DeepseekConfig.renorm_topk` (from `norm_topk_prob`, default `true`);
     `layer_step` now calls `forward_with(Atenia)` when `true` (renorm) and
     `forward_with(HuggingFaceQwen)` when `false` (no-renorm + ungated shared, since
     DeepSeek has no shared-expert gate).

**Backward compatibility:** absent `rope_scaling` → `yarn = None` → plain RoPE +
plain scale (unchanged). Absent/`0` `first_k_dense_replace` → every layer MoE
(unchanged). `norm_topk_prob` default `true` → renorm (= the old
`forward_auto`→Atenia for DeepSeek). Mixtral/Qwen never enter these branches.

## Validation (real, measured)

Fixture: a tiny real `DeepseekV2ForCausalLM` (`.double()`) configured like V2-Lite
— **YaRN active, `first_k_dense_replace=1`, `q_lora_rank=null`,
`norm_topk_prob=false`, 3 layers (0 dense, 1–2 MoE)** —
`fixtures/moe/generate_deepseek_v2lite_mla0_reference.py` →
`deepseek_v2lite_mla0.{safetensors,json}` + `_config.json`.

| Test | Result |
|---|---|
| **Full forward vs HF f64** (`mla0_v2lite_full_forward_matches_hf`) | **max_abs_diff = 9.072e-5** (< 0.5 ADR-004 gate; < 5e-3 empirical), **per-position argmax exact** |
| Dense-first load (`mla0_v2lite_loads_with_dense_first_layer`) | ✅ loads, `num_layers=3`, family DeepSeek-MoE |
| Greedy → EOS + determinism | ✅ matches HF greedy prefix |
| YaRN unit (`yarn_inv_freqs_differ_from_plain_and_no_yarn_is_unchanged`) | ✅ YaRN reparametrises inv_freq; no-yarn unchanged; `mscale>1`, `mscale(1,·)=1` |

The 9.072e-5 drift is f32-vs-f64 (MoE-block dominated), tighter than the existing
`deepseek_full` (1.475e-3). **YaRN-on changes the result; YaRN-off is unchanged;
the dense-first layer is respected; the no-renorm convention matches HF** — all
proven by the forward matching HuggingFace.

## Regression (no threshold lowered, nothing broken)

- DeepSeek existing: `moe_deepseek_runtime_test` 4/4 (full 1.475e-3, attn 9.999e-6),
  `moe_deepseek_block_test` 2/2 (2.196e-4) — **unchanged**.
- All families: `moe_scale_cert` 3/3, `moe_certification` 4/4, `qwen_moe_cert` 6/6,
  `mixtral_cert` 4/4.
- `src/moe/mla.rs` unit tests 4/4. **Full lib suite: 838 passed / 0 failed.**

## Files

- `src/moe/mla.rs` — YaRN (`YarnParams`, helpers, `rope_inv_freqs`, `attn_scale`),
  `DenseFfn` + `DeepseekFfn` enum, `DeepseekConfig.{yarn,renorm_topk}`,
  `layer_step` dense/MoE + convention, RoPE tests + a YaRN unit test.
- `src/moe/runtime.rs` — `build_deepseek` parses `rope_scaling`/`norm_topk_prob`/
  `first_k_dense_replace`, assembles dense-first FFNs; `load_core` skips MoE
  assembly for dense DeepSeek layers (gated on family, byte-identical for
  Mixtral/Qwen).
- `fixtures/moe/generate_deepseek_v2lite_mla0_reference.py` + the 3 fixture files.
- `tests/moe_mla0_deepseek_v2lite_test.rs` — 3 tests.
- `docs/STATUS.md`, this handoff.

## What MLA-0 did NOT do

No real DeepSeek-V2-Lite download/cert; no latent/absorb KV cache; no Q-LoRA; no
YaRN long-context beyond what V2-Lite needs (the formula is general, but only the
all-position effects are exercised); no productive-loader lift; no DeepSeek-V3 /
Kimi; no graph/CUDA/VRAM MLA; no dense/runtime-policy/Adapter-Toolkit/numerics
change; **no threshold lowered**.

## Next (MLA-1)

With MLA-0 landed, the prerequisites for a faithful V2-Lite cert are closed.
**MLA-1** = provision DeepSeek-V2-Lite (~31 GB) + run ADR-007 C1–C5 (reusing ~70–80%
of the Qwen tooling per `DEEPSEEK_V2_LITE_FEASIBILITY`) → L1→L2→L3.
