# DeepSeek-V2-Lite Certification Feasibility — DEEPSEEK-V2-LITE-FEASIBILITY (analysis only)

> **Outcome (post-MLA-3): realized.** This feasibility was confirmed end-to-end —
> **DeepSeek-V2-Lite is now MoE-certified L3** (MLA-1 C1+C2+C4+C5, MLA-2 disk
> expert-tier, MLA-3 YaRN mscale fix). Real-weight C5 active-path on this 32 GB
> notebook: whole-model `max_abs_diff 2.587e-5 < 0.5`, argmax 4/4, deterministic.
> Not dense ADR-004 `CERTIFIED`; L4 reserved/unreachable. See `docs/HANDOFF_MLA_3.md`
> + `docs/numcert/deepseek-v2-lite.moecert.json`.

> **MLA-0 update.** The three prerequisites this audit identified — **YaRN**
> (critical), **dense-first layer** (`first_k_dense_replace=1`), and the
> **`norm_topk_prob` routing convention** — are now **implemented + validated**
> (experimental) by **MLA-0** (`docs/HANDOFF_MLA_0.md`): a tiny real V2-Lite-like
> `DeepseekV2ForCausalLM` full forward matches HF at **9.072e-5** (argmax exact).
> The current MLA is therefore now **sufficient to start MLA-1** (provision real
> V2-Lite → ADR-007 C1–C5). Remaining (not needed for the cert): latent cache,
> Q-LoRA, productive loader.

**Audit + analysis only — no code, no downloads, no certification, no commits.**
Determines whether Atenia's **current** MLA implementation (`src/moe/mla.rs`,
MOE-FULL-12) is sufficient to *start* an ADR-007 certification of the **real**
DeepSeek-V2-Lite. Strict vocabulary: **implemented / validated / certified /
missing**; **no compatibility assumed without in-repo or config evidence**.

> **Headline (and the unexpected finding).** Atenia targets the **right MLA
> variant** (V2-Lite has `q_lora_rank=null`, which the code already assumes), the
> MLA *math* is validated on tiny fixtures, and the model is **Qwen-scale → RAM-
> feasible on this notebook**. **But the current MLA is NOT sufficient to start a
> *faithful* cert as-is.** Inspecting the **exact V2-Lite config** surfaces two
> **config-confirmed blockers the MLA-COVERAGE-AUDIT under-counted**: (1) **YaRN**
> is active at *every* position (mscale² on the attention scale + reparametrised
> `inv_freq`), not just long context — and Atenia uses plain RoPE + plain scale;
> (2) **`first_k_dense_replace=1`** makes **layer 0 a dense FFN**, but
> `build_deepseek` assembles a MoE block for *every* layer. Both must be addressed
> before a real-weight forward can match HF.

## FASE 1 — Internal DeepSeek inventory (evidence)

From `src/moe/mla.rs`, `src/moe/runtime.rs::build_deepseek`,
`docs/HANDOFF_MOE_FULL_12.md`, `MODEL_FAMILY_VALIDATION.md`, `STATUS.md`:

| Piece | State | Evidence |
|---|---|---|
| MLA low-rank KV (`kv_a`/`kv_a_ln`/`kv_b`) | **Implemented (exp.)** | `DeepseekLayer`, `build_deepseek` parses `kv_lora_rank`, `qk_nope/rope_head_dim`, `v_head_dim` |
| Decoupled interleaved RoPE | **Implemented (exp.)** | `mla.rs::rope_interleaved` |
| Direct query (**no Q-LoRA**) | **Implemented (exp.)** | `w_q: [n_heads*qk_head_dim, hidden]` — matches `q_lora_rank=null` |
| Prefill + per-head KV-cache decode | **Implemented (exp.)** | `MlaLayerCache`, `mla_decode_step` |
| DeepSeek MoE block (softmax top-k + shared) | **Implemented (exp.)** | reuses `RealMoeLayer::forward_auto` |
| MLA numerics | **Validated (tiny)** | attn 9.999e-6 / full 1.475e-3 / scale 7.806e-3 |
| **YaRN** RoPE scaling / **mscale** | **Missing** | no `rope_scaling`/`mscale` field; `mla.rs:209 scale = 1/√qkh` (plain) |
| **Dense-first layer** (`first_k_dense_replace`) | **Missing** | `build_deepseek` `for l in 0..n_layers` assembles MoE for **every** layer; `DeepseekLayer.moe` non-optional |
| Latent/absorb KV cache | **Missing** | explicit "not the latent-compressed cache" (orthogonal to correctness) |
| Productive load (`load_from_dir`) | **Refused** | "refuses DeepSeek-MoE (MLA)"; only `load_from_files`+opt-in on tiny |
| ADR-007 certification | **Missing** | no Ln ladder for DeepSeek |
| Real-weight DeepSeek run | **Missing** | tiny synthetic only |

## FASE 2 — Architectural comparison: real V2-Lite vs Atenia MLA

**DeepSeek-V2-Lite exact config** (HuggingFace `deepseek-ai/DeepSeek-V2-Lite`):
hidden 2048, **27 layers**, 16 heads; `q_lora_rank=null`, `kv_lora_rank=512`,
`qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`, `rope_theta=10000`;
**64 routed experts, top-6, 2 shared**, `moe_intermediate_size=1408`,
`norm_topk_prob=false`, `scoring_func="softmax"`, `topk_method="greedy"`,
`n_group=1`, `topk_group=1`, `routed_scaling_factor=1.0`; **`first_k_dense_replace=1`,
`moe_layer_freq=1`**; **`rope_scaling = {type: yarn, factor: 40,
original_max_position_embeddings: 4096, mscale: 0.707, mscale_all_dim: 0.707,
beta_fast: 32, beta_slow: 1}`**.

| Dimension | Real V2-Lite | Atenia MLA | Classification |
|---|---|---|---|
| **MLA core** (low-rank KV, kv_a_ln, kv_b) | yes (kv_lora 512) | implemented | ✅ match |
| **Decoupled interleaved RoPE** | yes (nope 128 / rope 64) | implemented | ✅ match |
| **Q-LoRA** | **no** (`q_lora_rank=null`) | not implemented (and not needed here) | ✅ match (no gap) |
| **MLA head dims** | nope128/rope64/v128, 16 heads | parsed generically | ✅ match |
| **YaRN / mscale** | **yes** (active at all positions) | **missing** (plain RoPE + plain scale) | 🔴 **gap (critical)** |
| **Dense-first layer** | **yes** (`first_k_dense_replace=1` → layer 0 dense) | **missing** (every layer = MoE) | 🟠 **gap (moderate-critical)** |
| **Routing** | softmax top-k, `norm_topk_prob=false`, scale 1.0, greedy, n_group=1 (no real grouping) | softmax top-k; DeepSeek descriptor says **renorm=true** | 🟡 **gap (verify convention)** |
| **Shared experts** | 2 (DeepSeekMoE) | supported (shared path) | ✅ match (count generic) |
| **Experts** | 64 routed / top-6, classic SwiGLU | supported (classic experts) | ✅ match |
| **Latent KV cache** | (inference memory opt) | missing | ⚪ not needed for correctness cert |
| **Loader** | sharded safetensors (~31 GB bf16) | sharded supported; `load_from_dir` refuses DeepSeek | 🟡 needs the loader lift |

## FASE 3 — Gap analysis: what blocks a real cert *today*

| Gap | Severity | Why |
|---|---|---|
| **YaRN (mscale + inv_freq reparam)** | **Critical** | active at **every** position (`softmax_scale *= mscale²`, mscale≈1.26 → ~1.59×; `inv_freq` NTK-by-parts) → Atenia's plain-RoPE forward **will not match HF even on a 4-token input**. A faithful cert is impossible without it (or it would certify a *different* model). |
| **Dense-first layer** (`first_k_dense_replace=1`) | **Moderate–Critical** | `build_deepseek` assembles a MoE block for layer 0, which has **no experts** (it's a dense SwiGLU MLP) → assembly fails / wrong. Needs an optional dense-FFN layer variant + a branch on `first_k_dense_replace`. |
| **Routing convention** | **Moderate** | `norm_topk_prob=false` + `routed_scaling_factor=1.0`, but the DeepSeek descriptor renormalises (`renorm=true`) → must confirm/fix the combine to match HF (the deepseek_scale fixture matched a *fabricated* convention, not necessarily V2-Lite's). |
| **Productive loader lift / `load_from_dir`** | **Moderate** | currently refuses DeepSeek; the cert harness can use `load_from_files`+opt-in (as the fixtures do), so this is **not** a hard blocker for a cert (only for productive use). |
| **f32-vs-f64 block drift** | **Low (documented)** | DeepSeek MoE block drift ~2e-4/layer → full-forward ~1e-3 (vs Qwen ~1e-7); still ≪ 0.5 with exact argmax — an *honest looser bound*, not a blocker. |
| **Latent cache / Q-LoRA / YaRN-long / graph** | **Not blocking the cert** | correctness cert is single-stream short-input; these are serving / V3 concerns. |

**Direct answer — "what blocks a real cert today?"** **YaRN (critical)** and the
**dense-first layer (moderate-critical)**, plus a **routing-convention check
(moderate)**. The MLA *core* (KV compression, decoupled RoPE, q_lora=0) is the
correct V2-Lite variant and is *not* a blocker. So the current MLA is
**necessary-but-not-sufficient**: a small, bounded prerequisite (YaRN + dense-first
+ routing) must land first.

## FASE 4 — ADR-007 obligation reuse (vs Qwen tooling)

| Obligation | Reuse of Qwen tooling | Note |
|---|---|---|
| **C1** per-expert | **~90%** | classic SwiGLU experts (gate/up/down_proj); same generator + harness, name map + **skip the dense layer 0**; 64×26 = **1664** experts |
| **C2** router | **~90%** | softmax top-k set-equality + margin; `norm_topk_prob=false` like Qwen; k=6 |
| **C3** attention | **~20%** | MLA ≠ GQA → **the NEW work** (YaRN + the existing MLA math); Qwen's attention cert does not transfer; existing MLA validation (9.999e-6) is the seed |
| **C4** assembly/topology | **~as-is** | `deepseek_scale` 7.806e-3 already exists (MOE-FULL-15) = the designated C4 (looser drift documented) |
| **C5** active-path | **~70%** | the harness (real forward vs HF-layer-at-a-time f64) transfers; the reference generator needs the **DeepSeek HF layer (MLA + YaRN)**, and Atenia must run V2-Lite correctly (gated on the C3 prereqs) |

**Overall tooling reuse ≈ 70–80%** for C1/C2/C4/C5 scaffolding — **but** C3 is
genuinely new MLA work (YaRN + dense-first), which Qwen never needed. So
"reuse the Qwen tooling" is true for the *certification machinery*, **not** for the
*attention*, which is the part that needs code.

## FASE 5 — Feasibility on this notebook (i9-14650HX / 32 GB / RTX 4070 8 GB)

**The good news: V2-Lite is Qwen-scale, not Mixtral-scale.** Its experts are
**identical in size to Qwen** (`moe_intermediate=1408`, hidden 2048 → 3·1408·2048 =
**8.65 M params/expert**, F64 69 MB), 64 experts/layer (≈ Qwen's 60). So the
MIXTRAL RAM wall does **not** apply.

| Level | Feasibility | RAM / cost |
|---|---|---|
| **L1** (C1+C2 real weights) | **🟢 VERDE** (post-prereqs) | one-expert-at-a-time ~5 GB; 1664 experts (vs Qwen 1440); ~31 GB download; ~15–30 min run |
| **L2** (fold C4) | **🟢 VERDE** | `deepseek_scale` already certified; docs/manifest fold-in |
| **L3** (C5 active-path) | **🟡 AMARILLO** (post-prereqs) | C5 reference one HF layer at a time ≈ **Qwen-like ~16 GB peak** (experts Qwen-sized) → **fits 32 GB** (unlike Mixtral's ~28–30 GB); Atenia disk-tier run ~Qwen time |

**Caveat:** all three are **gated on the YaRN + dense-first + routing prerequisites**
(FASE 3). Hardware is *not* the blocker for V2-Lite — **code is**. Download ~31 GB
(have 813 GB free).

## FASE 6 — Recommendation

**Should DeepSeek-V2-Lite be Atenia's next big certification? — YES, it is the
right target — but NOT with the current MLA as-is; it needs a bounded prerequisite
milestone first.**

Technical justification:
- **Right variant, right scale.** V2-Lite is `q_lora_rank=null` (the exact MLA
  variant Atenia's math already targets) and is **Qwen-scale → feasible on this 32
  GB notebook** (experts are Qwen-sized; none of the Mixtral RAM problem). Download
  is ~31 GB (vs Mixtral's 94 GB). It is the **smallest real frontier-MLA model** and
  the gateway toward DeepSeek-V3 / Kimi.
- **But the current MLA is necessary-not-sufficient.** The exact config reveals
  **YaRN active at all positions** and a **dense-first layer** — both absent in
  Atenia today. A *faithful* cert is impossible until these land. This is a small,
  well-specified milestone (**MLA-0**), not the Complex latent-cache work.
- **High tooling reuse, low new-code surface.** C1/C2/C4/C5 reuse ~70–80% of the
  Qwen machinery; the only genuinely new code is **YaRN + dense-first + routing-
  convention** (Moderate) — far cheaper than a new family.
- **Versus Mixtral:** V2-Lite certifies a *frontier* architecture at *lower* data
  and RAM cost than legacy Mixtral, and exercises MLA (the strategic gap).

**Recommended sequencing:** **MLA-0** (implement YaRN + dense-first-layer + verify
routing convention, in the experimental MLA path; validate on a tiny YaRN fixture
vs HF) → **MLA-1** (provision V2-Lite ~31 GB; ADR-007 C1–C5 → L1→L2→L3, reusing the
Qwen tooling). Defer latent cache / Q-LoRA / graph (V3/serving concerns).

## Executive summary

DeepSeek-V2-Lite is the **right next certification target** — it is the exact MLA
variant Atenia already implements (`q_lora_rank=null`), its MLA math is validated
on tiny fixtures (attn 9.999e-6), and it is **Qwen-scale, so L1/L2/L3 are RAM-
feasible on this 32 GB notebook** (its experts are Qwen-sized; ~31 GB download).
**However, the current MLA is not sufficient to start a *faithful* cert as-is:**
inspecting the exact config surfaces two real, code-confirmed blockers the prior
audit under-counted — **YaRN** (mscale² + reparametrised `inv_freq`, active at
*every* position, so Atenia's plain RoPE diverges from HF even on 4 tokens) and a
**dense-first layer** (`first_k_dense_replace=1`, which `build_deepseek` does not
handle) — plus a **routing-convention check** (`norm_topk_prob=false` vs the
descriptor's renorm). These form a small, bounded prerequisite (**MLA-0**: Moderate
effort, in the experimental path) that must precede the cert. After it, ~70–80% of
the Qwen ADR-007 tooling transfers (C1/C2/C4/C5); only the MLA attention (C3) is new.
**Verdict: YES to V2-Lite — but via MLA-0 (YaRN + dense-first + routing) → MLA-1
(cert), not by pointing the current MLA at the real weights.**

- **Current compatibility:** MLA core + q_lora=0 ✅; YaRN ❌ (critical); dense-first ❌
  (moderate-critical); routing convention ❓ (verify).
- **Real gaps:** YaRN, dense-first layer, routing convention (all bounded/Moderate).
- **Qwen tooling reuse:** ~70–80% (C1/C2/C4/C5); C3 (MLA+YaRN) is new.
- **Feasibility:** L1 🟢 / L2 🟢 / L3 🟡 — **post-prereqs**; hardware is not the blocker.
- **Recommendation:** YES — sequence MLA-0 (prereqs) → MLA-1 (cert).

## External sources consulted

- DeepSeek-V2-Lite model card + `config.json` — [huggingface.co/deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) (exact: `q_lora_rank=null`, `kv_lora_rank=512`, `qk_nope=128`/`qk_rope=64`/`v_head=128`, 64 experts/top-6/2 shared, `first_k_dense_replace=1`, `rope_scaling` yarn factor 40 / mscale 0.707).
- DeepSeek-V2 paper (MLA + DeepSeekMoE) — arXiv [2405.04434](https://arxiv.org/abs/2405.04434).
- DeepSeek-V3 Technical Report (Q-LoRA contrast, YaRN) — arXiv [2412.19437](https://arxiv.org/pdf/2412.19437); transformers [deepseek_v3 docs](https://huggingface.co/docs/transformers/v4.52.2/model_doc/deepseek_v3).
- PromptLayer / Open Laboratory DeepSeek-V2-Lite specs (15.7B/2.4B active, 27 layers) — [promptlayer.com](https://www.promptlayer.com/models/deepseek-v2-lite), [openlaboratory.com](https://openlaboratory.com/models/deepseek-coder-v2-lite/).
- NVIDIA Megatron/NeMo DeepSeek-V2 model docs — [docs.nvidia.com](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/deepseek-v2.html).

*Audit + analysis only — no code/model/cert/commit changes; external research
documented above; compatibility claimed only where config or in-repo evidence
exists.*
