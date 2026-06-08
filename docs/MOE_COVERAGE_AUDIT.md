# MoE Architectural Coverage Audit — MOE-COVERAGE-AUDIT (analysis only)

> **Post-MIXTRAL-L3 update (current).** **Three** real MoE families are now
> **MoE-certified L3 (active-path)**: **Qwen1.5-MoE-A2.7B**, **DeepSeek-V2-Lite (MLA)**,
> and **Mixtral-8x7B-v0.1** (MIXTRAL-CERT-1/2/3; C5 active-path worst `3.185e-4` < 0.5,
> argmax exact 4/4, deterministic). All three are **not** dense ADR-004 `CERTIFIED`;
> **L4** (global F64) reserved/unreachable. The current frontier roadmap supersedes the
> FASE 7 recommendation below — see `docs/POST_MIXTRAL_L3_ROADMAP_AUDIT.md`.
>
> **Post-MLA-3 update.** The "DeepSeek-MoE = experimental / tiny fixtures only"
> conclusion below is **superseded**: **DeepSeek-V2-Lite is now MoE-certified L3**
> (real-weight C1+C2, C4 topology, and **C5 active-path** end-to-end on the real
> weights via the MLA-2 disk expert-tier; whole-model `max_abs_diff 2.587e-5`,
> argmax 4/4, after the MLA-3 YaRN mscale fix). `load_from_dir` now loads DeepSeek
> (experimental opt-in). Still **not** dense ADR-004 `CERTIFIED`; **L4** (global F64)
> reserved/unreachable. See `docs/HANDOFF_MLA_3.md` +
> `docs/numcert/deepseek-v2-lite.moecert.json`. The frontier is no longer "basic
> MLA" but Q-LoRA / modern routing / latent-compressed cache / V3-Kimi scale /
> multi-input certification.

**Audit + analysis only — no code, no downloads, no certification, no commits.**
Measures Atenia's *real* MoE architectural coverage against the **actual 2024–2026
ecosystem** (external research, sources at the end), not a historical list.
Vocabulary is kept strict and **evidence-based**:

- **Supported** = a code path loads/builds it (registry + runtime evidence).
- **Validated** = run end-to-end on a real/representative checkpoint (text/logits).
- **Certified** = passes the ADR-007 MoE ladder (Ln) or the dense ADR-004 gate.
- **Hypothetical** = plausibly works by mechanism, but **no evidence** — flagged as
  such, never counted as support.

## FASE 1 — Atenia MoE capability inventory (evidence-grounded)

From `src/moe/{family,detect,runtime,layer,sparse,convention}.rs`, `MOE_OVERVIEW.md`,
the MOE-CERT-2/2-ext/3/4 manifests, and `HANDOFF_MIXTRAL_CERT_1.md`:

**Recognised families (`MoeFamily` enum — exactly three):**
| Family | Router / experts detected | Shared | Renorm | Status (evidence) |
|---|---|---|---|---|
| **Mixtral** | `block_sparse_moe.gate` + `experts.{e}.w1/w3/w2` (classic) or packed `gate_up_proj` | no | **yes** | **Certified L0** (topology `mixtral_scale` 1.639e-7); runtime runs it; **real 8x7B weights not present** |
| **Qwen-MoE** | `mlp.gate` (or `mlp.router`) + `experts.{e}.{gate,up,down}_proj` + `shared_expert` | yes (sigmoid gate) | no (`norm_topk_prob`) | **Certified L3** (real Qwen1.5-MoE-A2.7B, MOE-CERT-2/2-ext/3/4) |
| **DeepSeek-MoE (MLA)** | `kv_a_proj_with_mqa` (MLA) + shared | yes | `norm_topk_prob` | **DeepSeek-V2-Lite: MoE-certified L3** (real-weight C1 1664 experts + C2 + C4 topology + **C5 active-path 2.587e-5**, via MLA-2 disk tier; MLA-1/MLA-3). `load_from_dir` loads it (experimental opt-in). Classic DeepSeek-MoE topology `deepseek_scale` 7.806e-3 = L0. |

**Explicit capabilities (evidence):**
- **Routing:** softmax top-k (any k) + renormalise flag + tie-break (lower index);
  router-parity + routing-margin certified (MOE-CERT-2). **Selection is softmax
  top-k only.**
- **Shared experts:** ungated (Mixtral=none) and **sigmoid-gated** (Qwen) — both
  certified; DeepSeek multi-shared experimental.
- **Expert layouts:** classic per-expert (`w1/w3/w2`, `gate/up/down_proj`) +
  packed/fused (`gate_up_proj`/`down_proj`); both HF-certified.
- **Attention × MoE:** MHA ✓, **GQA ✓**, **Q/K/V bias ✓** (Qwen), **MLA ⚠️
  experimental** (DeepSeek, low-rank KV + decoupled RoPE, MOE-FULL-12).
- **Scale/residency:** sharded load (MOE-PROD-1), **disk-tier expert residency**
  (MOE-PROD-2, ~few GB RAM), bf16/int8 expert tiers, expert LRU cache.
- **Certification substrate:** ADR-007 ladder L0–L4 + per-obligation manifest
  (`schema_variant: moe-decomposition`).

**Known limitations (evidence / handoffs):**
- **Only 3 families hard-coded**; a new family needs Rust (no declarative onboarding
  — `MOE_ADAPTER_SPEC_AUDIT.md`: the ATK `moe` DSL is declarative-but-non-authoritative).
- **Unsupported variants (documented refusals):** **Qwen3-MoE** (per-head QK-norm),
  **DeepSeek Q-LoRA** (`q_a/q_b_proj`), **YaRN** RoPE scaling, **GGUF MoE**.
- **Router is softmax top-k only** — no sigmoid-gated / auxiliary-loss-free /
  bias-corrected / expert-choice / node-limited routing.
- **Decoder-only, attention-or-MLA only** — no SSM/Mamba layers, no encoder-decoder.
- **No multi-token-prediction (MTP) head.**

## FASE 2 — The real MoE ecosystem (external research, 2024–2026)

| Style / family | Architecture essentials | Adoption | Real models |
|---|---|---|---|
| **Mixtral-style** (classic sparse MoE) | 8 experts, top-2, **no shared**, **renorm**, softmax routing, GQA | first open MoE (2023); now **legacy** | Mixtral-8x7B/8x22B |
| **Qwen-MoE-style** | many experts, top-k, **shared expert + sigmoid gate**, no-renorm, GQA + qkv bias | common | Qwen1.5/2-MoE |
| **Fine-grained + shared (DeepSeek-MoE)** | **many small experts** (fine-grained segmentation) + **1–2 shared**, softmax routing | the dominant modern template | DeepSeek-MoE 16B, DeepSeek-V2 |
| **DeepSeek-V2** | **MLA** (latent KV, ~10× KV-cache cut) + DeepSeekMoE (fine-grained + shared) | high | DeepSeek-V2 / V2-Lite (15.7B) |
| **DeepSeek-V3 / V3.x** | MLA + **256 routed/top-8 + 1 shared**, **sigmoid gating**, **auxiliary-loss-free** bias-corrected + **node-limited** routing, **MTP** | **frontier** (671B/37B) | DeepSeek-V3/V3.2, distills |
| **MLA family (beyond DeepSeek)** | latent attention reused widely | rising | Kimi K2 (1T, 384 experts), GLM-4.5/5, Ling |
| **Switch / GShard** | **top-1** (Switch) / top-2 (GShard) routing, capacity factor | foundational; encoder-decoder (T5) | Switch-T5 |
| **Expert-Choice routing** | **experts pick tokens** (not tokens pick experts) → perfect load balance, variable experts/token | research / some training | Google EC, MoE-uniform |
| **Hierarchical MoE** | multi-level routing (global → local sub-experts) | research | Sparse-Transformer++ |
| **DBRX** | **16 experts, top-4** (fine-grained vs Mixtral), no shared, softmax | 2024 | DBRX (132B/36B) |
| **Grok-1** | 8 experts, top-2, softmax | 2024 (Apache-2.0) | Grok-1 (314B) |
| **OLMoE / JetMoE** | fully-open small MoE; 64 experts/top-8 (OLMoE) | open-research | OLMoE-1B-7B, JetMoE-8B |
| **Llama-4** (Scout/Maverick) | MoE, **shared + routed**, 17B active, very low active ratio | frontier 2025 | Llama-4 Scout/Maverick |
| **gpt-oss** | open MoE (21B/3.6B, 117B/5.1B) | 2025 | gpt-oss-20b/120b |
| **MoE + SSM/Mamba (hybrid)** | interleave Mamba/attention + MoE every other layer (16 exp/top-2) | growing | **Jamba**, Jamba-1.5 (52B/12B) |
| **MoE + linear attention** | Gated DeltaNet / sparse attention + MoE | emerging 2026 | Qwen3.5-Next, MiniMax |

**Cross-cutting 2025–2026 trends (sourced):** MoE is now the *default* for frontier
open weights; **active/total ratio keeps dropping** (~25% Mixtral → ~3–4%
Llama-4/Qwen3/DeepSeek-V3); **MLA** is widely adopted (DeepSeek, Kimi, GLM, Ling);
**shared experts** are kept by most (DeepSeek/Kimi/GLM) but **Qwen3 dropped them**;
**routing is moving past softmax top-k** (sigmoid + auxiliary-loss-free bias,
expert-choice, node-limited); **hybrid SSM/linear-attention + MoE** is rising.

## FASE 3 — Coverage matrix (no support claimed without evidence)

| Capability | State | Evidence / note |
|---|---|---|
| Softmax top-k routing | **Certified** | MOE-CERT-2 router parity (Qwen, real) |
| Renormalised top-k (Mixtral conv.) | **Certified (L3)** | Mixtral-8x7B real, C5 active-path (MIXTRAL-CERT-3) |
| No-renorm (`norm_topk_prob=false`) | **Certified (L3)** | Qwen real |
| Shared expert — ungated | **Certified** | Mixtral none / Atenia conv. |
| Shared expert — **sigmoid-gated** | **Certified (L3)** | Qwen real |
| Sparse top-k activation (only routed run) | **Certified** | sparse.rs + residency |
| Router parity (top-k set equality) | **Certified** | MOE-CERT-2 (all 24 layers) |
| GQA + MoE | **Certified (L3)** | Mixtral-8x7B real GQA 4:1 (C5 active-path); Qwen MHA |
| Q/K/V bias + MoE | **Certified (L3)** | Qwen real |
| Packed & classic expert layouts | **Certified** | both, vs HF |
| **MLA + MoE** | **Validated (experimental)** | MOE-FULL-12 tiny (9.999e-6 attn); runtime refuses in `load_from_dir`; **not certified, not real-scale** |
| Disk-tier residency (huge MoE) | **Validated** | Qwen real 14.3B end-to-end |
| **Sigmoid / aux-loss-free / bias routing** (DeepSeek-V3) | **Not supported** | router is softmax top-k only |
| **Expert-choice routing** | **Not supported** | no evidence; tokens-pick-experts only |
| **Hierarchical / multi-level routing** | **Not supported** | single-level router |
| **Encoder-decoder MoE** (Switch-T5) | **Not supported** | decoder-only engine |
| **MoE + SSM / Mamba** (Jamba) | **Not supported** | no SSM layer type |
| **MoE + linear attention** (DeltaNet) | **Not supported** | no linear-attn |
| **Fine-grained many-expert at scale** (256+/top-8) | **Hypothetical** | mechanism is k-agnostic, but **untested** at that count/scale |
| Switch **top-1** routing | **Hypothetical** | top_k with k=1 likely works; **untested** |
| Qwen3-MoE (QK-norm) | **Not supported** | documented refusal |
| MTP head | **Not supported** | no evidence |
| GGUF MoE | **Not supported** | documented gap |

## FASE 4 — Family coverage

| Family | Classification | Justification (evidence) |
|---|---|---|
| **Qwen-MoE** | **Fully supported + Certified L3** | real 14.3B, C1–C5, MOE-CERT-4 |
| **Mixtral** | **Supported + Certified L3 (active-path)** | real 8x7B provisioned (87 GB); C1 256 experts + C2 top-2 + C4 topology + **C5 active-path** (worst 3.185e-4 < 0.5, argmax exact 4/4, deterministic), MIXTRAL-CERT-1/2/3 |
| **DeepSeek-V2-Lite (MLA)** | **Supported + Certified L3 (active-path)** | real 16B-A2.4B; C1+C2+C4+**C5 active-path** 2.587e-5 via MLA-1/2/3 disk-tier |
| **DeepSeek-MoE (classic)** | **Partially supported (experimental)** | MLA forward + scale-topology only; refused by productive loader; tiny fixtures |
| **DeepSeek-V2** | **Probably supported (unverified)** | = MLA + DeepSeekMoE (softmax) ≈ what's experimental; **never run** on real V2 weights |
| **DeepSeek-V3 / V3.x** | **Not supported** | needs **sigmoid + aux-loss-free + node-limited routing** + MTP — none implemented |
| **DBRX** | **Probably supported (unverified)** | 16-exp/top-4 softmax, no shared, classic — mechanically Mixtral-like; **untested** |
| **Grok-1** | **Probably supported (unverified)** | 8-exp/top-2 softmax — Mixtral-like; **untested**, 314B impractical locally |
| **OLMoE / JetMoE** | **Probably supported (unverified)** | top-k softmax, small; **untested** (naming/QK-norm must be checked) |
| **Llama-4 (Scout/Maverick)** | **Not supported (unverified)** | shared+routed MoE but new routing/attention specifics unconfirmed; **untested** |
| **Switch Transformer** | **Not supported** | encoder-decoder + top-1 capacity routing |
| **Jamba (SSM+MoE)** | **Not supported** | hybrid Mamba layers — no SSM engine |
| **Kimi K2 / GLM-4.5+** | **Not supported** | MLA + modern routing at 1T scale |

"Probably supported (unverified)" = the **mechanism** (softmax top-k, classic/packed
experts, GQA) matches, but **no run exists** → explicitly **hypothetical**, not a
support claim.

## FASE 5 — Real coverage estimate

**1. % of the current MoE ecosystem Atenia covers** — split honestly:
- **By popularity of *currently deployed* open MoE:** the classic-softmax cluster
  (Mixtral / Qwen-MoE / DBRX / Grok / OLMoE / JetMoE-style) is a large share, and
  Atenia *mechanically* fits most of it → **~50–60% loadable**, but only **2
  families actually validated/certified**.
- **By the *2025–2026 frontier* (DeepSeek-V3/Kimi/Llama-4/Qwen3/GLM):** these need
  MLA + modern routing + (sometimes) hybrid attention → Atenia covers **~20–30%**
  of the frontier honestly (MLA only experimental; new routing absent).
- **By architectural-diversity count (~12 distinct families above):** **~3
  supported, ~1 experimental → ~25–30%.**

**2. % certified:** still small but tripled. **Three** real MoE families at
real-weight **MoE-certified L3 (active-path)** — **Qwen1.5-MoE-A2.7B**,
**DeepSeek-V2-Lite (MLA)**, and **Mixtral-8x7B-v0.1** (MIXTRAL-CERT-3). Everything
else is supported/experimental/hypothetical. **Certified ≈ 3 of ~12 families (~25%);
not dense ADR-004 CERTIFIED; L4 reserved/unreachable.**

**3. % depending on hypotheses:** the "probably supported (unverified)" set (DBRX,
Grok, OLMoE/JetMoE, DeepSeek-V2, fine-grained-at-scale, Switch-top-1) is **~30–40%
of nominal coverage resting on untested mechanism extrapolation** — must not be
sold as support.

**4. Biggest holes:** (a) **modern routing** (sigmoid / auxiliary-loss-free /
bias-corrected / expert-choice / node-limited) — the frontier moved here; (b)
**MLA productionisation + certification** (gates DeepSeek-V2/V3, Kimi, GLM, Ling);
(c) **declarative family onboarding** (3 hard-coded families vs a fast-diversifying
ecosystem); (d) **hybrid stacks** (SSM/Mamba + MoE, linear-attention + MoE);
(e) Qwen3-MoE (QK-norm) — a major current family, refused.

## FASE 6 — Strategic gaps

**HIGH impact**
- **MLA productionisation + ADR-007 certification** — keystone for the entire
  modern frontier (DeepSeek-V2/V3, Kimi, GLM, Ling). Atenia already has an
  *experimental* MLA (MOE-FULL-12) → highest leverage, partly built.
- **Modern routing** (sigmoid-gated + auxiliary-loss-free bias; later expert-choice)
  — without it, DeepSeek-V3 / Llama-4-class models are uncertifiable even with MLA.
- **Qwen3-MoE (QK-norm, no shared)** — large current family, currently refused.
- **Declarative MoE family onboarding (Adapter Toolkit authoritative path)** — turns
  "1 family = 1 Rust milestone" into config-driven, scaling coverage with the ecosystem.

**MEDIUM impact**
- **DeepSeek-V2-Lite real certification** (15.7B, MLA + fine-grained + shared) — the
  smallest real frontier-arch model; feasible on this hardware (Qwen-scale).
- **Validate the "probably supported" cluster** (DBRX, OLMoE/JetMoE) — cheap runs to
  convert hypothesis → validated.
- **Fine-grained many-expert at scale** (256+/top-8) — confirm the router/residency
  hold at DeepSeek expert counts.

**LOW impact**
- ~~**Mixtral L1–L3**~~ — **DONE** (MIXTRAL-CERT-1/2/3 → MoE-certified L3, active-path). 2023 legacy but now fully certified; no further work.
- **Encoder-decoder MoE (Switch-T5)** — legacy/training-era.
- **MoE + SSM/Mamba (Jamba), hierarchical MoE** — niche/emerging; large new-engine cost.
- **GGUF MoE** — ergonomics, not architecture.

## FASE 7 — Strategic recommendation

**Primary choice: C) Productionise + certify MLA — executed via DeepSeek-V2-Lite
(the entry point of option B), NOT Mixtral (A).**

Rationale:
- The audit's central finding is that **the frontier moved to MLA + fine-grained +
  shared + modern routing** (DeepSeek-V2/V3, Kimi, GLM, Ling, and the template most
  2025–2026 models share). Atenia certifies that world at **~0%**.
- **Mixtral (A) is legacy (2023)** and costs a **94 GB download** for a family whose
  architecture is no longer where the ecosystem is — low strategic ROI; finish it
  only if specific Mixtral users demand it.
- **MLA is half-built already** (experimental MOE-FULL-12, scale-cert 7.806e-3) and
  the **ADR-007 ladder + harnesses are proven** (Qwen L3) and **~80% reusable** — so
  the marginal cost to take MLA from experimental → certified is far below building
  a family from scratch.
- **DeepSeek-V2-Lite (15.7B)** is the **smallest real model with the frontier
  architecture** and is **feasible on this 32 GB host** (Qwen-MoE-scale; the
  MIXTRAL-L3-FEASIBILITY RAM lessons apply). Certifying it productionises MLA and
  *transfers* toward DeepSeek-V3 / Kimi / GLM.
- It also forces the **first real "modern routing" work** (DeepSeek softmax→sigmoid
  path), seeding the HIGH-impact routing gap.

**Close second: D) Adapter Toolkit (authoritative MoE onboarding)** — the long-term
multiplier; pursue right after MLA so new families (Qwen3-MoE, DBRX, Llama-4) become
declarative rather than per-family Rust. **A (Mixtral) is explicitly deprioritised.**

## Executive summary

Atenia has a **deep but narrow** MoE stack: a proven ADR-007 certification ladder,
disk-tier residency for huge models, and **three families certified to real-weight L3
(active-path): Qwen-MoE, DeepSeek-V2-Lite (MLA), and Mixtral-8x7B-v0.1** (post-MIXTRAL-CERT-3;
MLA is now real-weight-certified, no longer experimental) — but it recognises **only 3
families**, and its router is **softmax/sigmoid top-k only** (modern aux-loss-free /
group-limited routing absent). Against the real 2024–2026
ecosystem (~12 distinct MoE families, frontier dominated by **MLA + fine-grained +
modern routing + hybrid stacks**), honest coverage is **~25–30% by architectural
diversity**, **~50–60% "mechanically loadable" classic-softmax MoE** (mostly
untested), and **only ~8% certified** (1 family). The largest holes are **modern
routing** and **MLA productionisation**, which gate essentially every frontier
model (DeepSeek-V2/V3, Kimi, GLM, Llama-4). The highest-leverage next move is **not
finishing legacy Mixtral** but **productionising + certifying MLA via
DeepSeek-V2-Lite**, reusing the proven, ~80%-transferable Qwen tooling, with the
**Adapter Toolkit authoritative path** as the immediate follow-up multiplier.

## External sources consulted

- DeepSeek-V3 Technical Report — arXiv [2412.19437](https://arxiv.org/abs/2412.19437) (MLA, fine-grained + shared experts, sigmoid + auxiliary-loss-free routing, MTP).
- Mixture-of-Experts with Expert Choice Routing — arXiv [2202.09368](https://arxiv.org/abs/2202.09368) (NeurIPS 2022).
- Jamba: Hybrid Transformer-Mamba LM — arXiv [2403.19887](https://arxiv.org/abs/2403.19887); Jamba-1.5 — arXiv [2408.12570](https://arxiv.org/pdf/2408.12570).
- OLMoE: Open Mixture-of-Experts LMs — arXiv [2409.02060](https://arxiv.org/pdf/2409.02060).
- Mixture of Experts in LLMs (survey) — arXiv [2507.11181](https://arxiv.org/html/2507.11181v2).
- Sebastian Raschka, "The Big LLM Architecture Comparison" — [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) (DeepSeek-V3/Kimi/Qwen3/Llama-4 routing + MLA + shared-expert trends).
- Cameron R. Wolfe, "Mixture-of-Experts (MoE) LLMs" — [cameronrwolfe.substack.com](https://cameronrwolfe.substack.com/p/moe-llms).
- "DBRX, Grok, Mixtral: MoE is a trending architecture" — [aimlapi.com](https://aimlapi.com/blog/dbrx-grok-mixtral-mixture-of-experts-is-a-trending-architecture-for-llms).
- "The MoE-ification of the Open Model Ecosystem" — [digitalocean.com](https://www.digitalocean.com/community/tutorials/mixture-of-experts-inference-cost) (DBRX/Arctic/Grok/active-ratio trend).
- "The Rise of MoE: 2025's Leading MoE Models" — [friendli.ai](https://friendli.ai/blog/moe-models-comparison).
- Rohan Paul, "MoE Architectures: 2024–2025 Literature Review" — [rohan-paul.com](https://www.rohan-paul.com/p/mixture-of-experts-moe-architectures).

*Audit + analysis only — no code/model/cert/commit changes; external research
documented above; support claimed only where in-repo evidence exists.*
