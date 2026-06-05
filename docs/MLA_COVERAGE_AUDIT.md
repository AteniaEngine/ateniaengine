# MLA Coverage Audit — MLA-COVERAGE-AUDIT (analysis only)

> **Post-MLA-3 update (supersedes the matrix below).** Several rows this audit
> marked **Absent** are now done on **real weights**: **ADR-007 certification of
> DeepSeek-V2-Lite → MoE-certified L3** (C1 1664 experts + C2 + C4 topology + **C5
> active-path** end-to-end, whole-model `2.587e-5`, argmax 4/4), **real-weight MLA
> run** (the real V2-Lite, MLA-2 disk expert-tier, ~4 GB RAM), and the **productive
> `load_from_dir`** path (experimental opt-in). The MLA attention numerics are now
> validated on the **real** weights end-to-end (not just tiny fixtures), and the
> **YaRN mscale** bug found doing so was fixed (MLA-3). **Still Absent:**
> latent/absorb compressed KV cache, Q-LoRA, graph/CUDA MLA, V3/Kimi modern routing
> + scale, multi-input certification. Not dense ADR-004 `CERTIFIED`; L4 reserved.
> See `docs/HANDOFF_MLA_3.md`, `docs/HANDOFF_MLA1_C5_ROOT_CAUSE.md`,
> `docs/numcert/deepseek-v2-lite.moecert.json`.

> **MLA-0 update (post-audit).** Two gaps this audit listed as **Absent** are now
> **Implemented + validated (experimental)** by **MLA-0** (`docs/HANDOFF_MLA_0.md`):
> **YaRN** (inv_freq reparam + `mscale²` on the attention scale) and the
> **dense-first layer** (`first_k_dense_replace`), plus the DeepSeek
> `norm_topk_prob` routing convention — validated on a tiny V2-Lite-like real
> `DeepseekV2ForCausalLM` (full forward vs HF 9.072e-5). Still **Absent**: latent/
> absorb KV cache, Q-LoRA, graph/CUDA MLA, productive loader, certification.

**Audit + analysis only — no code, no downloads, no certification, no commits.**
Measures Atenia's *real* Multi-head Latent Attention (MLA) coverage against the
state of the art (external research, sources at the end). Strict, evidence-based
vocabulary:

- **Implemented** = code exists in-repo (file/line evidence).
- **Experimental** = behind opt-in, not on the productive path.
- **Validated** = run end-to-end vs a reference (numbers).
- **Certified** = passes ADR-007 (Ln) / ADR-004.
- **Hypothetical / Absent** = no evidence — flagged, never counted as support.

## FASE 1 — Internal MLA inventory (evidence-grounded)

Source: `src/moe/mla.rs` (MOE-FULL-12), `src/moe/runtime.rs`,
`docs/HANDOFF_MOE_FULL_12.md`, `docs/HANDOFF_MOE_FULL_15.md`, `MOE_OVERVIEW.md`.

Atenia **does** have an MLA implementation — `src/moe/mla.rs`, **MOE-FULL-12** —
but it is deliberately a *correctness-first imperative* path, **not** the
memory-optimised one:

| MLA component | In Atenia? | Evidence |
|---|---|---|
| Low-rank **KV** compression (`kv_a_proj_with_mqa` → `kv_lora_rank+qk_rope`, `kv_a_layernorm`, `kv_b_proj` expand) | ✅ Implemented (experimental) | `mla.rs` `DeepseekLayer.w_kv_a/kv_a_ln/w_kv_b` |
| **Decoupled RoPE** (qk_nope + qk_rope; key RoPE shared across heads; **interleaved GPT-J** rotation) | ✅ Implemented (experimental) | `mla.rs::rope_interleaved`, `mla_attention` |
| **Asymmetric head dims** (`qk_head_dim = nope+rope`, separate `v_head_dim`, scale `qk_head_dim^-0.5`) | ✅ Implemented (experimental) | `DeepseekConfig::qk_head_dim` |
| Prefill + **incremental KV-cache decode** | ✅ Implemented (experimental) | `MlaLayerCache`, `mla_decode_step` |
| MLA **numerics** vs HF | ✅ Validated (tiny) | attn **9.999e-06**; full-forward **1.475e-03**; scale-topology **7.806e-03** (argmax/greedy exact) — MOE-FULL-12/15 |
| MLA **+ MoE** coexistence | ✅ Validated (tiny) | reuses certified `RealMoeLayer::forward_auto` |
| **Latent / compressed KV cache** ("absorb" mode — *the actual MLA memory win*) | ❌ **Absent** | `mla.rs`: *"Correctness-first (**not the latent-compressed cache**)"* — caches full decompressed per-head K/V |
| **Q-LoRA** query compression (`q_a_proj`/`q_b_proj`/`q_a_layernorm`) | ❌ Absent | `DeepseekLayer.w_q` is direct; MOE-FULL-12 limitation "`q_lora_rank=None` only" |
| **YaRN**-scaled RoPE (long context 32K→128K) | ❌ Absent | MOE-FULL-12 limitation "default RoPE only" |
| **Absorption** SVD inference trick | ❌ Absent | not implemented |
| **Graph / kernel / CUDA / VRAM** integration | ❌ Absent | imperative CPU-only; not on the AMG graph |
| **Productive loader** support | ❌ Refused | `runtime.rs`: `load_from_dir` "refuses DeepSeek-MoE (MLA)"; only `load_from_files` + opt-in on tiny fixtures |
| **ADR-007 certification** (DeepSeek/MLA) | ❌ Absent | no Ln ladder for DeepSeek; tiny fixtures + f32 drift only |
| **Real-weight MLA run** | ❌ Absent | tiny synthetic fixtures only |

**One-line truth:** Atenia has the **MLA *math*** (the V2-Lite variant) **validated
on tiny fixtures**, but **not MLA's *point*** — the compressed latent KV cache —
nor Q-LoRA, YaRN, graph integration, productive loading, or certification.

## FASE 2 — MLA taxonomy (external research)

**What problem MLA solves.** The KV cache is the memory bottleneck of long-context
/ high-batch transformer inference. MHA caches full K,V per head; GQA shrinks it by
sharing K/V across head groups. **MLA** (DeepSeek-V2, 2405.04434) instead caches a
single **low-rank latent vector** per token (`kv_lora_rank`, e.g. 512) and
**decompresses K,V on demand**, cutting KV-cache memory ~**10×** *while matching or
beating MHA quality* — the rare "better AND cheaper" attention.

**What changes architecturally:**
- **Joint low-rank KV** down-projection + per-head up-projection.
- **Decoupled RoPE:** split q/k into a `nope` part (no RoPE, compressible) and a
  small `rope` part (carries position; the key's RoPE part is shared across heads).
  RoPE can't commute through the low-rank absorb, so it's kept separate.
- **Asymmetric q/k vs v head dims.**
- **(DeepSeek-V3)** also **Q-LoRA**: the query is low-rank compressed too
  (`q_lora_rank=1536`); V2-Lite leaves the query uncompressed (`q_lora_rank=0`).
- **(Long context)** **YaRN** RoPE scaling (4K→32K→128K).

**What it demands of the runtime:**
- An **"absorb" inference path** that stores only the latent (`kv_lora_rank +
  qk_rope`) per token and folds `kv_b`/`o` into the q/output projections so K,V are
  never materialised — *this is where the memory win lives*.
- A **modified KV-cache representation** (latent, not per-head K/V).
- **Interleaved RoPE** + the decoupled split in the attention kernel.
- For V3/Kimi: **Q-LoRA** projections + **YaRN** scaling.

**Family map (research):**
- **DeepSeek-V2 / V2-Lite** — MLA (KV-only, `q_lora_rank=0`) + DeepSeekMoE (softmax,
  fine-grained + shared). *V2-Lite = 15.7B, the smallest real MLA model.*
- **DeepSeek-V3/V3.2** — MLA + **Q-LoRA (1536)** + YaRN + sigmoid/aux-loss-free
  routing + MTP (+ V3.2 sparse attention).
- **Kimi K2** — MLA (DeepSeek recipe scaled, 64 heads, 1T, Q-LoRA).
- **GLM-4.5** — **GQA** (partial RoPE), **NOT MLA**. **GLM-5** — MLA + sparse attn.
- **Sarvam, others** — pattern: GQA at small scale, **switch to MLA when the family
  scales up** ("MLA is the upgrade path once a family scales"; "GQA 8:1 is the
  default unless you are on DeepSeek/Kimi infra").

## FASE 3 — MLA coverage matrix

| MLA capability | State | Evidence |
|---|---|---|
| Low-rank KV compression (compute) | **Implemented (experimental)** | `mla.rs` |
| Latent **KV cache** (absorb) | **Absent** | explicit "not the latent-compressed cache" |
| MLA attention path (math) | **Implemented + Validated (tiny)** | 9.999e-06 vs HF |
| MLA inference (prefill+decode) | **Implemented (experimental)** | `mla_decode_step`, greedy/EOS |
| MLA numerics | **Validated (tiny)** | attn 9.999e-6 / full 1.475e-3 (f32-vs-f64) |
| MLA **certification** | **Absent** | no ADR-007 ladder; tiny fixtures only |
| MLA + MoE coexistence | **Validated (tiny)** | reuses certified MoE block |
| MLA routing interactions | **Validated (tiny)** | DeepSeek softmax MoE (top-6, 2 shared) |
| **Q-LoRA** query compression | **Absent** | `q_lora_rank=None` only |
| **YaRN** long-context RoPE | **Absent** | default RoPE only |
| Graph / CUDA / VRAM MLA | **Absent** | imperative CPU |
| Productive MLA load path | **Absent (refused)** | dense + `load_from_dir` refuse |
| Real-weight MLA run | **Absent** | synthetic tiny only |

## FASE 4 — Gap analysis (what's missing for *productive* MLA)

| Gap | Effort | Why |
|---|---|---|
| **ADR-007 certification of DeepSeek-V2-Lite** (real weights, C1–C5) | **Moderate** | reuses the existing MLA math + the ~80%-transferable Qwen tooling; correctness cert does **not** need the latent cache → the cheapest real win |
| **Q-LoRA** query compression | **Moderate** | add `q_a/q_b_proj` + `q_a_layernorm`; gates V3 / Kimi |
| **YaRN** RoPE scaling | **Moderate** | long-context DeepSeek/Kimi |
| **Latent / absorb KV cache** | **Complex** | new cache representation + decode hot-path + the `kv_b`/`o` absorption algebra — *the actual MLA memory benefit*; without it MLA ≈ slow MHA |
| **Graph / kernel / CUDA / VRAM** MLA | **Complex** | the imperative path has no graph node → no perf, no GPU |
| **Productive loader lift + CLI** for DeepSeek/MLA | **Moderate** | mirror the dense fail-loud-lift + `generate` routing already done for Mixtral/Qwen |
| **DeepSeek-V3 modern routing** (sigmoid + aux-loss-free bias + node-limited) + **MTP** | **Critical (separate)** | not MLA per se, but co-required for V3/Kimi; router is softmax-top-k only today |

**Direct answer — "what's missing for productive MLA?"** Correctness-wise, very
little for the **V2-Lite** variant (the math is done; needs real-weight cert +
loader lift). Serving-wise, the **latent/absorb KV cache + graph/GPU integration**
(Complex) are missing — i.e. Atenia can *certify* MLA correctness soon, but cannot
*serve* MLA at its memory advantage without the Complex work. V3/Kimi additionally
need **Q-LoRA + YaRN + modern routing**.

## FASE 5 — Modern-family compatibility

| Family | Classification | Justification |
|---|---|---|
| **DeepSeek-V2 / V2-Lite** | **Partially supported (experimental)** | MLA math (q_lora=0) + DeepSeekMoE match what's implemented + validated on tiny; **blocked from certified/productive by**: no real-weight run, loader refusal, no ADR-007 ladder, no latent cache (perf). *Not* blocked by missing math. |
| **DeepSeek-V3 / V3.2** | **Blocked by MLA *and* other factors** | needs **Q-LoRA** (absent) + **YaRN** (absent) + **sigmoid/aux-loss-free routing** (absent) + **MTP** (absent) (+ V3.2 sparse attn). Multiple blockers. |
| **Kimi K2** | **Blocked by MLA + scale + routing** | MLA + Q-LoRA (absent) + new routing + 1T scale (infeasible locally). |
| **GLM-4.5** | **Blocked by *other* factors, NOT MLA** | GLM-4.5 is **GQA** (which Atenia supports) — its blockers are its specific MoE config / routing / size, not MLA. *Honest correction: GLM-4.5 is not an MLA model.* |
| **GLM-5** | **Blocked by MLA + sparse attention** | MLA (cache/Q-LoRA gaps) + DeepSeek Sparse Attention (absent). |
| **DeepSeek-R1 distills** | **Supported (dense)** | distills are Llama/Qwen dense — already load (not MoE/MLA). |

## FASE 6 — MLA roadmap (proposed)

| Milestone | Scope | Effort | Risk | Dependencies |
|---|---|---|---|---|
| **MLA-1 — DeepSeek-V2-Lite real cert** | provision V2-Lite (~15.7 B); ADR-007 C1–C5 reusing the MLA math + Qwen tooling (correctness, *not* the latent cache) → DeepSeek-MoE/MLA from L0 toward L1–L3 | **M** | **Med** (f32-vs-f64 block drift ~1e-3 documented; C5 RAM per MIXTRAL-L3-FEASIBILITY lessons) | download + opt-in run; no new attention code |
| **MLA-2 — Productive loader lift + CLI for DeepSeek/MLA** | route detected DeepSeek to `MoeRuntime` behind opt-in (mirror MOE-INTEGRATE-2) | **S–M** | Low | MLA-1 |
| **MLA-3 — Q-LoRA + YaRN** | `q_a/q_b_proj` + `q_a_layernorm`; YaRN RoPE scaling → unlocks DeepSeek-**V3** / Kimi attention | **M** | Med | MLA-1 |
| **MLA-4 — Latent/absorb KV cache** | compressed-latent cache + `kv_b`/`o` absorption + decode hot-path → the real MLA *memory* win | **L (Complex)** | High | MLA-1..3 |
| **MLA-5 — Graph / CUDA / VRAM MLA** | move MLA off the imperative path onto the AMG graph + GPU kernels | **XL (Complex)** | High | MLA-4 |
| *(separate)* **modern routing** (sigmoid + aux-loss-free) + MTP | co-required for V3/Kimi end-to-end | **M–L** | Med | independent of MLA |

## FASE 7 — Strategic recommendation

**Should MLA be Atenia's next big investment? — Yes, but *scoped and sequenced*,
starting with certification (MLA-1), not serving-grade MLA.**

Technical justification:
- **The math already exists and is validated** (9.999e-6 attn). The cheapest,
  highest-value step is **MLA-1: certify DeepSeek-V2-Lite** — it converts
  experimental MLA into a **real-weight ADR-007 certificate** of the *dominant
  frontier attention family*, reusing the proven, ~80%-transferable Qwen tooling,
  and **needs none of the Complex work** (the latent cache is a perf optimisation
  orthogonal to numeric correctness). This is the natural continuation of the
  MoE-CERT series and squarely fits Atenia's niche (**auditable numerics on a
  laptop**, not high-throughput serving).
- **MLA is strategically central but not universal.** It gates DeepSeek-V2/V3 and
  Kimi (highest-profile open frontier) and is "the upgrade path once a family
  scales" — *but* GLM-4.5 stays GQA and "GQA 8:1 is the default unless on
  DeepSeek/Kimi infra". So MLA earns a **focused** bet, not an all-in pivot.
- **Honesty about the limit:** MLA's headline benefit is **KV-cache memory** for
  **long-context / high-batch serving** — exactly the regime Atenia does *not*
  target. So **MLA-4/5 (the Complex serving-grade cache + graph) are lower
  priority for Atenia** than for an inference vendor. Atenia should bank the
  **correctness certificate** (MLA-1..3) and defer the memory/graph optimisation.
- **Versus Mixtral:** MLA-1 (DeepSeek-V2-Lite) delivers a *frontier-architecture*
  certificate at Qwen-comparable scale/cost, whereas finishing Mixtral certifies a
  2023 legacy family for 94 GB. **MLA-1 ≫ Mixtral** in strategic value.

**Recommendation: YES — execute MLA-1 (DeepSeek-V2-Lite certification) next;** then
MLA-2 (loader lift) and MLA-3 (Q-LoRA+YaRN for V3/Kimi); **defer MLA-4/5** (latent
cache + graph/GPU) until serving is actually a goal.

## Executive summary

Atenia is **further along on MLA than the coverage audit implied — but on the
*wrong half***. It has a **validated (tiny) imperative implementation of MLA's
*math*** (low-rank KV, decoupled interleaved RoPE, asymmetric heads, prefill+decode;
attn 9.999e-6 vs HF) that **reuses the certified MoE block** — yet it **lacks MLA's
entire reason to exist**: the **compressed latent KV cache** (it caches full
decompressed K/V), plus **Q-LoRA**, **YaRN**, graph/GPU integration, productive
loading, and any **certification**. So MLA coverage is: **math implemented +
validated on tiny fixtures; latent cache / Q-LoRA / YaRN / productive / certified =
absent.** DeepSeek-V2-Lite is *partially supported* (math matches; blocked only by
the missing real-weight cert + loader lift), DeepSeek-V3 / Kimi are *blocked by MLA
(Q-LoRA/YaRN) and by modern routing*, and **GLM-4.5 is GQA — not MLA — so it isn't
blocked by MLA at all**. The strategic answer is **yes, invest in MLA — but
scoped**: take the cheap, high-value **DeepSeek-V2-Lite certification** first
(reusing the existing math + Qwen ADR-007 tooling, no latent cache required), then
Q-LoRA+YaRN for V3/Kimi, and **defer the Complex latent-cache + graph work** (a
serving optimisation outside Atenia's correctness-on-constrained-hardware niche).

## External sources consulted

- DeepSeek-V2 (MLA origin) — arXiv [2405.04434](https://arxiv.org/abs/2405.04434).
- DeepSeek-V3 Technical Report (MLA + Q-LoRA + YaRN + routing + MTP) — arXiv [2412.19437](https://arxiv.org/pdf/2412.19437); transformers docs [deepseek_v3](https://huggingface.co/docs/transformers/v4.52.2/model_doc/deepseek_v3).
- "Towards Economical Inference: Enabling MLA in Any Transformer" (partial-RoPE + SVD absorption) — arXiv [2502.14837](https://arxiv.org/pdf/2502.14837).
- "Hardware-Centric Analysis of DeepSeek's MLA" — arXiv [2506.02523](https://arxiv.org/pdf/2506.02523).
- Kimi K2 Technical Report (MLA, 64 heads, 1T) — arXiv [2507.20534](https://arxiv.org/pdf/2507.20534).
- Sebastian Raschka — [MLA gallery](https://sebastianraschka.com/llm-architecture-gallery/mla/) and [Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) (GLM-4.5 = GQA, GLM-5 = MLA; "GQA 8:1 default unless DeepSeek/Kimi").
- Turing Post — [Kimi K2 vs DeepSeek-R1 vs Qwen3 vs GLM-4.5](https://www.turingpost.com/p/chinesemodels).
- "A Visual Walkthrough of DeepSeek's MLA" — [towardsai.net](https://towardsai.net/p/artificial-intelligence/a-visual-walkthrough-of-deepseeks-multi-head-latent-attention-mla-%EF%B8%8F); planetbanatt [MLA](https://planetbanatt.net/articles/mla.html).

*Audit + analysis only — no code/model/cert/commit changes; external research
documented above; MLA support claimed only where in-repo evidence exists.*
