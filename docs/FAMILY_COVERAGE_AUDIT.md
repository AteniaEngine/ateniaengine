# Family Coverage Audit (FAMILY-COVERAGE-AUDIT-1)

**Audit only — no code changed, no new families, no commits.** A precise,
family-centric snapshot of what Atenia actually supports, after the
MODEL-INTAKE / FORMAT-INTAKE / FP8 / STREAMING-LOADER milestones. Grounded in the
adapter registry (`src/model_adapters/`), the intake compat layer
(`src/model_adapters/compat.rs`), the numeric certificates (`docs/numcert/`), and
the functional record ([MODEL_FAMILY_VALIDATION.md](./MODEL_FAMILY_VALIDATION.md)
/ [STATUS.md](./STATUS.md)). Complements the broader
[MODEL_COVERAGE_EXECUTIVE_AUDIT.md](./MODEL_COVERAGE_EXECUTIVE_AUDIT.md) with a
sharper certification lens.

## Status vocabulary (FASE 2)

Ordered weakest → strongest:

- **EXPERIMENTAL** — opt-in, gated, not on the default path (MoE).
- **SUPPORTED** — a registered adapter builds + loads it; no real-checkpoint
  generation logged in-repo.
- **VALIDATED** — exercised end-to-end on a *real* checkpoint (load → coherent
  greedy text → EOS), but **no** f64 numeric certificate.
- **CERTIFIED** — passes the ADR-004 f64 gate (`max_abs_diff < 0.5` vs a PyTorch
  F64 reference) with a committed `numcert` manifest.
- **PARTIAL** — supported only on a subset (e.g. text path of a multimodal
  family; cert infrastructure present but evidence pending).

These are deliberately distinct: "loads" ≠ "generates correctly" ≠ "is
numerically certified". The word *certified* is reserved for ADR-004.

**MoE certification is a parallel lane, not this scale.** A MoE family cannot be
ADR-004 `CERTIFIED` (a single global F64 forward is infeasible at scale +
incomplete under sparse routing — [ADR-007](./decisions/ADR-007-moe-certification-ladder.md)).
MoE families are graded on the **L0–L4 decomposition ladder** and labelled
**`MoE-certified Ln`** (e.g. `MoE-certified L2`). This token is **never collapsed
into** the dense `CERTIFIED`, and the two are **never summed** into one
"N families certified" count without splitting dense-certified from
MoE-certified-Ln. **Current ladder state (post-MLA-3):** **Qwen1.5-MoE-A2.7B = L3**
and **DeepSeek-V2-Lite = L3** (real-weight C1+C2+C4+C5 active-path, MOE-CERT + MLA-1/2/3);
**Mixtral-8x7B-v0.1 = partial L1** (real weights provisioned; real-weight C1 256
experts + C2 top-2 set-equality on all 32 layers, MIXTRAL-CERT-1; C3 mechanism; C4
topology available → L2/L3 are follow-ups). All are
`MoE-certified Ln`, never the dense `CERTIFIED`; **L4 reserved/unreachable**. (The
older "all three at L0" wording referred to the pre-MOE-CERT state.) Raising Mixtral
up the ladder is the remaining MOE-CERT work.

---

## FASE 1 + 3 — Family inventory & adapter map

**Dense decoder families — 7 registered adapters** (`ModelFamily` enum,
`ADAPTERS` registry):

| Family | Arch string | Adapter module | Distinct mechanics | Status |
|---|---|---|---|---|
| **Llama** | `LlamaForCausalLM` | `llama_family.rs` (+ base `nn/llama`) | GQA, SwiGLU, RMSNorm, RoPE; the fallback adapter | **CERTIFIED** |
| **Qwen2 / 2.5** | `Qwen2ForCausalLM` | `llama_family.rs` | + QKV bias, tied embeddings, large vocab | **CERTIFIED** |
| **Qwen3** | `Qwen3ForCausalLM` | `qwen3.rs` | per-head QK-Norm pre-RoPE, tied lm_head | **VALIDATED** |
| **Mistral** | `MistralForCausalLM` | `llama_family.rs` | pure Llama topology (SWA deferred) | **VALIDATED** |
| **Phi-3 / 3.5** | `Phi3ForCausalLM` | `phi3.rs` | fused QKV + gate_up, LongRoPE | **VALIDATED + cert-ready** |
| **Gemma 2** | `Gemma2ForCausalLM` | `gemma2.rs` | dual-norm, attn/final soft-cap, GeGLU, scaled embed, `q_dim≠hidden` | **VALIDATED + cert-ready** |
| **Gemma 3** (text) | `Gemma3ForCausalLM` | `gemma3.rs` | + per-head QK-Norm, dual-RoPE (local/global), SWA pattern | **PARTIAL (text) + cert-ready** |

**Intake compat layer (MODEL-INTAKE-1)** — extends reach without new adapters:
- **Allowlist** (evidence-gated, → Llama adapter): `LLaMAForCausalLM` (legacy
  casing), `YiForCausalLM` (Yi adopts the Llama architecture). Most
  Llama-compatible models (Vicuna, NousHermes, SmolLM, OpenLLaMA, TinyLlama,
  Yi-HF) already declare `LlamaForCausalLM` → load **natively**.
- **Generic opt-in** (`ATENIA_INTAKE_GENERIC=1`): any unknown arch that passes
  `check_llama_topology` runs as Llama, loudly **UNCERTIFIED**, fail-loud at
  weight binding.

**MoE families — EXPERIMENTAL** (gated by `ATENIA_EXPERIMENTAL_MOE` /
`ATENIA_ENABLE_MOE`, off the dense default path):

| MoE family | Detection | Real run | Status |
|---|---|---|---|
| **Qwen-MoE** (`Qwen2MoeForCausalLM`) | shared_expert + per-expert gate/up/down | ✅ real Qwen1.5-MoE-A2.7B e2e (disk-tier, slow) | **EXPERIMENTAL · MoE-certified L3** (MOE-CERT-2/3/4) |
| **DeepSeek-MoE (+ MLA)** (`DeepseekV2ForCausalLM`) | `kv_a_proj_with_mqa` + shared | ✅ real DeepSeek-V2-Lite e2e (MLA-2 disk-tier, ~4 GB RAM) | **EXPERIMENTAL · MoE-certified L3** (real-weight C1+C2+C4+**C5 active-path 2.587e-5**; MLA-1/2/3) |
| **Mixtral** (`MixtralForCausalLM`) | `block_sparse_moe` | ✅ real Mixtral-8x7B-v0.1 provisioned (87 GB) | **EXPERIMENTAL · partial L1** (real-weight C1 256 experts + C2 top-2; C3 mechanism; C4 topology available; MIXTRAL-CERT-1) |

---

## FASE 2 — Certification status (the honest tiering)

| Tier | Families |
|---|---|
| **CERTIFIED (f64 gate)** | **Llama, Qwen2** — committed `numcert` manifests with real `max_abs_diff_vs_f64` (fixtures: TinyLlama, SmolLM2, Qwen2.5-1.5B, Llama-3.2-1B). |
| **VALIDATED (real e2e, no f64)** | Phi-3.5 (RUNTIME-REAL-4), Gemma 2 (RUNTIME-REAL-3), Mistral, Qwen3; Qwen-MoE (experimental real run). |
| **PARTIAL / cert-ready** | Phi-3, Gemma 2, Gemma 3 — **CERTIFY-BREADTH-1** wired the f64 harness + generator + manifests, but every `max_abs_diff_vs_f64` is **null** (evidence pending; blocked on RAM for the PyTorch-F64 pass). |
| **SUPPORTED only** | Gemma 3 beyond the text path = **not** supported (multimodal out of scope). |

**Key gap:** *functional* coverage (7 families generate real text) far exceeds
*certified* coverage (2 families). This is the #1 honesty caveat — the
infrastructure to lift Phi-3/Gemma exists; only the RAM-gated f64 run is missing.

---

## FASE 4 — Gap analysis (missing families, difficulty, ROI)

Families **not** supported today (rejected at load unless they declare
`LlamaForCausalLM`), with build difficulty and demand:

| Family | Why not supported | Difficulty | Demand / ROI |
|---|---|---|---|
| **Yi (distinct arch)** | `YiForCausalLM` allowlisted → Llama; HF-format Yi is native | trivial (done) | covered |
| **DeepSeek dense (V2-lite/distill)** | distills declare Llama/Qwen → already load; native DeepSeek dense arch unmapped | low (often already native) | medium |
| **Falcon 3** | declares Llama-compatible → loads; classic Falcon distinct | low / N-A | low–med |
| **Falcon (classic), RWForCausalLM** | LayerNorm (not RMS), parallel attention, multi-query fused-QKV | **high** (new graph) | low (legacy) |
| **InternLM2** | interleaved fused Wqkv layout | medium-high | medium |
| **OLMo / OLMoE** | non-parametric LayerNorm; OLMoE is MoE | high | low–med |
| **Cohere / Command-R** | LayerNorm, logit scale, tied embed | medium-high | medium |
| **StableLM** | partial rotary + QK layernorm + biases | medium | low |
| **GPT-2 / NeoX / StarCoder / MPT** | learned pos-emb / different blocks | high | low (legacy) |
| **DeepSeek-V3 (MLA + FP8 MoE)** | MLA + huge MoE | **very high** (+ out of scope per rules) | high but excluded |
| **Vision / multimodal, SSM/Mamba, encoder-decoder, embeddings/rerank** | different engine class | very high | out of charter |

**Pattern:** the cheap wins are mostly *already covered* (compatible models reuse
`LlamaForCausalLM`). The remaining distinct-arch families each need real
graph-builder work, and the highest-demand one (DeepSeek-V3) is explicitly out of
scope. So the near-term ROI is **lifting certification on the 7 existing
families**, not adding new ones.

---

## FASE 5 — Ecosystem impact (estimated coverage)

- **By popularity-weighted demand (what people deploy): ~60–70%.** The 7
  supported families + the large "declared-as-Llama" tail (Yi, Vicuna, SmolLM,
  many DeepSeek distills, Nous/Hermes, OpenChat, Zephyr…) dominate real-world
  open-weight *text-generation* usage, and Atenia now loads them across **all
  common containers/dtypes** (safetensors single+sharded, `.bin` single+sharded,
  GGUF, FP8) — so the *format* friction that used to block them is gone.
- **By raw architecture count on HF: ~25–35%.** The long tail of distinct
  decoders (Falcon/InternLM/OLMo/Cohere/StableLM/GPT-NeoX/…) plus the entire
  non-decoder world pulls this down.
- **By *certified* demand: ~25–35%.** Only Llama + Qwen2 clear the f64 bar today;
  Phi/Gemma are GREEN-behavioural but uncertified.

**Honest framing:** Atenia is **deep + now wide on intake, but narrow on
certification**. The format work closed the "can't even load it" gap; the open
gap is "loads and runs, but only 2 families are numerically certified."

---

## FASE 6 — Roadmap (NOW / NEXT / LATER / NEVER)

**NOW** (highest ROI, no new families):
- **Finish CERTIFY-BREADTH-1 evidence** — run the committed f64 harness on
  Gemma-3-1B → Gemma-2-2B → Phi-3.5 (needs ≥8/21/30 GiB free RAM). Pure
  execution; lifts 3 families VALIDATED→CERTIFIED. This is the single biggest
  trust win and the infrastructure already exists.

**NEXT** (real ROI, modest effort):
- **Qwen3 + Mistral real-checkpoint validation** (SUPPORTED → VALIDATED) — they
  load; log a real e2e run.
- **A tested generic-decoder allowlist expansion** (DeepSeek-dense distills, more
  Llama-compatible distinct strings) — coverage via the existing compat layer,
  no new adapters.

**LATER** (genuine new families, each a graph-builder project):
- **InternLM2** or **Cohere/Command-R** (medium-demand, distinct blocks) — the
  proof that ATKv2 → v1 can add breadth.
- **DeepSeek-MoE / MLA productionisation** (high demand, large effort; currently
  out of scope).

**NEVER** (out of charter / measured dead-ends):
- Vision/multimodal, SSM/Mamba, encoder-decoder, embedding/rerank (different
  engine).
- Classic Falcon / GPT-2-class legacy blocks unless demand reverses.
- GPTQ/AWQ as *families* (they are a quantized-weight *decoding* concern, not a
  family).

---

## ENTREGA FINAL — answers

1. **Supported families (7 dense + 3 MoE):** Llama, Qwen2, Qwen3, Mistral, Phi-3,
   Gemma 2, Gemma 3-text (dense, registered adapters) · Qwen-MoE, Mixtral,
   DeepSeek-MoE (experimental, opt-in) · plus the Llama-compatible tail via the
   intake allowlist + generic opt-in.
2. **Certified families:** **Llama, Qwen2** only (ADR-004 f64). The bar is
   reserved and not over-claimed.
3. **Partial families:** Phi-3, Gemma 2, Gemma 3 — VALIDATED behaviourally,
   certification-infrastructure-ready, **f64 evidence pending** (CERTIFY-BREADTH-1,
   RAM-gated); Gemma 3 is text-only (multimodal out of scope).
4. **Missing families:** classic Falcon, InternLM2, OLMo/OLMoE, Cohere/Command-R,
   StableLM, GPT-2/NeoX/StarCoder/MPT, DeepSeek-V3 (MLA), and all non-decoder
   classes. (Yi / Vicuna / SmolLM / many DeepSeek distills are **not** missing —
   they load as Llama.)
5. **ROI ranking:** (1) finish f64 certs for Phi/Gemma [done-infra, exec only];
   (2) real-validate Qwen3 + Mistral; (3) expand the tested allowlist;
   (4) one new mid-demand family (InternLM2/Cohere); (5) DeepSeek MLA (big);
   never: multimodal/SSM/enc-dec/legacy-GPT.
6. **Risks:** (a) functional ≫ certified — easy to *over-claim* "supports family
   X" when only 2 are f64-certified; (b) the **silent Llama-fallback for
   `LlamaForCausalLM`** runs unknown checkpoints on *assumed* Llama topology —
   correctness risk if they deviate; (c) Gemma/Phi certs blocked on RAM, not
   code; (d) MoE is experimental + non-interactive (min/token), one family proven.
7. **Estimated coverage:** ~60–70% popularity-weighted intake; ~25–35% raw-arch;
   **~25–35% certified-by-demand** (Llama + Qwen2).
8. **Strategic recommendation:** the moat is **auditable numerics on constrained
   hardware**, now backed by broad **format intake**. Lead with that. Spend the
   next effort *certifying the families already supported* (Phi/Gemma) rather than
   adding exotic families — it raises the trustworthy-coverage number with
   infrastructure that already exists, and makes the breadth claims honest.
9. **Next objective:** **CERTIFY-BREADTH-2** — execute the committed f64 harness
   to fill the Gemma-3-1B / Gemma-2-2B / Phi-3.5 certificates (on a box with
   adequate free RAM), moving 3 families from VALIDATED to CERTIFIED. Pure
   execution + evidence; no new code, no new families.

---

*Audit only — no source changed, no commits, no CI. Sources:
`src/model_adapters/` (registry + compat), `docs/numcert/`,
`docs/MODEL_FAMILY_VALIDATION.md`, `docs/STATUS.md`,
`docs/MODEL_COVERAGE_EXECUTIVE_AUDIT.md`, and the HANDOFF series
(RUNTIME-REAL-1..4, MODEL-INTAKE-1, CERTIFY-BREADTH-1, FORMAT-INTAKE-1/2,
FP8-SAFETENSORS-1, STREAMING-LOADER-1).*
