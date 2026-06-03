# Model Coverage — Executive Audit

**Scope.** Documentation-only audit. No code, no commits, no CI. Answers one
question for a decision-maker:

> If a user brings an arbitrary model **today**, can Atenia run it? With what
> probability? What would block it?

Grounded in the current tree (`src/model_adapters/`, `src/adapter_toolkit/`,
`src/moe/`, `src/v17/loader/`) and the production-signal record in
[STATUS.md](./STATUS.md). It supersedes nothing; it is a snapshot for roadmap
selection after the NUMERIC-POLICY series closed.

---

## TL;DR

- Atenia is a **dense-decoder, text-generation** engine with **strong, narrow**
  coverage: **7 dense families** with real adapters, **4 validated end-to-end on
  real checkpoints**, **2 numerically certified against an f64 reference**.
- **The single biggest lever on "can it run this?" is the `architectures` string
  in `config.json`.** A model whose architecture is one of the 7 known names (or
  the literal `LlamaForCausalLM`) loads; **anything else is rejected at load**
  (`resolve_adapter → None`). There is **no generic/automatic fallback** for an
  unknown architecture.
- **MoE** is real but experimental and effectively **Qwen-MoE only** (one real
  run); Mixtral is unblocked in code but unrun (no local weights); DeepSeek/MLA
  is detected but not productionised.
- **Format reality:** safetensors (single + sharded) and GGUF (read, common
  quants) — **yes**. PyTorch `.bin`/`.pth`, GPTQ/AWQ/bitsandbytes, FP8 — **no**
  (or experimental-CPU-only, not on the load path).
- **Realistic coverage of the modern open-weight ecosystem: ~55–65 % by
  popularity-weighted demand, ~25–35 % by raw architecture count.** High on the
  families people actually deploy (Llama/Qwen/Mistral/Gemma/Phi dense); zero on
  vision/multimodal, SSM/Mamba, encoder-decoder, and embedding/rerank models.

---

## FASE 1 — Family Inventory

`ModelFamily` (`src/model_adapters/mod.rs:62`) is a **closed enum of 7**:
`Llama, Qwen2, Qwen3, Mistral, Phi3, Gemma2, Gemma3`. The adapter registry
(`ADAPTERS`, 7 entries, first-match-wins) dispatches on the `architectures[0]`
string (primary) / `model_type` (fallback).

| Family | Recognized arch | Adapter module | Real e2e run | f64-certified | State |
|---|---|---|---|:---:|---|
| **Llama** | `LlamaForCausalLM` | `llama_family.rs` (+ base `nn/llama`) | RUNTIME-REAL-1 (TinyLlama 1.1B) | ✅ (0.076) | **GREEN, certified** |
| **Qwen2** | `Qwen2ForCausalLM` | `llama_family.rs` | RUNTIME-REAL-2 (Qwen2.5-1.5B) | ✅ (0.000335) | **GREEN, certified** |
| **Qwen3** | `Qwen3ForCausalLM` | `qwen3.rs` | — (unit-validated: QK-norm) | partial | **GREEN, structural** |
| **Mistral** | `MistralForCausalLM` | `llama_family.rs` | smoke-validated (dense) | via Llama bar | **GREEN, structural** |
| **Phi-3** | `Phi3ForCausalLM` | `phi3.rs` | RUNTIME-REAL-4 (Phi-3.5-mini) | ❌ (no f64 pipeline) | **GREEN behavioural** |
| **Gemma 2** | `Gemma2ForCausalLM` | `gemma2.rs` | RUNTIME-REAL-3 (Gemma-2-2B) | ❌ (no f64 pipeline) | **GREEN behavioural** |
| **Gemma 3** | `Gemma3ForCausalLM` | `gemma3.rs` | — (text-only path) | ❌ | **GREEN structural (text only)** |

**MoE families** (recognized by tensor-name fingerprint, `src/moe/family.rs`,
behind `ATENIA_EXPERIMENTAL_MOE`/`ATENIA_ENABLE_MOE`):

| MoE family | Detection | Real run | State |
|---|---|---|---|
| **Qwen-MoE** (`Qwen2MoeForCausalLM`) | shared_expert + per-expert gate/up/down | ✅ **Qwen1.5-MoE-A2.7B-Chat real e2e** (coherent text, disk-tier) | **experimental, real-GREEN behavioural** |
| **Mixtral** (`MixtralForCausalLM`) | `block_sparse_moe` | ❌ no local real weights (sharded loader now exists) | **experimental, UNRUN** |
| **DeepSeek-MoE** (+ MLA) | `kv_a_proj_with_mqa` / `shared_experts` | ❌ | **detected only, not tiered/run** |

**Buckets:**
- **Supported + certified (f64):** Llama, Qwen2.
- **Supported + validated (real e2e, behavioural):** + Phi-3, Gemma 2, (Qwen-MoE experimental).
- **Supported, structural only (no real-weight run yet):** Qwen3, Mistral, Gemma 3.
- **Experimental:** all MoE (Qwen-MoE real-run; Mixtral unrun; DeepSeek detect-only).
- **Not supported:** everything else (see FASE 3 / FASE 7).

---

## FASE 2 — Adapter Toolkit Audit

Two layers exist:

1. **v1 adapter registry** (`src/model_adapters/`) — the **executing** layer. 7
   adapters implementing the full `AteniaModelAdapter` surface (HF + GGUF weight
   mapping, GGUF↔HF name mapping, graph builder, residency hints, config policy).
   Family-specific capability bits: `fused_qkv`/`fused_gate_up` (Phi-3),
   `gemma2_softcaps` (Gemma 2). This is where coverage actually lives.

2. **Adapter Toolkit v2** (`src/adapter_toolkit/`) — a **declarative DSL +
   generator + validator + registry + introspect** layer that *names* the
   recurring cross-family patterns (RoPE kind, attention kind, GQA, fused
   QKV/MLP, multi-EOS, tied embeddings) and validates a spec, then maps it onto a
   v1 family. **Key finding: ATKv2 does not add new executable families.**
   `resolve_family` (`spec.rs:318`) accepts only
   `llama|qwen|qwen2|qwen3|gemma|gemma2|gemma3|phi|phi3|mistral` → the **same 7
   v1 families**. An unknown family string is a resolution error; an unregistered
   architecture falls back to v1 (and to `None` if truly unknown).

**Audit verdict:**
- **Existing adapters:** 7, all complete (7/7 trait surface), no dead adapters.
- **Redundant adapters:** none. Qwen2/Mistral correctly *reuse* the Llama base
  rather than duplicating it — good factoring, not redundancy.
- **Missing adapters:** every non-enumerated architecture (Falcon, Yi, InternLM,
  OLMo, Cohere/Command-R, StableLM, GPT-NeoX/2, StarCoder, MPT, DeepSeek-dense,
  PhiMoE, plus all non-decoder shapes).
- **Declarative coverage gap:** ATKv2 can describe patterns it **cannot yet
  execute as a new family** — the DSL is ahead of the executable backend. Adding
  a genuinely new family (e.g. Falcon's parallel-attention/partial-rotary block,
  or DeepSeek MLA dense) still requires **v1 graph-builder work**, not just a DSL
  entry. ATKv2's value today is *validation/introspection of the 7*, not breadth.

---

## FASE 3 — Architecture Coverage Classification

| Architecture | Status | Mechanism / note |
|---|---|---|
| **Llama** (1/2/3/3.1/3.2) | ✅ Certified | Dedicated base; fallback for `LlamaForCausalLM` |
| **Qwen** (Qwen2/2.5) | ✅ Certified | `Qwen2ForCausalLM` via Llama base (+QKV bias, tied) |
| **Qwen3** | ✅ Supported | Dedicated (per-head QK-norm); no real-weight run logged |
| **Mistral** (7B dense) | ✅ Supported | `MistralForCausalLM` via Llama base |
| **Mixtral** (8x7B/8x22B) | 🟡 Experimental | MoE detected; sharded loader exists; **no real run** |
| **DeepSeek** (V2/V3/MoE, MLA) | 🟡 Detect-only | MLA + shared-experts recognized; **not tiered, not run**; dense DeepSeek not an adapter |
| **Phi** (Phi-3/3.5 mini) | ✅ Supported (behavioural) | Dedicated (fused QKV/gate_up, LongRoPE); no f64 |
| **Gemma** (2, 3-text) | ✅ Supported (behavioural) | Dedicated (softcaps, dual-norm, scaled embed); no f64; **Gemma 3 multimodal NOT covered** |
| **Falcon** (Falcon/Falcon3) | ❌ Not supported | `FalconForCausalLM` → `resolve_adapter == None` (rejected). Falcon3 *may* ship as Llama-compatible and route via fallback, but classic Falcon (parallel attn, multi-query, no adapter) does not |
| **Yi** | ⚠️ Conditional | Yi ships as `LlamaForCausalLM` → would route through the Llama fallback and **likely run untested**; not explicitly validated |
| **InternLM / InternLM2** | ❌ Not supported | `InternLM2ForCausalLM` unknown → rejected |
| **OLMo / OLMoE** | ❌ Not supported | Unknown arch; OLMoE is MoE-unsupported too |
| **Other relevant** (Cohere/Command-R, StableLM, GPT-2/NeoX, StarCoder/MPT, PhiMoE) | ❌ Not supported | No adapter; rejected at load |

**The conditional row matters:** a non-trivial slice of HF models declare
`LlamaForCausalLM` regardless of their actual brand (Yi, many SmolLM, Vicuna,
several DeepSeek distills, NousHermes, etc.). Those route through the Llama
adapter and **usually work**, but are **untested** and silently assume exact
Llama topology — a correctness risk if the checkpoint deviates (e.g. different
RoPE scaling or norm placement) without changing the arch string.

---

## FASE 4 — Format Coverage

| Format / capability | Supported | Where | Limits |
|---|:---:|---|---|
| **safetensors (single file)** | ✅ | `v17/loader/safetensors_reader.rs` | F32/BF16/F16; **no FP8**; whole file into RAM (no mmap) |
| **safetensors (sharded, index.json)** | ✅ | `shard_index.rs` + `sharded_reader.rs` | Full `weight_map`, per-shard streaming; proven on 13B+ and 8-shard MoE |
| **HF directory (config.json)** | ✅ | `gguf_config.rs` / adapters | Parses Llama-config + MoE metadata |
| **GGUF (read)** | ✅ | `gguf_reader.rs` + `gguf_decode.rs` | F32/F16/Q8_0/Q5_0/Q5_K/Q6_K decode→F32; **Q4_K partial** |
| **PyTorch `.bin`** (single-file) | ✅ | `v17/loader/pytorch_bin.rs` | **FORMAT-INTAKE-1** — `torch.save` ZIP+pickle transcoded to safetensors; contiguous F32/F16/BF16; sharded `.bin` + `.pth` still pending |
| **GPTQ / AWQ pre-quant** | ⚠️ Experimental | `src/quant/*` | CPU reconstruction→F32 only; **not on the load path** |
| **bitsandbytes / FP8** | ❌ | — | No reader |
| **Dense** | ✅ | core | The mature path |
| **MoE** | 🟡 Experimental | `src/moe/*` | Opt-in flag; Qwen-MoE real, Mixtral unrun, DeepSeek detect-only; disk-tier + int8/bf16 tier certified |
| **Tokenizer** | ✅ | `src/tokenizer/` | HF `tokenizers` (json + SP + BPE) + `minijinja` chat templates; special tokens/multi-EOS |
| **ONNX / Raw** | ❌ | enum stub only | Declared in `ModelFormat`, no loader |

**Practical takeaways:**
- A user with **safetensors (HF layout)** of a supported family: ✅ smooth.
- A user with a **GGUF Q8_0/Q6_K** of a supported family: ✅ (decoded to F32).
- A user with a **`.bin` checkpoint, a GPTQ/AWQ pack, or FP8**: ❌ must convert
  first. This is a real friction point for a chunk of HF downloads.

---

## FASE 5 — Certification Coverage

The numeric-certification bar (default `Certified` mode, f64 reference,
`max_abs_diff < 0.5`) is what separates "trustworthy" from "runs":

| Family | f64 reference fixture | Behavioural real-run | Certification class |
|---|:---:|:---:|---|
| Llama | ✅ (TinyLlama, SmolLM2, Llama-3.2-1B) | ✅ | **Certified** |
| Qwen2 | ✅ (Qwen-2.5-1.5B) | ✅ | **Certified** |
| Qwen3 | ❌ | structural | **Validated (structural)** |
| Mistral | (inherits Llama bar) | smoke | **Validated** |
| Phi-3 | ❌ (no f64 pipeline) | ✅ | **Partially validated (behavioural)** |
| Gemma 2 | ❌ (no f64 pipeline) | ✅ | **Partially validated (behavioural)** |
| Gemma 3 | ❌ | structural | **Partially validated** |
| Qwen-MoE | ❌ (topology/block scale only) | ✅ real | **Experimental, behaviourally validated** |
| Mixtral / DeepSeek-MoE | ❌ | ❌ | **Unvalidated** |

**Two-tier reality:** only **Llama + Qwen2** clear the full f64 bar. Phi-3 and
Gemma are GREEN *behaviourally* but lack an f64 reference (the PyTorch-f64
pipeline was never wired for those families) — a known, documented gap, not a
regression. The NUMERIC-POLICY-3 governance layer now *persists and enforces*
certificates, but for non-f64 families there is no f64 ground truth to certify
*against* — closing that needs the f64 pipeline extended, not more governance.

---

## FASE 6 — Reality Check

**Can Atenia run an arbitrary model brought today?**

The decision tree the loader actually executes:

1. Is the format safetensors or GGUF? If `.bin`/GPTQ/AWQ/FP8 → **No** (convert first).
2. Is `architectures[0]` one of the 7 known names **or** `LlamaForCausalLM`? If
   not → **No** (`resolve_adapter → None`, hard reject).
3. Is it MoE? → **Only Qwen-MoE** behind an opt-in flag, slow (disk-tier); Mixtral unrun; others no.
4. Is it a plain dense decoder of a supported family? → **Yes**, and for
   Llama/Qwen with high numerical confidence.

**Estimated ecosystem coverage:**
- **By popularity-weighted demand (what people actually download/deploy):
  ~55–65 %.** The supported families (Llama, Qwen, Mistral, Gemma, Phi) plus the
  large "declared-as-Llama" tail dominate real-world open-weight usage; Atenia
  hits the dense text-gen case for all of them.
- **By raw architecture diversity on HF: ~25–35 %.** Long tail of unsupported
  decoders (Falcon, InternLM, OLMo, Cohere, StableLM, GPT-NeoX, StarCoder, MPT,
  DeepSeek-dense, PhiMoE) and the entire non-decoder world (vision/multimodal,
  SSM/Mamba, encoder-decoder T5/BART, embedding/rerank, BERT-class) pulls this down.
- **By format: ~70 %** of supported-family checkpoints ship as safetensors/GGUF
  (✅); the `.bin`/quantized-pack minority is blocked.

**Honest framing:** Atenia is **deep, not wide**. On the narrow slice it targets
(dense Llama-lineage text generation with auditable numerics on 8 GB VRAM) it is
strong and certified. Outside that slice it declines fast.

---

## FASE 7 — Top 10 Gaps (real, not theoretical)

Ordered by how often they would actually block a real user:

1. **Hard reject of unknown `architectures`.** No graceful "try as generic
   decoder." One unrecognized arch string = dead stop, even when the topology is
   Llama-identical. *(Highest real-world friction.)*
2. **No PyTorch `.bin`/`.pth` loader.** A large share of HF repos are .bin-first;
   users must convert externally.
3. **No GPTQ/AWQ/bitsandbytes pre-quant ingestion.** The most-downloaded "small
   VRAM" community formats can't be read directly.
4. **MoE is effectively Qwen-MoE-only and slow.** Mixtral (the most-deployed MoE)
   has never run on real weights; per-token disk-tier latency (~min/token) is not
   interactive.
5. **No f64 certification for Phi-3 / Gemma.** GREEN behaviourally but below the
   Llama/Qwen trust bar; the governance layer can't certify without a reference.
6. **DeepSeek (MLA + MoE) unfinished.** Detected, not tiered, not run — and
   DeepSeek is a top-demand family in 2025–26.
7. **No FP8 safetensors.** Modern checkpoints increasingly ship FP8; the reader
   rejects them.
8. **Falcon / InternLM / OLMo / Cohere / StableLM absent.** Each is a distinct,
   non-Llama block (parallel attention, partial rotary, etc.) needing real
   graph-builder work — ATKv2's DSL names them but can't execute them.
9. **No vision/multimodal, SSM/Mamba, encoder-decoder, or embedding/rerank.**
   Entire model classes out of scope (acceptable, but it caps "any model" claims).
10. **Single-file safetensors loads whole into RAM (no mmap).** Large dense
    single-file checkpoints stress RAM where mmap would not.

---

## FASE 8 — ROI Ranking (NOW / NEXT / LATER / NEVER)

**NOW** (high demand × low-moderate effort × unblocks many models):
- **Generic-decoder fallback + architecture allowlist widening.** Let a config
  opt-in (or a known-compatible-arch table: Yi, Vicuna, SmolLM, Llama-distills)
  route through the Llama adapter *deliberately and tested*, instead of either a
  silent fallback or a hard reject. Converts gap #1 into coverage cheaply.
- **f64 reference pipeline for Phi-3 + Gemma 2.** Pure correctness ROI; turns two
  behavioural-GREEN families into certified, and makes NUMERIC-POLICY-3
  governance meaningful for them.

**NEXT** (high demand × moderate effort):
- **PyTorch `.bin` loader** (or a documented, built-in convert step). Unblocks a
  broad slice of HF repos (gap #2).
- **GPTQ/AWQ read path** wired to the load path (the experimental CPU
  reconstruction already exists in `src/quant/`). Unblocks the small-VRAM
  community tail (gap #3).
- **Mixtral real-weight run** (the loader is unblocked; needs weights +
  validation) and a usable MoE speed story (the measured truth is it's
  I/O-bound; the real lever is the AV-exclusion + fewer-bytes work, already
  scoped).

**LATER** (real but lower demand or higher effort):
- DeepSeek MLA + MoE productionisation (gap #6) — high demand, but large
  architecture + tiering effort.
- FP8 safetensors reader (gap #7).
- One genuinely new non-Llama family via ATKv2→v1 (Falcon or InternLM) as the
  *proof* that the toolkit can add breadth (gap #8).
- mmap single-file loads (gap #10).

**NEVER** (out of charter, or measured dead-ends):
- Vision/multimodal, SSM/Mamba, encoder-decoder, embedding/rerank — different
  engine, not this charter.
- More expert-matmul / CUDA-offload / tier-consolidation for MoE speed —
  **measured dead-ends** (MOE-PERF-2/3, MOE-IO-1: matmul < 1 % of wall, GPU
  upload loses, consolidation makes AV scanning worse).
- int4 without evidence-gated certification.

---

## FASE 9 — Roadmap Recommendation (what to do after NUMERIC-POLICY)

The numeric architecture is now complete *and* governed. The next bottleneck on
*usefulness* is **breadth of intake**, not more precision work. Recommended
ordering:

1. **MODEL-INTAKE-1 — "say yes more often, safely."** A widened, tested
   architecture allowlist + an explicit, opt-in generic-Llama-decoder path with a
   compatibility check, replacing the silent fallback and the hard reject. This
   is the highest coverage-per-unit-effort move and directly attacks gap #1.
2. **CERTIFY-BREADTH-1 — f64 pipeline for Phi-3 + Gemma.** Lifts two families to
   the certified tier and makes the NP-3 governance real for them.
3. **FORMAT-INTAKE-1 — `.bin` and/or GPTQ-AWQ read path.** Broadens the formats a
   real user shows up with.
4. *(Operational, parallel, zero-code)* the AV-exclusion guidance from MOE-IO-1
   remains the biggest MoE-speed win.

Explicitly **not** recommended next: another MoE-perf compute milestone (measured
dead-end) or new exotic families before the intake/cert breadth above pays off.

---

## ENTREGA FINAL — Executive Summary

1. **Supported families (7 dense + 3 MoE recognized):** Llama, Qwen2, Qwen3,
   Mistral, Phi-3, Gemma 2, Gemma 3 (dense); Qwen-MoE, Mixtral, DeepSeek-MoE
   (MoE, experimental/opt-in).
2. **Certified families (full f64 bar):** **Llama, Qwen2.** Only these two.
3. **Partial families (real-run/structural, no f64):** Phi-3 & Gemma 2
   (behavioural real-run), Qwen3, Mistral, Gemma 3 (structural), Qwen-MoE
   (experimental real-run).
4. **Missing families:** Falcon, Yi (only via untested Llama-fallback), InternLM,
   OLMo/OLMoE, Cohere/Command-R, StableLM, GPT-2/NeoX, StarCoder, MPT,
   DeepSeek-dense, PhiMoE — and all non-decoder classes (vision/multimodal,
   SSM/Mamba, encoder-decoder, embedding/rerank).
5. **Estimated real coverage:** ~**55–65 %** of popularity-weighted demand;
   ~**25–35 %** of raw HF architecture diversity; ~**70 %** of supported-family
   checkpoints by format. **Deep, not wide.**
6. **Adapter Toolkit state:** v1 registry = 7 complete adapters, no dead/redundant
   ones, good factoring (Qwen2/Mistral reuse Llama). ATKv2 = a solid declarative
   DSL/validator/introspect layer that **names but does not yet execute new
   families** — the DSL is ahead of the backend; new breadth still needs v1
   graph-builder work.
7. **Top 10 gaps:** unknown-arch hard reject · no `.bin` loader · no GPTQ/AWQ/bnb
   intake · MoE = Qwen-only & slow · no f64 cert for Phi/Gemma · DeepSeek/MLA
   unfinished · no FP8 · Falcon/InternLM/OLMo/Cohere/StableLM absent · no
   multimodal/SSM/enc-dec/embeddings · single-file loads fully into RAM.
8. **Top 10 priorities (ROI):** NOW — widened/tested arch allowlist + opt-in
   generic decoder; f64 pipeline for Phi/Gemma. NEXT — `.bin` loader; GPTQ/AWQ
   read path; Mixtral real run + AV-exclusion op-win. LATER — DeepSeek MLA/MoE;
   FP8; one new ATKv2→v1 family; mmap loads.
9. **Risks:** (a) the **silent Llama-fallback** runs unknown checkpoints with
   *assumed* Llama topology — correctness risk if they deviate; (b) two
   "supported" families (Phi/Gemma) are **uncertified** against f64; (c) MoE is
   **experimental + non-interactive** (min/token) and only one family is proven;
   (d) format friction (`.bin`/quant packs) blocks otherwise-supported families;
   (e) "supports many families" is easy to *over-claim* — the honest claim is
   "dense Llama-lineage text gen, certified on two."
10. **Strategic recommendation:** Atenia's moat is **auditable numerics on
    constrained hardware**, not breadth. Lead with that. Grow breadth *cheaply*
    (allowlist + generic decoder + `.bin`/quant intake) before building exotic
    families; lift Phi/Gemma to certified so the breadth you *do* claim is
    trustworthy. Don't chase MoE speed (measured dead-end) or multimodal (off
    charter).
11. **Next project:** **MODEL-INTAKE-1** — widen what loads, safely (tested
    architecture allowlist + opt-in generic-Llama-decoder with a compatibility
    check), immediately followed by **CERTIFY-BREADTH-1** (f64 pipeline for Phi-3
    + Gemma). Highest coverage-and-trust per unit effort, and it makes the
    just-finished NUMERIC-POLICY-3 governance pay off across more of the catalog.

---

*Audit only. No code changed, no commits, no CI. Sources: `src/model_adapters/`,
`src/adapter_toolkit/`, `src/moe/`, `src/v17/loader/`, and [STATUS.md](./STATUS.md)
(RUNTIME-REAL-1..4, RUNTIME-MOE-1/2, MOE-PROD-1..8, MOE-PERF-1..3, MOE-IO-1,
NUMERIC-POLICY-1..3).*
