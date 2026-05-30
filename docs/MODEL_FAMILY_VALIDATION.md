# Model Family Validation

This document records the **functional family validation** carried
out for Atenia Engine before the Adapter Toolkit v2 work — the
"family mastery" batteries run per family.

## What this document is

This is a record of **functional validation**, also called *family
mastery*: each family was exercised end to end (load → 1-turn
greedy generation → EOS) across a representative set of real
checkpoints, formats and quantisations, on the dev box (RTX 4070
Laptop, 8 GB VRAM, certified mode).

**This is not numeric certification.** "Functional / family
validation" means *the model loads and generates coherent text and
stops correctly*. It is distinct from the ADR-004 numeric
certification (`numcert`), which is a strict per-checkpoint
numeric-drift gate with a versioned manifest. Only the four ADR-004
fixture models carry numcert manifests; every other entry below is
functional validation only.

Terminology used here — "family validation", "functional
validation", "mastery battery" — is deliberate. The word
"certified" is reserved for ADR-004 numcert and is **not** claimed
for the GGUF quants validated functionally.

**Family validation ≠ AQS quantization evaluation.** This document is
about *which model families load and generate correctly*. It is unrelated
to **AQS (Atenia Quantization Search)**, the experimental subsystem that
evaluates *quantization policies* (BF16 / INT8 / AWQ / Hybrid / GPTQ) on a
model against the F64 reference. AQS is CPU-only, opt-in, experimental, and
not production certification — see [AQS_OVERVIEW.md](./AQS_OVERVIEW.md).

**Mixture-of-experts remains out of scope.** No MoE execution path exists;
Mixtral / Mistral-MoE are not supported and fail loud. DeepSeek-R1 distill
checkpoints are dense Llama/Qwen derivatives (not MoE) and validate under
their base family.

## Per-family results

Legend — Result: PASS = loaded and generated coherent text,
stopped on EOS. Format: ST = HF safetensors, GGUF = quantised GGUF.

### Llama

| Checkpoint | Format | Quants | Result |
|------------|--------|--------|--------|
| TinyLlama 1.1B Chat | ST + GGUF | Q4_K_M, Q8_0 | PASS |
| Llama 2 13B Chat | ST | — | PASS |
| Llama 3.2 1B Instruct | ST | — | PASS |
| Llama 3.2 3B Instruct | ST | — | PASS |
| Llama 3.1 8B Instruct | ST + GGUF | Q4_K_M, Q5_K_M, Q6_K | PASS |
| DeepSeek-R1-Distill-Llama 8B | GGUF | Q4_K_M | PASS |

Notes / gaps:
- Llama 2 7B is **gated** on HuggingFace (license acceptance) — not
  downloaded, not validated. It is pure Llama topology and would
  route through the same path.
- TinyLlama 1.1B is one of the four ADR-004 numcert fixtures.

### Qwen

| Checkpoint | Format | Quants | Result |
|------------|--------|--------|--------|
| Qwen2.5 1.5B Instruct | ST | — | PASS |
| Qwen2.5 3B Instruct | ST | — | PASS |
| Qwen2.5 7B Instruct | ST + GGUF | Q4_K_M, Q5_K_M, Q6_K | PASS |
| Qwen2.5-Coder 1.5B Instruct | ST | — | PASS |
| Qwen3 0.6B | ST | — | PASS |
| Qwen3 4B | ST | — | PASS |
| Qwen3 8B | GGUF | Q4_K_M | PASS |
| DeepSeek-R1-Distill-Qwen 7B | GGUF | Q4_K_M | PASS |

Real fixes made during Qwen validation:
- Qwen3 tied `lm_head` handling (the Qwen3 0.6B/4B checkpoints tie
  the LM head to the embedding; the builder's tied branch was
  corrected).
- GGUF architecture whitelist extended to `qwen2` / `qwen3`.
- Qwen3 per-head QK-Norm γ tensor name mapping + transform rules.

### Gemma

| Checkpoint | Format | Quants | Result |
|------------|--------|--------|--------|
| Gemma 2 2B IT | ST + GGUF | Q4_K_M | PASS |
| Gemma 2 9B IT | ST + GGUF | Q4_K_M, Q5_K_M, Q6_K | PASS |
| Gemma 3 1B IT | ST + GGUF | Q4_K_M | PASS |
| Gemma 3 4B IT | GGUF | Q4_K_M | PASS (text path) |

Real fixes made during Gemma validation:
- Multi-EOS support (`eos_token_ids` set; Gemma instruct ends a
  turn on `<end_of_turn>`).
- Q5_0 GGUF quant decoder added (the ggml-org Gemma 3 conversion
  mixes Q5_0 into a nominally-Q4_K_M file).
- Gemma 3 text graph builder (`build_gemma3`).

Out of scope:
- Gemma 3 4B **safetensors** ships as a multimodal wrapper
  (`Gemma3ForConditionalGeneration`); the vision tower is out of
  scope. The text path was validated via the GGUF text export.

### Phi

| Checkpoint | Format | Quants | Result |
|------------|--------|--------|--------|
| Phi-3 mini 4k Instruct | ST | — | PASS |
| Phi-3 mini 128k Instruct | ST | — | PASS |
| Phi-3.5 mini Instruct | ST + GGUF | Q4_K_M, Q5_K_M, Q6_K | PASS |
| Phi-4 mini Instruct | ST + GGUF | Q4_K_M | PASS |

Real fixes made during Phi validation:
- Optional LongRoPE: a plain-RoPE Phi-3 checkpoint no longer
  panics when the `rope_scaling` block is absent.
- Phi-4 partial rotary factor handling.
- Phi-4 GQA fused-QKV expansion.
- Turn terminator `<|end|>` wired into the multi-EOS resolution.

### Mistral (dense)

| Checkpoint | Format | Quants | Result |
|------------|--------|--------|--------|
| Mistral 7B v0.3 base | ST | — | PASS |
| Mistral 7B Instruct v0.3 | ST + GGUF | Q4_K_M, Q5_K_M, Q6_K | PASS |
| Mistral 7B Instruct v0.2 | ST + GGUF | Q4_K_M | PASS |

Validation-only phase — Mistral dense is pure Llama topology (GQA,
SwiGLU, RMSNorm, RoPE, no QKV bias, untied LM head); no source
change was needed.

Out of scope:
- Mixtral / Mistral-MoE — there is no mixture-of-experts code path
  and none was added.

### SmolLM

| Checkpoint | Format | Quants | Result |
|------------|--------|--------|--------|
| SmolLM2 135M Instruct | ST | — | PASS |
| SmolLM2 360M Instruct | ST | — | PASS |
| SmolLM2 1.7B Instruct | ST + GGUF | Q4_K_M, Q5_K_M, Q6_K | PASS |
| SmolLM 135M Instruct | ST | — | PASS |
| SmolLM 360M Instruct | ST | — | PASS |
| SmolLM 1.7B Instruct | ST | — | PASS |

Validation-only phase — SmolLM / SmolLM2 declare
`architectures = ["LlamaForCausalLM"]` / `model_type = "llama"` and
are classified as **Llama-compatible**; they resolve to the v1
Llama adapter with no SmolLM-specific code. SmolLM2 1.7B is an
ADR-004 numcert fixture. Falcon3-7B-Instruct was smoked here as an
optional regression of the Llama path.

Notes:
- The small (135M / 360M) instruct models are verbose / mildly
  repetitive under greedy decoding — a model-capacity property,
  not an engine defect; the first sentence is correct in every
  case.

### Falcon

| Checkpoint | Format | Quants | Result |
|------------|--------|--------|--------|
| Falcon3 1B Instruct | ST | — | PASS |
| Falcon3 3B Instruct | ST | — | PASS |
| Falcon3 7B Instruct | ST + GGUF | Q4_K_M, Q5_K_M, Q6_K | PASS |
| Falcon 7B Instruct (classic) | ST + GGUF | Q4_K_M | Out of scope — see below |

Real fix made during Falcon validation:
- `bos_token_id` fallback: Falcon3-1B / Falcon3-3B omit
  `bos_token_id` from `config.json` (it lives only in
  `generation_config.json`); the config parser now falls back to
  `eos_token_id` when the field is absent.

Classic Falcon classification:
- Classic Falcon (`FalconForCausalLM` / `RWForCausalLM`) uses
  LayerNorm (not RMSNorm), parallel attention, and a multi-query
  fused-QKV layout — a genuinely different architecture with no v1
  graph builder. It is **classified out of scope**, not forced
  onto the Llama path.
- Both classic paths fail loud cleanly: the safetensors load fails
  with a typed config error (missing `num_key_value_heads`); the
  GGUF load fails with `unsupported general.architecture =
  "falcon"`. These are correct fail-loud rejections, not crashes.

Verdict: **Falcon3 validated; classic Falcon out of scope.**

## What this validation does NOT mean

To stay honest about scope, this validation explicitly does **not**
claim:

- **No MoE.** Mixture-of-experts models (Mixtral, Mistral-MoE) are
  not supported; no MoE code path exists.
- **No multimodal.** Vision / multimodal models are not supported.
  Gemma 3 4B was validated only on its text path.
- **No classic Falcon.** `FalconForCausalLM` / `RWForCausalLM` is
  out of scope (distinct architecture, no graph builder).
- **No full numcert for every GGUF.** Only the four ADR-004
  fixture models carry numeric-certification manifests. Every GGUF
  quant above is functionally validated, not numerically
  certified.
- **No performance certification.** These results say a model
  loads and generates coherently. They say nothing about
  throughput, latency, or memory-tier behaviour beyond "it ran on
  the 8 GB dev box".
- **No exhaustive checkpoint coverage.** One representative set
  per family was validated; not every published checkpoint or
  every quant of every size.

## See also

- `docs/STATUS.md` — current engine status, including the
  per-family mastery battery summaries.
- `docs/MILESTONES.md` — full milestone history with the
  per-family validation sections.
- `docs/ADAPTER_TOOLKIT_V2.md` — the declarative adapter toolkit
  built on top of the validated v1 family adapters.

## MoE families — experimental validation only (not product-certified)

The Mixture-of-Experts experimental track (MOE-0 → MOE-18, closed MOE-19)
validated **tiny** MoE checkpoints through an isolated, CPU-only, opt-in path —
this is **not** family mastery and **not** product certification.

- **Validated experimentally** (layer-0 MoE block, ~1e-10 HuggingFace
  numerical parity, argmax-matching):
  - `katuni4ka/tiny-random-qwen1.5-moe` (classic per-expert + shared expert)
  - `hf-internal-testing/tiny-random-Qwen2MoeForCausalLM` (packed experts)
  - `hf-internal-testing/tiny-random-Qwen3MoeForCausalLM` (packed experts, no
    shared, `mlp.router` naming — QWEN-MOE-CERT-1)
  - `hf-internal-testing/tiny-random-MixtralForCausalLM` (packed experts)
- **Qwen-MoE family — partially certified (experimental)** via
  **QWEN-MOE-CERT-1** (`docs/HANDOFF_QWEN_MOE_CERT_1.md`): Qwen1.5-MoE,
  Qwen2-MoE and Qwen3-MoE tiny checkpoints certified for MoE-block numerical
  parity with HuggingFace (~1e-10) across classic + packed experts, shared /
  no-shared, both `norm_topk_prob` modes and both router namings. **Not**
  product-certified (no real full model, no full transformer, fail-loud still
  active).
- **Mixtral family — partially certified (experimental)** via
  **MIXTRAL-CERT-1** (`docs/HANDOFF_MIXTRAL_CERT_1.md`): two real Mixtral
  checkpoints certified for MoE-block numerical parity with HuggingFace under
  the Atenia convention — `tiny-random-Mixtral` (packed, 4 experts, 1.164e-10,
  committed CI fixture) and `TitanML/tiny-mixtral` (classic
  `block_sparse_moe.w1/w3/w2`, 8 experts, d_model 1024, 1.49e-8, local-only —
  ~352 MB fixture too large to commit). Both on-disk Mixtral layouts validated
  end-to-end on real data; no `src/` changes needed. **Not** product-certified
  (no Mixtral-8x7B, no full transformer, fail-loud active). No MoE GGUF support
  (gap → MIXTRAL-GGUF-1).
- **Not product-certified.** These are tiny random-weight test checkpoints, not
  real full models. No end-to-end generation, no full transformer path, no
  numcert manifest. The productive loader still **fails loud** on MoE.
- **Future certification candidates.** Qwen1.5-MoE-A2.7B / Qwen2-MoE and
  Mixtral 8x7B are the natural first targets once the Adapter Toolkit MoE spec,
  full transformer path, and a large-model memory strategy land. See
  `docs/MOE_OVERVIEW.md` for the exact production-blocker list.
