# HANDOFF — RUNTIME-REAL-3: Gemma 2 end-to-end validation

Milestone: **RUNTIME-REAL-3** — validate load + real generation on a **Gemma 2**
checkpoint, the most structurally distinct supported family so far. **Validation
only**: no new architecture, families, math, or graph ops. Predecessors:
`0b7b03a` (Llama GREEN), `582e519` (Qwen2.5 GREEN).

## FASE 1 — Audit

**Gemma 2 support.** `Gemma2ForCausalLM` is a first-class supported family. The
adapter routes it to the **gemma2 builder** (`src/nn/llama/gemma2.rs`), which
extends the Llama graph with Gemma-2-specific nodes — **no new family**, all
realised with the existing `SoftCap`/`Scale`/norm graph ops:

| Gemma-2 feature | Where | Value (2B) |
|---|---|---|
| Attention logit softcap | `gb.soft_cap(scores, cap)` (gemma2.rs:256) | 50.0 |
| Final logit softcap | `gb.soft_cap(logits, cap)` (gemma2.rs:435) | 30.0 |
| Embedding scale ×√hidden | `embed_scale_or_passthrough` (gemma2.rs:672) | √2304 = 48.0 |
| Query pre-attn scalar | `1/√query_pre_attn_scalar` (gemma2.rs:711) | 1/√256 |
| Dual norms (pre+post) | gemma2 builder | per layer |
| GeGLU (`gelu_pytorch_tanh`) | gemma2 builder | — |

**Checkpoint chosen:** `models/gemma-2-2b-it/` (sharded safetensors, 2 shards,
~4.87 GiB source). 2B fits the 8 GB host via VRAM+RAM tiering; 9B is also present
but heavier.

| Property | Value |
|---|---|
| Architecture | `Gemma2ForCausalLM` (GQA, softcaps, scaled-embed, dual-norm) |
| safetensors | sharded: 4.99 GB + 0.24 GB (~4.87 GiB source) |
| hidden / layers | 2304 / 26 |
| head_dim | 256 (independent of hidden) |
| heads | 8 q-heads, 4 kv-heads (GQA 2:1) |
| intermediate / vocab | 9216 / **256000** |
| eos_token_id | **[1, 107]** (multiple EOS incl. `<end_of_turn>`) |
| sliding_window | 4096 (hybrid; **deferred** — see FASE 5) |
| dtype | bf16 |
| numeric contract | `model.numcert.json` (family `gemma2`) |

**Features this validates (FASE 5):** logit softcap (30), attention softcap
(50), scaled embeddings (×48), query_pre_attn_scalar, GeGLU, dual-norm, and the
multiple-EOS `[1,107]` stop set.

**Risks identified:**
1. **No f64 reference for Gemma 2** — the numcert sidecar carries
   `max_abs_diff_vs_f64: null`; its notes state the PyTorch f64 generation
   pipeline is not wired for Gemma 2 (the committed f64 fixture is Llama-family
   only). So a strict Atenia-vs-f64 comparison is **not available** here. (Not a
   regression — it was never generated.)
2. Sharded loading (2 shards via `model.safetensors.index.json`).
3. Very large vocab (256000) → large embedding/logit tensors, slow load.
4. Softcap numerical saturation must stay finite.
5. Sliding-window attention is **deferred** (full causal attention used for
   context < 4096) — fine for short prompts; a known simplification.

**Prior evidence already in-tree:** numcert sidecar (certified vs fast
**bit-identical** decoded text on the M11.C reference prompt; no NaN/Inf);
`SoftCap` node regression suite in `src/amg/builder.rs` (CI lib tests);
STATUS.md records a prior "8/8 in-scope PASS" across the Gemma 2 + Gemma 3
generation scope. There is **no** Gemma integration test or f64 fixture in
`tests/`.

## FASE 2-3 — Preparation + real load

Host: RTX 4070 Laptop (sm_89), 8 GB VRAM, ~32 GB RAM, Windows 11, CUDA build.

```
Architecture: Gemma2ForCausalLM - routing to gemma2 adapter
  (dual-norm, GeGLU, SoftCap@50/30, embedding scale;
   sliding-window deferred - full causal attention for context < 4096)
Adapter residency: adapter=gemma2 family=Gemma2 ... gemma2_softcaps=true
Tier planner: source 4.87 GiB, all-resident 9.74 GiB @ F32, free VRAM 7.76 GiB,
              VRAM budget 3.76 GiB, GPU-eligible 182/288 tensors
Tier-aware loader plan:  VRAM 96 tensors (3.76 GiB) | RAM 192 (2.99 GiB) | Disk 0
model loaded in 89.6s
```

- **288 tensors** across 2 shards; gemma2 softcaps confirmed active.
- **Memory used:** ~6.75 GiB resident (VRAM 3.76 + RAM 2.99). F32 all-resident
  estimate (9.74 GiB) exceeds VRAM, so the planner correctly split VRAM/RAM.
- **Load time:** ~89.6 s (sharded + 256000-vocab embedding + mixed VRAM/RAM).

## FASE 4 — Real generation (greedy)

| Prompt (mode) | Output | tok/s | EOS |
|---|---|---|---|
| `What is the capital of France?` (chat) | **`The capital of France is **Paris**. 🇫🇷`** | 0.53 | **yes** (14 tok, stop=107) |
| `Hello` (chat) | `Hello! 👋  How can I help you today? 😊` | 0.53 | **yes** (16 tok) |
| `Rust is a programming language that` (raw) | `is gaining popularity for its speed, reliability, and versatility. It is used in a wide range of applications, including web development, game development` | 0.57 | no (max) |
| `Explain what a database is.` (chat) | `Imagine a giant, organized library filled with information about everything you could possibly want to know. That's essentially what a database is!` | 0.53 | no (max) |

Coherent, fluent, factually correct, characteristic Gemma markdown/emoji style;
no corruption, no NaN/Inf, no repetition loops. Throughput is low (~0.5 tok/s)
because ~⅔ of tensors are RAM-tiered on the 8 GB host and roundtrip per step —
correctness-first, not optimised.

## FASE 5 — Gemma features confirmed at runtime

- **Softcaps execute** (`gemma2_softcaps=true`; attention cap 50, logit cap 30):
  coherent finite logits, no NaN/Inf across all runs. Backed by the `SoftCap`
  regression suite (8/8, below).
- **Scaled embeddings** (×√2304 = 48) and **query_pre_attn_scalar** (1/√256) are
  on the active path — correct factual output ("Paris") depends on them.
- **GeGLU + dual-norm** active (adapter banner).
- **Multiple-EOS [1,107]:** the capital + Hello prompts both stopped on
  `<end_of_turn>`=107 — multi-EOS stop set handled correctly.
- **Sliding window:** deferred to full causal attention for context < 4096
  (documented simplification; all test prompts are well under 4096, so behaviour
  is exact for these).

## FASE 6 + 8 — HF comparison + certification (honest scope)

**A strict f64 comparison is not available for Gemma 2** — the numcert sidecar
explicitly records no f64 reference (`max_abs_diff_vs_f64: null`) because the
PyTorch f64 generation pipeline was never wired for this family. **Not
fabricated here.** Available certification evidence:

- numcert: certified vs fast modes produced **bit-identical decoded text** on
  the M11.C reference prompt; no NaN/Inf in logits.
- `SoftCap` regression: **8/8 pass** — `soft_cap_saturates_finite_no_nan`
  (inputs to ±f32::MAX/2, cap 50 → finite saturated), `soft_cap_gemma2_attention_shape`,
  `soft_cap_at_cap_equals_cap_times_tanh_one`, `soft_cap_identity_at_zero`,
  builder rejects zero/NaN/negative cap, shape-preservation.
- **Determinism:** two greedy runs → **identical 14 token ids** (stop=107).

This is **weaker** than Llama (0.076 f64) and Qwen2.5 (0.000335 f64): Gemma 2 has
no committed f64 ground truth. Coherence + determinism + softcap saturation are
the available bars.

## FASE 7 — Robustness

| Case | Result |
|---|---|
| empty prompt (chat) | no crash, generates, exit 0 |
| short prompt (`Hello`) | clean, EOS=107 |
| moderate prompt (`Explain…`) | coherent, stable, no NaN |
| `--max-tokens 0` | rejected, exit 2 |
| EOS | confirmed (multi-EOS [1,107], stop=107) |
| determinism | identical token ids across runs |

## FASE 11 — Final validation (real, exit 0)

- `cargo test --release --lib soft_cap` → **8/8 pass** (CI-blocking lib tests).
- Real CLI generation: 4 prompts coherent; robustness 5/5; determinism
  confirmed; multi-EOS stop confirmed.
- No Gemma integration/f64 test exists in `tests/` to run (documented gap).

## Problems found

**None.** Gemma 2 validated cleanly with no unexpected behaviour and no code
change. The one honest limitation (no f64 reference) is pre-existing and
documented, not introduced here.

## FASE 10 — Strategic review

**Can Gemma-family be considered GREEN?** → **GREEN with a caveat.**

GREEN for Gemma 2 dense text generation **behaviourally**: real sharded 4.87 GiB
checkpoint loads + VRAM/RAM tiers correctly; the full Gemma-2 feature set
(softcaps, scaled embed, query scalar, GeGLU, dual-norm, multi-EOS) executes and
produces coherent, factually-correct, deterministic text; robust to bad input;
softcap saturation regression-locked.

**Evidence:** real CLI generation (4 prompts), determinism, SoftCap 8/8,
certified↔fast bit-identical (numcert), prior 8/8 generation scope.

**What's missing (the caveat):**
- **No f64 numerical certification** for Gemma 2 (numcert null; pipeline not
  wired). Llama/Qwen have it; Gemma does not. This is the one thing keeping it
  short of the Llama/Qwen certification bar.
- Sliding-window attention deferred (exact only for context < 4096).
- Only 2B exercised this pass (9B present, not re-validated).
- Throughput very low on this host (~0.5 tok/s, RAM-tiered).

## Files modified

- `docs/HANDOFF_RUNTIME_REAL_3.md` — this file.
- `docs/STATUS.md` — RUNTIME-REAL-3 evidence note.

No production code, tests, architecture, families, math, or graph ops changed.

## Deliverable answers

1. **Loaded correctly?** Yes — 288 tensors / 2 shards, ~6.75 GiB resident,
   VRAM/RAM tiered, softcaps active.
2. **Coherent text?** Yes — "…is **Paris**. 🇫🇷", correct Rust/database
   explanations, clean EOS (107).
3. **Important differences vs HF?** No f64 reference available for Gemma 2
   (documented); behaviourally coherent + deterministic, certified↔fast
   bit-identical, no NaN/Inf.
4. **Problems found?** None.
5. **Memory used?** ~6.75 GiB resident (VRAM 3.76 + RAM 2.99).
6. **Load time?** ~89.6 s (sharded, 256000-vocab, mixed tiering).
7. **Generation time?** ~0.5–0.6 tok/s greedy (RAM-tiered; correctness-first).
8. **Gemma-family → GREEN?** GREEN behaviourally; caveat: no f64 certification.
9. **Files modified:** see above.
10. **Commit:** see git log (RUNTIME-REAL: Gemma2 end-to-end validation).
11. **CI:** docs-only commit → CI skipped by design (`paths-ignore`).
12. **Next recommendation:** see below.

## Next recommendation

Two reasonable paths: (a) **close the dense breadth campaign** — three
structurally-distinct families (Llama, Qwen2.5, Gemma 2) now validated
end-to-end, which is strong breadth-at-scale evidence; or (b) **one more family**
(Phi-3, distinct partial-rotary + fused-QKV) for a 4th data point. If numerical
rigour matters for Gemma specifically, a separate scoped task could wire the
PyTorch f64 generation pipeline for Gemma 2 to lift it to the Llama/Qwen
certification bar — but that is **not** a validation task (it adds reference
infra) and should be its own milestone.
