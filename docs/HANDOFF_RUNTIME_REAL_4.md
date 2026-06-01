# HANDOFF — RUNTIME-REAL-4: Phi-3 end-to-end validation

Milestone: **RUNTIME-REAL-4** — validate load + real generation on a **Phi-3**
checkpoint, the 4th structurally-distinct supported family, before closing the
Dense breadth campaign. **Validation only**: no new architecture, families,
math, or graph ops. Predecessors: `0b7b03a` (Llama), `582e519` (Qwen2.5),
`f755f3f` (Gemma 2).

## FASE 1 — Audit

**Phi-3 support.** `Phi3ForCausalLM` is a first-class supported family. The
adapter (`src/model_adapters/phi3.rs`) routes to the **phi3 builder**
(`src/nn/llama/phi3.rs`) with `fused_qkv_weight_mapping: true` +
`fused_gate_up_weight_mapping: true`. It extends the Llama graph with Phi-3
specifics — **no new family**, realised with existing ops:

| Phi-3 feature | Where |
|---|---|
| Fused **QKV** split (`qkv_proj` → q,k,v) | `split_fused_qkv` (phi3.rs:71) |
| Fused **gate_up** split (`gate_up_proj` → gate,up) | `split_fused_gate_up` (phi3.rs:132) |
| **LongRoPE** (`short_factor`/`long_factor` per-dim) | `nn::rope` longrope path |
| Untied lm_head, MHA (no GQA), SiLU MLP | shared Llama path |

Adapter banner at load: *"Phi3ForCausalLM - routing to phi3 adapter (LongRope +
fused QKV / gate_up split via SliceLastDim)."*

**Checkpoint chosen:** `models/phi-3.5-mini-instruct/` (sharded safetensors, 2
shards, ~7.6 GB). It is the Phi-3 family member that **carries a numcert
sidecar** (the plain phi-3-mini-4k / 128k do not), so it gives the most
certification evidence. Phi-3.5-Mini is `Phi3ForCausalLM` with LongRoPE.

| Property | Value |
|---|---|
| Architecture | `Phi3ForCausalLM` (MHA, fused QKV, fused gate_up, LongRoPE) |
| safetensors | sharded: 4.97 GB + 2.67 GB (~7.1 GiB source) |
| hidden / layers | 3072 / 32 |
| heads | 32 q-heads, 32 kv-heads (**MHA, no GQA**) |
| intermediate / vocab | 8192 / 32064 |
| rope_scaling | **longrope** (short/long factor, 48 dims) |
| partial_rotary_factor | **null** (full RoPE — see FASE 5) |
| tie_word_embeddings | false |
| eos_token_id | 32000 |
| dtype | bf16 |
| numeric contract | `model.numcert.json` (family `phi3-longrope`) |

**Features this validates (FASE 5):** fused QKV, fused gate_up, LongRoPE, and the
phi3-only adapter mapping.

**Risks identified:**
1. **No f64 reference for Phi-3** — numcert carries `max_abs_diff_vs_f64: null`;
   its notes state the PyTorch f64 generation pipeline is not wired for Phi-3
   (the committed f64 fixture is Llama-family only). Strict Atenia-vs-f64 is
   **not available** (same as Gemma 2). Not a regression — never generated.
2. **3.8B / 7.6 GB** exceeds 8 GB VRAM → heavy RAM tiering, slow generation.
3. Sharded loading (2 shards).
4. LongRoPE per-dimension scaling correctness.
5. Fused-weight splits must reconstruct q/k/v and gate/up exactly.

**Prior evidence already in-tree:** numcert sidecar (certified vs fast
**bit-identical** argmax + decoded text on the M11.B reference prompt); a large
phi3 unit-test suite in `src/nn/llama/phi3.rs` + `model_adapters` +
`nn::rope::longrope_scaling_tests` (CI lib tests); STATUS.md records a prior
Phi-3.5 generation smoke. There is **no** Phi integration test or f64 fixture in
`tests/`.

## FASE 2-3 — Preparation + real load

Host: RTX 4070 Laptop (sm_89), 8 GB VRAM, ~32 GB RAM, Windows 11, CUDA build.

```
Architecture: Phi3ForCausalLM - routing to phi3 adapter
  (LongRope + fused QKV / gate_up split via SliceLastDim)
Adapter residency: adapter=phi3 family=Phi3 ... fused_qkv=true fused_gate_up=true
Tier-aware loader plan:  VRAM 32 tensors (3.74 GiB) | RAM 163 (5.25 GiB) | Disk 0
model loaded in 137.5s
```

- **195 tensors** across 2 shards; fused QKV + fused gate_up confirmed active.
- **Memory used:** ~8.99 GiB resident (VRAM 3.74 + RAM 5.25). 3.8B at F32
  exceeds VRAM, so the planner correctly RAM-tiered the bulk (163/195 tensors).
- **Load time:** ~137.5 s (sharded + 3.8B + fused-split + heavy RAM tiering).

## FASE 4 — Real generation (greedy)

| Prompt (mode) | Output | tok/s |
|---|---|---|
| `What is the capital of France?` (chat) | **`The capital of France is Paris. It is not only the largest city in France but also a major European city known for its art, fashion, g`** | 0.26 |
| `Hello` (chat) | `Hello! I'm Phi, an AI language model. How can` | 0.20 |
| `Rust is a programming language that` (raw) | `is known for its performance and safety. It is a systems programming language that is used for developing operating systems, embedded systems, and high-` | 0.26 |
| `Explain what a database is.` (chat) | `A database is an organized collection of data that is stored and accessed electronically. Databases are designed to manage, store, and retrieve large amounts of information efficiently.` | 0.22 |

Coherent, fluent, factually correct, no corruption or repetition. Phi-3.5 is
verbose, so none of these short-budget runs hit EOS (the model keeps elaborating)
— see EOS note below. Throughput is very low (~0.2–0.26 tok/s) because 163/195
tensors are RAM-tiered on the 8 GB host and roundtrip per step — correctness
first, not optimised.

## FASE 5 — Phi features confirmed at runtime

- **Fused QKV split** (`fused_qkv=true`): the single `qkv_proj` matrix is split
  into q/k/v; correct attention (→ correct factual output) confirms the split.
  Unit-locked by `split_qkv_*` (round-trip concat == input, phi3.5 shape).
- **Fused gate_up split** (`fused_gate_up=true`): single `gate_up_proj` split
  into gate/up; coherent MLP output confirms it. Unit-locked by `split_gate_up_*`.
- **LongRoPE** active (banner + `longrope_attention_factor_phi35_mini` test):
  per-dimension short/long factor RoPE; coherent long-range text.
- **Partial rotary:** **does not apply** to Phi-3 (`partial_rotary_factor: null`
  → full RoPE). Partial rotary is a Phi-1/Phi-2 feature; honestly out of scope
  for this checkpoint.

## FASE 6 + 8 — HF comparison + certification (honest scope)

**A strict f64 comparison is not available for Phi-3** — numcert records no f64
reference (`max_abs_diff_vs_f64: null`); the PyTorch f64 generation pipeline was
never wired for this family. **Not fabricated here.** Available evidence:

- numcert: certified vs fast modes produced **bit-identical argmax + decoded
  text** on the M11.B reference prompt; recommended mode `certified`.
- phi3 unit suite: **23/23 pass** — `split_qkv_round_trip_concat_matches_input`,
  `split_qkv_phi35_mini_shape`, `split_gate_up_phi35_mini_shape`,
  `phi3_adapter_maps_fused_qkv`, `fused_qkv_and_gate_up_are_phi3_only`,
  `longrope_attention_factor_phi35_mini`, `rope_scaling_longrope_on_non_phi3_errors`,
  reject malformed shapes/lengths.
- **Determinism:** two greedy runs → **identical 16 token ids**.

This is **weaker** than Llama (0.076) / Qwen2.5 (0.000335): Phi-3 has no
committed f64 ground truth (same situation as Gemma 2). Coherence + determinism +
fused-split/longrope unit locks are the available bars.

## FASE 7 — Robustness

| Case | Result |
|---|---|
| empty prompt (chat) | no crash, generates, exit 0 |
| short prompt (`Hello`) | clean, coherent |
| moderate prompt (`Explain…`) | coherent, stable |
| `--max-tokens 0` | rejected, exit 2 |
| EOS | shared greedy loop (already EOS-validated on Llama/Qwen/Gemma); not triggered in these short verbose runs — see note |
| determinism | identical token ids across runs |

> **EOS note (honest):** Phi-3.5-Mini is verbose; with the small token budgets
> used here it kept elaborating rather than emitting `</s>`/32000. The EOS stop
> mechanism is the same `generate_greedy` path that stopped cleanly on Llama
> (TinyLlama), Qwen2.5 (EOS in 8 tok), and Gemma 2 (stop=107). Not separately
> re-triggered for Phi-3 this pass.

## FASE 11 — Final validation (real, exit 0)

- `cargo test --release --lib phi3` → **23/23 pass** (CI-blocking lib tests:
  fused split, adapter mapping, LongRoPE).
- Real CLI generation: 4 prompts coherent; robustness 4/4; determinism
  confirmed.
- No Phi integration/f64 test exists in `tests/` to run (documented gap).

## Problems found

**None.** Phi-3 validated cleanly with no unexpected behaviour and no code
change. The one honest limitation (no f64 reference) is pre-existing and
documented.

## FASE 10 — Strategic review

**Can Phi-family be considered GREEN?** → **GREEN with a caveat** (same shape as
Gemma 2).

GREEN for Phi-3 dense text generation **behaviourally**: real sharded 7.6 GB
checkpoint loads + VRAM/RAM tiers correctly; fused QKV + fused gate_up + LongRoPE
execute and produce coherent, factually-correct, deterministic text; robust to
bad input; fused-split + LongRoPE unit-locked (23/23). **Caveat:** no f64
numerical certification (numcert null; pipeline not wired). EOS not separately
re-triggered (verbose model, short budgets).

**Can the Dense campaign be closed?** → **Yes — recommend closing it.** Four
structurally-distinct families are now validated end-to-end on real checkpoints:

| Family | Distinctive | Verdict | f64 cert |
|---|---|---|---|
| Llama (TinyLlama) | baseline GQA | GREEN | yes (0.076) |
| Qwen2.5 | QKV bias, tied embed, 151936 vocab | GREEN | yes (0.000335) |
| Gemma 2 | softcaps, scaled embed, dual-norm, multi-EOS | GREEN* | no (null) |
| Phi-3 (3.5-Mini) | fused QKV, fused gate_up, LongRoPE | GREEN* | no (null) |

\* behavioural GREEN; lacks committed f64 reference.

That is strong, diverse breadth-at-scale. The remaining gaps are **not new
validation** — they are: (1) wiring the PyTorch f64 generation pipeline for
Gemma 2 + Phi-3 to lift them to the Llama/Qwen numerical bar (a reference-infra
task, its own milestone); (2) throughput; (3) larger checkpoints / more
variants. None block the breadth story.

## Files modified

- `docs/HANDOFF_RUNTIME_REAL_4.md` — this file.
- `docs/STATUS.md` — RUNTIME-REAL-4 evidence note + Dense matrix close.

No production code, tests, architecture, families, math, or graph ops changed.

## Deliverable answers

1. **Loaded correctly?** Yes — 195 tensors / 2 shards, ~8.99 GiB resident,
   VRAM/RAM tiered, fused QKV + gate_up active.
2. **Coherent text?** Yes — "…is Paris.", correct Rust/database explanations.
3. **Important differences vs HF?** No f64 reference available for Phi-3
   (documented); behaviourally coherent + deterministic, certified↔fast
   bit-identical.
4. **Problems found?** None.
5. **Memory used?** ~8.99 GiB resident (VRAM 3.74 + RAM 5.25).
6. **Load time?** ~137.5 s (sharded 3.8B, heavy RAM tiering).
7. **Generation time?** ~0.2–0.26 tok/s greedy (RAM-tiered; correctness-first).
8. **Phi-family → GREEN?** GREEN behaviourally; caveat: no f64 certification.
9. **Dense campaign closeable?** Yes — 4 distinct families validated; recommend
   closing.
10. **Files modified:** see above.
11. **Commit:** see git log (RUNTIME-REAL: Phi3 end-to-end validation).
12. **CI:** docs-only commit → CI skipped by design (`paths-ignore`).
