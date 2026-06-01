# HANDOFF — RUNTIME-REAL-2: Qwen2.5 end-to-end validation

Milestone: **RUNTIME-REAL-2** — validate that the full runtime works end-to-end
on a **non-Llama** family, to raise breadth-at-scale beyond RUNTIME-REAL-1
(TinyLlama). **Validation only**: no new architecture, graph ops, math, or
supported families. Predecessor: `0b7b03a` (RUNTIME-REAL-1, dense Llama =
GREEN).

## FASE 1 — Audit

**Qwen support.** `Qwen2ForCausalLM` is a first-class supported family
(`atenia capabilities` lists `qwen2`). The adapter classifies Qwen2.5 as
**`llama-2-with-qkv-bias`**: it **reuses the Llama transformer graph** plus two
deltas handled by existing adapter machinery — **Q/K/V projection bias** and
**tied word embeddings** (`lm_head` shares `embed_tokens`). No Qwen-specific
graph/math. GQA k/v tiling, RoPE (θ=1e6), RMSNorm, SiLU MLP are the same ops as
Llama.

**Checkpoint chosen:** `models/qwen2.5-1.5b-instruct/` (single-file safetensors,
~2.88 GiB). Single-file → fast load; small enough to fully resident-tier on the
8 GB host; representative non-Llama family. (3B/7B/Coder-1.5B are also present
for future breadth.)

| Property | Value |
|---|---|
| Architecture | `Qwen2ForCausalLM` (dense, GQA, QKV-bias, tied embeddings) |
| `model.safetensors` | 3,087,467,144 bytes (~2.88 GiB source) |
| hidden / layers | 1536 / 28 |
| heads | 12 q-heads, 2 kv-heads (GQA 6:1) |
| intermediate / vocab | 8960 / **151936** |
| rope_theta | 1,000,000 |
| tie_word_embeddings | **true** |
| dtype | bf16 |
| tokenizer | `tokenizer.json` (7.0 MB, ChatML template) |
| numeric contract | `model.numcert.json` (family `llama-2-with-qkv-bias`) |

**Risks identified (all pre-handled, no surprises):** (1) tied embeddings —
`lm_head` reuses the embedding matrix (differs from TinyLlama's untied head);
(2) QKV bias path; (3) very large vocab (151936) → large logit tensors;
(4) rope_theta 1e6; (5) ChatML chat template differs from Llama. The adapter +
loader already cover all five.

**Adapter Toolkit parts involved:** family classification → `Qwen2`
(`llama-2-with-qkv-bias`); the Llama builder with QKV-bias hints + tied-lm_head
handling; `WeightMapper` GQA k/v `TileGroupedDim`; the M6 tier planner; the
numcert per-tensor policy stamper.

## FASE 2-3 — Preparation + real load

Host: RTX 4070 Laptop (sm_89), 8 GB VRAM, ~32 GB RAM, Windows 11, CUDA build.

```
Adapter residency policy: adapter=qwen2 family=Qwen2 ... lm_head=UntiedOnly
Tier planner: source 2.88 GiB, all-resident 2.88 GiB @ BF16, free VRAM 7.76 GiB,
              GPU-eligible 196/338 tensors
Tier-aware loader plan:  VRAM 196 tensors (2.44 GiB) | RAM 142 (0.43 GiB) | Disk 0
Numeric contract: per-tensor policy applied — 196 VRAM tensors stamped (196 fast, 0 certified)
model loaded in 12.6s
```

- **338 tensors** loaded (0 skipped, 0 missing). Tied embeddings: no separate
  `lm_head` tensor — embedding matrix reused.
- **Memory used:** ~2.87 GiB resident (VRAM 2.44 + RAM 0.43); BF16 storage path.
- **Load time:** ~10.6–12.6 s (single-file safetensors + VRAM upload).
- **Tier planner:** 196 → VRAM, 142 → RAM, 0 → disk. numcert recommended mode
  is **fast** for this checkpoint, so all 196 VRAM tensors stamped fast.

## FASE 4 — Real generation (greedy)

| Prompt (mode) | Output | tok/s | EOS |
|---|---|---|---|
| `What is the capital of France?` (chat) | **`The capital of France is Paris.`** | 3.98 | **yes** (8 tok) |
| `Hello` (chat) | `Hello! How can I assist you today?` | 5.88 | **yes** (10 tok) |
| `Rust is a programming language that` (raw) | `is designed to be safe, concurrent, and memory-efficient. It is known for its strong type system, which helps catch errors at compile time` | 10.37 | no (max) |
| `Explain what a database is.` (chat) | `A database is a structured collection of data that is organized and stored in a computer system. It is a collection of related data that…` | 8.14 | no (max) |

Coherent, fluent, factually correct, no corruption or repetition loops. Quality
is clearly higher than TinyLlama-1.1B (expected — stronger model).

## FASE 5 + 7 — HF / f64 comparison + certification

Real-weight numerical test `qwen25_atenia_matches_f64_ground_truth` (committed
f64 reference, `tests/fixtures/qwen25_reference/expected_logits_f64.json`):

```
Atenia F32   max drift vs F64 truth: 0.000335   (threshold 0.5 → PASS, margin ~1500x)
PyTorch BF16 max drift vs F64 truth: 1.531417   (industry reference)
Argmax Atenia vs F64: 4/4 MATCH  (ids 330, 198, 198, 1)
```

Atenia's F32 forward is **closer to the f64 ground truth than PyTorch's own BF16**
(0.000335 vs 1.53). Committed `model.numcert.json` (bf16-kernel modes):

| Mode | max_abs_diff vs f64 | argmax | ADR-004 |
|---|---|---|---|
| certified | 0.022496 | 4/4 | strict pass, margin 22× |
| fast (recommended) | 0.184907 | 4/4 | strict pass, margin 2.7× |

No severe deviation from HF — far within bounds.

## FASE 6 — Robustness

| Case | Result |
|---|---|
| empty prompt (chat) | no crash, generates ("I'm sorry, but I'm not"), exit 0 |
| short prompt (`Hello`) | clean, EOS |
| moderate prompt (`Explain…`) | coherent, stable |
| `--max-tokens 0` | rejected, exit 2 |
| EOS | confirmed (capital/Hello prompts stopped at EOS) |
| determinism | two identical greedy runs → **identical 24 token ids** |

## FASE 10 — Final validation (real, exit 0)

- `qwen25_loads_and_executes_forward_with_real_weights` (`#[ignore]`, real
  weights) — **pass**: 338 tensors in 10.57s, forward 2.83s, logit shape +
  sanity OK.
- `qwen25_atenia_matches_f64_ground_truth` (`#[ignore]`) — **pass**: drift
  0.000335, argmax 4/4.
- Real CLI generation: 4 prompts coherent; robustness 5/5; determinism
  confirmed.

> Both real-weight tests are `#[ignore]` (need the 2.88 GB checkpoint, absent in
> CI). No code changed this milestone, so the CI-blocking suites are unaffected.

## Problems found

**None.** Unlike RUNTIME-REAL-1 (which surfaced a stale M5 param-tier
assertion), Qwen2.5 validated cleanly with no unexpected behaviour and no code
change required.

## FASE 9 — Strategic review

**Can Qwen-family be considered GREEN?** → **Yes, GREEN** for Qwen2.5 dense text
generation.

Evidence: real 2.88 GB checkpoint loads + tiers correctly; QKV-bias + tied
embeddings + 151936 vocab + θ=1e6 all handled with no new architecture;
generates coherent, factually-correct, deterministic text; EOS works; robust to
bad input; certified vs f64 at drift 0.000335 (argmax 4/4) with committed
fixtures and numcert sidecar.

**Sufficient evidence?** Yes for the 1.5B Instruct checkpoint end-to-end.

**What's missing / stays open:**
- Only **Qwen2.5-1.5B** exercised end-to-end this pass; 3B (sharded) / 7B /
  Coder-1.5B present locally but not re-validated here.
- Qwen3 is a distinct variant (QK-norm) — separate from Qwen2.5; out of scope.
- Throughput is correctness-first (4–10 tok/s), not optimised — **YELLOW**.

## Files modified

- `docs/HANDOFF_RUNTIME_REAL_2.md` — this file.
- `docs/STATUS.md` — RUNTIME-REAL-2 evidence note.

No production code, tests, architecture, families, math, or graph ops changed.

## Deliverable answers

1. **Loaded correctly?** Yes — 338 tensors, ~2.87 GiB resident, VRAM/RAM tiered,
   tied embeddings + QKV bias handled.
2. **Coherent text?** Yes — "The capital of France is Paris.", correct Rust /
   database explanations, clean EOS.
3. **Important differences vs HF?** No — F32 drift vs f64 = 0.000335, argmax 4/4
   (tighter than PyTorch BF16's own 1.53).
4. **Problems found?** None.
5. **Memory used?** ~2.87 GiB resident (VRAM 2.44 + RAM 0.43).
6. **Load time?** ~10.6–12.6 s.
7. **Generation time?** ~4–10 tok/s greedy on GPU.
8. **Qwen-family → GREEN?** Yes (Qwen2.5 dense); YELLOW on throughput + breadth.
9. **Files modified:** see above.
10. **Commit:** see git log (RUNTIME-REAL: Qwen2.5 end-to-end validation).
11. **CI:** see push result.
