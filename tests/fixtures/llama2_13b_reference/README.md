# Llama 2 13B Chat reference fixtures (M4.7.6.d)

Reference logits used by `tests/m4_7_6_d_llama2_13b_mode_a_test.rs`
(Mode A) and the M4.7.6.e tri-mode demo (Modes B / C add LRU spill
and force spill on top of the same loader / forward pipeline).

## Files

- `generate_bf16.py` — runs `meta-llama/Llama-2-13b-chat-hf` via
  PyTorch with `torch_dtype=torch.bfloat16`, captures the seq=4
  logits on token pattern `[1, 100, 200, 300]`, and writes
  `expected_logits_bf16.json`.
- `expected_logits_bf16.json` — the BF16 reference produced by the
  script. Serialised as F32 for JSON compatibility (BF16 is the
  upper 16 bits of an F32, so the upcast is exact). ~512 K F32
  values, ~1.5 GB on disk.
- `expected_logits_f64.json` — **deferred to v20 release**. F64
  fixture cannot be generated on the dev box (13B × 8 bytes = 104
  GB, exceeds 32 GB RAM). Will be produced on rented cloud GPU
  (L40S 96 GB or A100 80 GB) before the v20 tag and committed
  alongside this README.

## Why BF16 dev-local + F64 cloud (hybrid strategy)

Locked by the M4.7.6 investigation, decision (2). The BF16 fixture
is a **regression gate** that fits on the dev box and produces a
strict element-wise comparison contract: Atenia's BF16 storage
forward is bit-exact equivalent to PyTorch's BF16 forward at the
truncation level (same upper-16 round rule), so any drift signals
a real issue in Atenia's compute pipeline rather than a precision
artefact.

The F64 fixture is the **ADR-004 mathematical-truth lock**. It
preserves the same contract Atenia uses for the four M4.6 family
checkpoints (TinyLlama, SmolLM2, Qwen 2.5, Llama 3.2 — each
validated against PyTorch in F64). When the F64 fixture lands,
the M4.7.6.d test gains a second comparison gate against
`expected_logits_f64.json`.

## Generating the BF16 fixture

```powershell
# From the project root:
python tests/fixtures/llama2_13b_reference/generate_bf16.py \
    models/llama-2-13b-chat
```

Hardware prerequisite: ~28 GB free RAM on the dev box (26 GB
weights + ~2 GB activations / overhead). Close Chrome / VS Code
before running for the cleanest baseline.

Wall-clock on the reference dev hardware (RTX 4070 Laptop, 32 GB
RAM, model on F: USB HDD): load ~10 min (HDD-bound), forward on
CPU ~10 min — total ~20 minutes for one fixture generation.

## Token pattern

Same as the four M4.6 family fixtures: `[1, 100, 200, 300]`. This
is the canonical seq=4 input that has driven every Atenia LLM
validation since M4.5-d.1. Reusing it keeps the regression gates
comparable across models and milestones.

## Stats from the reference

Captured at fixture-generation time and printed by the generator;
not asserted in code (the fixture's `predicted_token_id`,
`predicted_logit`, `max_abs`, `mean_abs` fields are observability
metadata, not test contracts).
