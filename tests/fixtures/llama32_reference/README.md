# Llama 3.2 1B Instruct Reference Fixtures

PyTorch reference logits used by the Phase C numerical validation
tests:

- `tests/llama_3_2_numerical_validation_test.rs` (Phase C.5,
  seq_len = 4 — high-frequency band only)
- `tests/llama_3_2_long_context_validation_test.rs` (Phase C.6,
  seq_len ≥ 4096 — exercises the mid + low frequency bands and
  is the only falsifier of the rope_scaling wiring)

## Files

- `inputs.json` — token IDs used at seq_len = 4
  (`[1, 100, 200, 300]`, mirrors the rest of the family).
- `expected_logits.json` — PyTorch BF16 reference, seq_len = 4,
  shape `[1, 4, 128256]`. Reported as informative drift per
  ADR-004; **not** the assertion target.
- `expected_logits_f64.json` — PyTorch F64 ground truth,
  seq_len = 4, same shape. **Primary** validation target
  (`max_abs_diff < 0.5` per ADR-004).
- C.6 long-context fixtures land separately in this same
  directory once C.6 is implemented.
- `generate.py` / `generate_f64.py` — fixture generators.
- `requirements.txt` — Python dependencies.

## Regenerating

```bash
pip install -r requirements.txt
python generate.py     /path/to/llama-3.2-1b-instruct/
python generate_f64.py /path/to/llama-3.2-1b-instruct/
```

The F64 generator needs roughly 12–14 GB of RAM during the forward
pass.

## Methodology

Per ADR-002 (Mathematical Ground-Truth Validation Strategy) and
ADR-004 (F64 Reference as Default), Atenia is asserted against
the F64 fixture. The BF16 fixture is preserved for historical
continuity and to track how industry-default BF16 inference drifts
from mathematical truth on this model.
