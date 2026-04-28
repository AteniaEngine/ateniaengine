# Qwen 2.5 1.5B Reference Fixtures

PyTorch reference logits used by `tests/qwen25_numerical_validation_test.rs`
(M4.6 Phase B.5).

## Files

- `inputs.json` — token IDs used (`[1, 100, 200, 300]`, mirrors TinyLlama / SmolLM2).
- `expected_logits.json` — PyTorch BF16 reference, shape `[1, 4, 151936]`. Reported as informative drift per ADR-004; **not** the assertion target.
- `expected_logits_f64.json` — PyTorch F64 ground truth, same shape. **Primary** validation target (`max_abs_diff < 0.5` per ADR-004).
- `generate.py` / `generate_f64.py` — fixture generators.
- `requirements.txt` — Python dependencies for both scripts.

## Regenerating

```bash
pip install -r requirements.txt
python generate.py     /path/to/qwen2.5-1.5b-instruct/
python generate_f64.py /path/to/qwen2.5-1.5b-instruct/
```

The F64 generator needs roughly 14–16 GB of RAM during the forward
pass (1.5B parameters × 8 bytes for the model, plus activations
and the intermediate F32 ↔ F64 buffers HuggingFace allocates).

## Methodology

Per ADR-002 (Mathematical Ground-Truth Validation Strategy) and
ADR-004 (F64 Reference as Default), Atenia is asserted against
the F64 fixture. The BF16 fixture is preserved for historical
continuity and to track how industry-default BF16 inference drifts
from mathematical truth on this model.
