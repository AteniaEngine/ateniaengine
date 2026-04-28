# SmolLM2 Reference Fixture

PyTorch reference logits for validating Atenia's SmolLM2 forward pass
numerically as part of the M4.6 milestone (Llama-family expansion,
Phase A.1 empirical validation with tied embeddings).

## Files

- `generate.py`: PyTorch script that loads SmolLM2 1.7B Instruct and
  dumps reference logits.
- `requirements.txt`: Python dependencies.
- `inputs.json`: token IDs used (matches the Atenia M4.6 smoke test).
- `expected_logits.json`: PyTorch reference logits (committed, consumed
  by Rust tests).

## Regenerating

Only needed if:

- Token IDs change
- SmolLM2 model version changes
- HF transformers significantly changes its inference path

To regenerate:

    pip install -r requirements.txt
    python generate.py /path/to/smollm2-1.7b-instruct/

This produces `inputs.json` and `expected_logits.json` which must be
committed.

## Numerical considerations

Same precision-stack notes as the TinyLlama fixture: PyTorch runs in
BF16 (model native), Atenia upcasts to F32 on load. Drift accumulates
over 24 transformer blocks (vs TinyLlama's 22) plus a 2048×49152 LM
head reused-from-embed-tokens via the Phase A.1 tied-embedding path.

The validation test (`tests/smollm2_numerical_validation_test.rs`)
reports drift statistics rather than asserting a tight tolerance. The
catastrophic-drift safeguard is the same `max_abs_diff < 5.0` cutoff
used by `tinyllama_numerical_validation_test.rs`.
