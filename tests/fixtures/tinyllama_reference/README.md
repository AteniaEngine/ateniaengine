# TinyLlama Reference Fixture

PyTorch reference logits for validating Atenia's TinyLlama forward pass
numerically (M4.5-d).

## Files

- `generate.py`: PyTorch script that loads TinyLlama and dumps reference logits
- `requirements.txt`: Python dependencies
- `inputs.json`: token IDs used (matches the Atenia M4.5-c test)
- `expected_logits.json`: PyTorch reference logits (committed, consumed by Rust tests)

## Regenerating

Only needed if:

- Token IDs change
- TinyLlama model version changes
- HF transformers significantly changes its inference path

To regenerate:

    pip install -r requirements.txt
    python generate.py /path/to/tinyllama-1.1b/

This produces `inputs.json` and `expected_logits.json` which must be committed.

## Numerical considerations

PyTorch runs inference in BF16 (matching the model's native dtype on disk).
Atenia upcasts weights to F32 on load. This introduces numerical drift that
is expected and bounded — accumulating over 22 transformer blocks plus a
2048×32000 LM-head matmul, drift in the low single digits is plausible.

The validation test (`tests/tinyllama_numerical_validation_test.rs`)
reports drift statistics rather than asserting a tight tolerance, so the
first run can establish what "acceptable" actually means for this stack.
