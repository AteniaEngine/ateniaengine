# RoPE Reference Fixture

Reference outputs for validating Atenia's RoPE implementation
against PyTorch / HuggingFace Transformers convention.

## Layout

Half-split (HuggingFace), NOT interleaved (original paper).

## Regenerating

Only needed if:
- Adding new test cases (different shapes)
- Changing base_freq for new model targets
- PyTorch version updates change numerics

To regenerate:

    pip install -r requirements.txt
    python generate.py

This produces inputs.json and expected_outputs.json which
must be committed.

## Files

- generate.py: PyTorch reference implementation + fixture
  generator
- requirements.txt: Python dependencies (torch only)
- inputs.json: input tensor + config (committed, consumed
  by Rust tests)
- expected_outputs.json: PyTorch reference output
  (committed, consumed by Rust tests)
- README.md: this file
