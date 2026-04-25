"""
Generate fixture for RoPE half-split layout validation.

This script must be run with PyTorch installed:
  pip install -r requirements.txt

Run:
  python generate.py

Outputs inputs.json and expected_outputs.json that get
committed and consumed by tests/rope_test.rs.
"""

import json
import torch

# Smallest meaningful case: half-split needs half >= 1
# We use seq_len=4, n_heads=2, head_dim=4 so half=2
BATCH = 1
SEQ_LEN = 4
N_HEADS = 2
HEAD_DIM = 4
BASE_FREQ = 10000.0  # TinyLlama / Llama 2

def rotate_half(x):
    """HuggingFace convention: split last dim in half,
    rotate (-second_half, first_half)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope_reference(x, position_ids, base_freq=BASE_FREQ):
    """Reference implementation matching HuggingFace LlamaRotaryEmbedding."""
    head_dim = x.shape[-1]
    half = head_dim // 2

    # Frequencies: theta_i = base_freq^(-2i/d) for i in 0..half
    inv_freq = 1.0 / (base_freq ** (
        torch.arange(0, half, dtype=torch.float32) * 2.0 / head_dim
    ))

    # Position * frequency for each (position, freq) pair
    # position_ids shape: [seq_len]
    # inv_freq shape: [half]
    # freqs shape: [seq_len, half]
    freqs = torch.outer(position_ids.float(), inv_freq)

    # cos/sin: [seq_len, head_dim] (each freq repeated to fill head_dim)
    cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)
    sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)

    # Broadcast cos/sin to match x shape: [batch, seq_len, n_heads, head_dim]
    # cos/sin currently: [seq_len, head_dim]
    # Need: [1, seq_len, 1, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Apply: x_out = x * cos + rotate_half(x) * sin
    return x * cos + rotate_half(x) * sin

def main():
    torch.manual_seed(42)

    # Deterministic input
    x = torch.randn(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM, dtype=torch.float32)
    position_ids = torch.arange(SEQ_LEN)

    # Compute reference output
    x_rotated = apply_rope_reference(x, position_ids)

    # Save as JSON for Rust consumption
    inputs = {
        "shape": list(x.shape),
        "head_dim": HEAD_DIM,
        "base_freq": int(BASE_FREQ),
        "values": x.flatten().tolist(),
    }
    outputs = {
        "shape": list(x_rotated.shape),
        "values": x_rotated.flatten().tolist(),
    }

    with open("inputs.json", "w") as f:
        json.dump(inputs, f, indent=2)

    with open("expected_outputs.json", "w") as f:
        json.dump(outputs, f, indent=2)

    print(f"Generated fixture:")
    print(f"  Input shape: {list(x.shape)}")
    print(f"  Output shape: {list(x_rotated.shape)}")
    print(f"  base_freq: {BASE_FREQ}")
    print(f"  Sample input[0,0,0,:]: {x[0,0,0,:].tolist()}")
    print(f"  Sample output[0,0,0,:]: {x_rotated[0,0,0,:].tolist()}")

if __name__ == "__main__":
    main()
