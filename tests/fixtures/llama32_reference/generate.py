"""
Generate Llama 3.2 1B Instruct BF16 reference fixture for Atenia
numerical validation (M4.6 Phase C.5).

Loads meta-llama/Llama-3.2-1B-Instruct from a local checkpoint, runs
a forward pass with fixed input tokens, and dumps logits to JSON for
consumption by tests/llama_3_2_numerical_validation_test.rs.

Usage:
    pip install -r requirements.txt
    python generate.py /path/to/llama-3.2-1b-instruct/

Outputs:
    inputs.json           — token IDs used
    expected_logits.json  — PyTorch BF16 reference logits, shape [1, 4, 128256]
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

# Match the rest of the family (TinyLlama / SmolLM2 / Qwen) so the
# test interpretation is uniform.
TOKEN_IDS = [1, 100, 200, 300]
BATCH_SIZE = 1


def main():
    if len(sys.argv) != 2:
        print("Usage: generate.py /path/to/llama-3.2-1b-instruct/")
        sys.exit(1)

    model_dir = sys.argv[1]
    print(f"Loading Llama 3.2 1B from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.eval()

    input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long)
    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Token IDs:   {TOKEN_IDS}")

    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=False)

    logits = outputs.logits  # [batch, seq, vocab]
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Logits dtype: {logits.dtype}")

    logits_f32 = logits.float().numpy()
    print("Logits stats:")
    print(f"  Max abs : {abs(logits_f32).max():.4f}")
    print(f"  Mean abs: {abs(logits_f32).mean():.4f}")

    last_logits = logits_f32[0, -1, :]
    pred_id = int(last_logits.argmax())
    pred_logit = float(last_logits[pred_id])
    print(f"Predicted next token: id={pred_id} logit={pred_logit:.4f}")

    fixture_dir = Path(__file__).parent

    inputs_data = {
        "shape": list(input_ids.shape),
        "token_ids": TOKEN_IDS,
        "batch_size": BATCH_SIZE,
        "seq_len": len(TOKEN_IDS),
    }
    with open(fixture_dir / "inputs.json", "w") as f:
        json.dump(inputs_data, f, indent=2)
    print(f"Wrote {fixture_dir / 'inputs.json'}")

    outputs_data = {
        "shape": list(logits_f32.shape),
        "values": logits_f32.flatten().tolist(),
        "predicted_token_id": pred_id,
        "predicted_logit": pred_logit,
        "max_abs": float(abs(logits_f32).max()),
        "mean_abs": float(abs(logits_f32).mean()),
    }
    with open(fixture_dir / "expected_logits.json", "w") as f:
        json.dump(outputs_data, f)
    print(
        f"Wrote {fixture_dir / 'expected_logits.json'} "
        f"({len(outputs_data['values'])} values)"
    )


if __name__ == "__main__":
    main()
