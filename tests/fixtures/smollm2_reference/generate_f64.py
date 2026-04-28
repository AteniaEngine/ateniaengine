"""
Generate F64 reference logits for SmolLM2 using PyTorch (M4.6
Investigation F / ADR-002 Level 1).

This is the "practical mathematical truth" reference: PyTorch in F64
gives 15-17 decimal digits of precision, vs F32's 7 and BF16's ~3.
Both Atenia (F32) and PyTorch BF16 can be compared against this to
determine which is closer to truth.

Memory note: SmolLM2 1.7B in F64 is ~13.6 GB of weights. With raw
safetensors buffer (~3 GB) and activations, peak RAM may approach
the 16 GB limit. We free the raw buffer aggressively and use no
attention cache.

Usage:
    python generate_f64.py /path/to/smollm2-1.7b-instruct/
"""

import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

TOKEN_IDS = [1, 100, 200, 300]


def main():
    if len(sys.argv) != 2:
        print("Usage: generate_f64.py /path/to/smollm2-1.7b-instruct/")
        sys.exit(1)

    model_dir = sys.argv[1]
    print(f"Loading SmolLM2 from {model_dir} directly in F64...")
    # Load with F64 to avoid the BF16->F64 double-pass intermediary
    # buffer. PyTorch will cast from disk BF16 to F64 in one pass.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float64,
        local_files_only=True,
    )
    model.eval()
    gc.collect()

    # Sanity: confirm everything is F64
    sample_param = next(model.parameters())
    print(f"Model dtype check: first parameter is {sample_param.dtype}")
    assert sample_param.dtype == torch.float64, "Model must be F64"

    input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long)
    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Token IDs:   {TOKEN_IDS}")

    print("Running F64 forward pass... (slow)")
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False, output_hidden_states=False)

    logits_f64 = outputs.logits  # already F64
    print(f"Logits shape: {tuple(logits_f64.shape)}")
    print(f"Logits dtype: {logits_f64.dtype}")

    print("\n=== F64 reference stats ===")
    print(f"Max abs:  {logits_f64.abs().max().item():.10f}")
    print(f"Mean abs: {logits_f64.abs().mean().item():.10f}")

    # Per-position stats
    print("\nPer-position:")
    for pos in range(input_ids.shape[1]):
        row = logits_f64[0, pos, :]
        max_abs = row.abs().max().item()
        mean_abs = row.abs().mean().item()
        pred_id = int(row.argmax().item())
        pred_logit = float(row[pred_id].item())
        print(
            f"  Pos {pos}: max_abs={max_abs:.10f}  mean_abs={mean_abs:.10f}  "
            f"argmax id={pred_id} logit={pred_logit:.10f}"
        )

    # Save as F64 fixture
    fixture_dir = Path(__file__).parent
    out_path = fixture_dir / "expected_logits_f64.json"

    last_logits = logits_f64[0, -1, :]
    pred_id = int(last_logits.argmax().item())

    outputs_data = {
        "shape": list(logits_f64.shape),
        "values": logits_f64.flatten().cpu().numpy().tolist(),
        "dtype": "f64",
        "predicted_token_id": pred_id,
        "predicted_logit": float(last_logits[pred_id].item()),
        "max_abs": float(logits_f64.abs().max().item()),
        "mean_abs": float(logits_f64.abs().mean().item()),
    }
    with open(out_path, "w") as f:
        json.dump(outputs_data, f)
    print(
        f"\nWrote {out_path} ({len(outputs_data['values'])} F64 values, "
        f"file ~{out_path.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
