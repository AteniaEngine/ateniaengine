"""
Generate F64 reference logits for TinyLlama 1.1B Chat using PyTorch
(M4.6.1 retroactive F64 validation per ADR-004).

Context:
TinyLlama was originally validated in M4.5-d.1 against PyTorch BF16
(max_abs_diff < 5.0). ADR-004, accepted after M4.6 Phase A surfaced
the BF16-as-truth flaw on SmolLM2, established PyTorch F64 as the
primary numerical reference. This script closes the methodology debt
by producing the F64 fixture for TinyLlama; the original M4.5-d.1
test is preserved unchanged as historical record of the pre-ADR-004
methodology, and a new `tinyllama_f64_validation_test.rs` consumes
the fixture this script writes.

This is the "practical mathematical truth" reference: PyTorch in F64
gives 15-17 decimal digits of precision, vs F32's 7 and BF16's ~3.
Both Atenia (F32) and PyTorch BF16 can be compared against this.

Memory note: TinyLlama 1.1B in F64 is ~8.8 GB of weights (1.1B params
* 8 bytes). Plus activations and intermediate F32 -> F64 conversions,
peak RAM is comfortably within 16 GB.

Usage:
    python generate_f64.py /path/to/tinyllama-1.1b/
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
        print("Usage: generate_f64.py /path/to/tinyllama-1.1b/")
        sys.exit(1)

    model_dir = sys.argv[1]
    print(f"Loading TinyLlama from {model_dir} directly in F64...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float64,
        local_files_only=True,
    )
    model.eval()
    gc.collect()

    sample_param = next(model.parameters())
    print(f"Model dtype check: first parameter is {sample_param.dtype}")
    assert sample_param.dtype == torch.float64, "Model must be F64"

    input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long)
    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Token IDs:   {TOKEN_IDS}")

    print("Running F64 forward pass... (slow)")
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False, output_hidden_states=False)

    logits_f64 = outputs.logits
    print(f"Logits shape: {tuple(logits_f64.shape)}")
    print(f"Logits dtype: {logits_f64.dtype}")

    print("\n=== F64 reference stats ===")
    print(f"Max abs:  {logits_f64.abs().max().item():.10f}")
    print(f"Mean abs: {logits_f64.abs().mean().item():.10f}")

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
