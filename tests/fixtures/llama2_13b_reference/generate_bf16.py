"""
Generate BF16 reference logits for Llama 2 13B Chat using PyTorch
(M4.7.6.d — first end-to-end 13B forward, dev-local CI gate).

Context:
The M4.7 family ADR-004 contract uses **PyTorch F64 as the
practical mathematical truth** because F64 gives 15-17 decimal
digits of precision vs F32's 7 and BF16's ~3. Llama 2 13B in F64
weighs 13.0 B × 8 bytes = **104 GB** which does not fit on the
dev hardware (32 GB RAM, 8 GB VRAM). The M4.7.6 investigation
locked a **hybrid strategy**:

  1. Dev-local **BF16 reference** (this script). Runs on the
     dev box in 32 GB. Atenia's BF16 storage path is bit-exact
     equivalent to PyTorch's BF16 forward (same truncation
     rule), so the comparison is element-wise equality — far
     stricter than the 0.5 ADR-004 contract.
  2. **Cloud F64 fixture** (deferred, pre-tag v20). Will be
     generated on rented GPU (L40S 96 GB or A100 80 GB) and
     dropped under `expected_logits_f64.json` next to this
     `expected_logits_bf16.json`. The F64 fixture is the
     ADR-004 lock; the BF16 fixture is the regression gate
     that fits on the dev box.

Memory note:
  Llama 2 13B in BF16 is ~26 GB of weights + ~100 MB of
  activations at seq=4. Plus PyTorch/HF overhead (~1 GB),
  peak RAM is ~28 GB — fits the 32 GB dev box assuming a
  clean-ish state (close Chrome / IDE before running).

Usage:
    python generate_bf16.py /path/to/llama-2-13b-chat/
"""

import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

# Same input pattern the four-model harness uses: the M4.6.1
# canonical token sequence so logits land in the same numerical
# regime as the existing F64 fixtures.
TOKEN_IDS = [1, 100, 200, 300]


def main():
    if len(sys.argv) != 2:
        print("Usage: generate_bf16.py /path/to/llama-2-13b-chat/")
        sys.exit(1)

    model_dir = sys.argv[1]
    print(f"Loading Llama 2 13B Chat from {model_dir} in BF16...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        # No `device_map` — keep on CPU to match Atenia's
        # current BF16-storage CPU forward path.
    )
    model.eval()
    gc.collect()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    sample_param = next(model.parameters())
    print(f"Model dtype check: first parameter is {sample_param.dtype}")
    assert sample_param.dtype == torch.bfloat16, "Model must be BF16"

    input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long)
    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Token IDs:   {TOKEN_IDS}")

    print("Running BF16 forward pass... (slow on CPU at 13B)")
    fwd_start = time.time()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False, output_hidden_states=False)
    print(f"Forward: {time.time() - fwd_start:.1f}s")

    logits_bf16 = outputs.logits
    # Upcast to F32 for serialization — JSON doesn't carry BF16,
    # and the upcast is exact (BF16 is the upper 16 bits of an
    # F32 with the lower 16 zeroed, so reading the BF16 as F32
    # is an exact zero-extension).
    logits_f32 = logits_bf16.to(dtype=torch.float32)

    print(f"Logits shape: {tuple(logits_bf16.shape)}")
    print(f"Logits dtype: {logits_bf16.dtype} (serialized as f32 for JSON)")

    print("\n=== BF16 reference stats ===")
    print(f"Max abs:  {logits_f32.abs().max().item():.10f}")
    print(f"Mean abs: {logits_f32.abs().mean().item():.10f}")

    print("\nPer-position:")
    for pos in range(input_ids.shape[1]):
        row = logits_f32[0, pos, :]
        max_abs = row.abs().max().item()
        mean_abs = row.abs().mean().item()
        pred_id = int(row.argmax().item())
        pred_logit = float(row[pred_id].item())
        print(
            f"  Pos {pos}: max_abs={max_abs:.6f}  mean_abs={mean_abs:.6f}  "
            f"argmax id={pred_id} logit={pred_logit:.6f}"
        )

    fixture_dir = Path(__file__).parent
    out_path = fixture_dir / "expected_logits_bf16.json"

    last_logits = logits_f32[0, -1, :]
    pred_id = int(last_logits.argmax().item())

    outputs_data = {
        "shape": list(logits_bf16.shape),
        "values": logits_f32.flatten().cpu().numpy().tolist(),
        "dtype": "bf16-serialized-as-f32",
        "predicted_token_id": pred_id,
        "predicted_logit": float(last_logits[pred_id].item()),
        "max_abs": float(logits_f32.abs().max().item()),
        "mean_abs": float(logits_f32.abs().mean().item()),
    }
    with open(out_path, "w") as f:
        json.dump(outputs_data, f)
    print(
        f"\nWrote {out_path} ({len(outputs_data['values'])} F32 values, "
        f"file ~{out_path.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
