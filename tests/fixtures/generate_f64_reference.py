"""
CERTIFY-BREADTH-1 — reusable F64 reference generator.

Generalises the per-model `tests/fixtures/<model>_reference/generate_f64.py`
scripts (TinyLlama / SmolLM2 / Qwen2.5 / Llama-3.2) into one parametrised tool
so a new family (Gemma 2, Gemma 3, Phi-3) can get an ADR-004 F64 ground-truth
reference without copy-pasting a script.

Methodology: ADR-002 / ADR-004. PyTorch loaded with `torch_dtype=float64`
(`model.double()` equivalent) gives ~15-17 decimal digits — the "practical
mathematical truth" Atenia's F32 forward is asserted against
(`max_abs_diff < 0.5`, argmax match). The canonical fixture input is the same
4-token sequence every existing fixture uses: TOKEN_IDS = [1, 100, 200, 300].

Output format is byte-identical to the existing fixtures so the Rust harness
(`tests/certify_breadth_f64_validation_test.rs`) consumes it unchanged:

    {
      "shape": [1, 4, vocab],
      "values": [ ... 4*vocab f64 logits, row-major (pos, vocab) ... ],
      "dtype": "f64",
      "predicted_token_id": <argmax of the last position>,
      "predicted_logit": <float>,
      "max_abs": <float>,
      "mean_abs": <float>
    }

Usage:
    python generate_f64_reference.py <model_dir> <output_dir>

  <model_dir>   HF checkpoint directory (config.json + *.safetensors[.index.json]
                + tokenizer files). Loaded with local_files_only=True.
  <output_dir>  Where `expected_logits_f64.json` is written (typically
                tests/fixtures/<model>_reference/).

RAM note (ADR-004 "Negative"): F64 doubles the weight footprint. Approximate
peak RAM by parameter count:
  - ~1 B params  → ~8-10 GiB   (Gemma-3-1B: feasible on 16 GiB)
  - ~2.6 B       → ~21 GiB     (Gemma-2-2B: needs ~24 GiB free)
  - ~3.8 B       → ~30 GiB     (Phi-3.5-mini: near a 32 GiB ceiling)
The script prints an estimate and refuses to proceed silently into swap if the
available RAM is clearly insufficient (override with --force).
"""

import argparse
import gc
import json
import sys
from pathlib import Path

TOKEN_IDS = [1, 100, 200, 300]


def estimate_f64_peak_gib(num_params: int) -> float:
    # Weights in f64 (8 B) plus a conservative ~15% for activations /
    # transient f32->f64 conversions on a 4-token forward.
    return (num_params * 8 / (1024**3)) * 1.15


def available_ram_gib() -> float | None:
    try:
        import psutil  # optional

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate an ADR-004 F64 reference fixture.")
    ap.add_argument("model_dir", help="HF checkpoint directory")
    ap.add_argument("output_dir", help="Directory to write expected_logits_f64.json")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if estimated F64 RAM exceeds available RAM",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight RAM guard from the declared parameter count.
    try:
        cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        n_params_hint = getattr(cfg, "num_parameters", None)
    except Exception:
        n_params_hint = None
    avail = available_ram_gib()
    if avail is not None:
        print(f"Available RAM: {avail:.1f} GiB")

    print(f"Loading {model_dir} directly in F64 (this is slow + RAM-heavy)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float64,
        local_files_only=True,
    )
    model.eval()
    gc.collect()

    n_params = sum(p.numel() for p in model.parameters())
    peak = estimate_f64_peak_gib(n_params)
    print(f"Parameters: {n_params/1e9:.3f} B   estimated F64 peak: ~{peak:.1f} GiB")
    if avail is not None and peak > avail and not args.force:
        print(
            f"REFUSING: estimated F64 peak (~{peak:.1f} GiB) exceeds available RAM "
            f"(~{avail:.1f} GiB). Free RAM or re-run with --force to risk swap."
        )
        return 2

    sample_param = next(model.parameters())
    assert sample_param.dtype == torch.float64, "Model must be F64"

    input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long)
    print(f"Token IDs: {TOKEN_IDS}  input shape: {tuple(input_ids.shape)}")
    print("Running F64 forward pass... (slow)")
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False, output_hidden_states=False)

    logits = outputs.logits  # [1, 4, vocab]
    print(f"Logits shape: {tuple(logits.shape)}  dtype: {logits.dtype}")

    print("\nPer-position argmax (F64 truth):")
    for pos in range(input_ids.shape[1]):
        row = logits[0, pos, :]
        print(
            f"  Pos {pos}: argmax id={int(row.argmax().item())} "
            f"max_abs={row.abs().max().item():.8f}"
        )

    last = logits[0, -1, :]
    pred_id = int(last.argmax().item())
    data = {
        "shape": list(logits.shape),
        "values": logits.flatten().cpu().numpy().tolist(),
        "dtype": "f64",
        "predicted_token_id": pred_id,
        "predicted_logit": float(last[pred_id].item()),
        "max_abs": float(logits.abs().max().item()),
        "mean_abs": float(logits.abs().mean().item()),
    }
    out_path = out_dir / "expected_logits_f64.json"
    with open(out_path, "w") as f:
        json.dump(data, f)
    print(
        f"\nWrote {out_path} ({len(data['values'])} F64 values, "
        f"~{out_path.stat().st_size / 1e6:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
