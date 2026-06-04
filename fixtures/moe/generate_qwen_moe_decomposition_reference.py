#!/usr/bin/env python3
"""MOE-CERT-2 — per-expert + router F64 decomposition reference for Qwen-MoE.

ADR-007 (docs/decisions/ADR-007-moe-certification-ladder.md) certifies a real
MoE *by decomposition*: a global F64 forward is infeasible (the full weights do
not fit in F64 RAM) AND incomplete (a forward only routes to the top-k experts).
This generator produces the F64 reference the Rust harness checks Atenia against,
for the two obligations MOE-CERT-2 covers:

  C1 (per-expert) — every expert's SwiGLU output, computed in **float64**, ONE
     EXPERT AT A TIME (never materialises the whole model in F64 — exactly the
     point of decomposition). Oracle: NumPy float64 (an independent library from
     Atenia's Rust f64 path; the ADR-002 Level-1 ground-truth form).

  C2 (router) — the router logits in float64, the top-k expert index set, and
     the routing margin (gap between the k-th and (k+1)-th logit), so the harness
     can assert top-k SET EQUALITY (the hard gate) and report fragility.

What it does NOT do: it does not load the model end-to-end, does not run
attention/embeddings/lm_head, does not touch Atenia, and does not certify
anything by itself — it only emits the reference the Rust test gates against.
C3 (attention) is NOT generated here: it is already covered by the existing
Qwen-MoE full-forward certificate (GQA + Q/K/V bias vs HF f64, MOE-FULL-13).

Faithfulness rules (no fabricated numbers):
  * weights are the REAL trained Qwen1.5-MoE-A2.7B layer-0 tensors;
  * the input is deterministic (fixed seed, recorded), so the reference is
    reproducible bit-for-bit;
  * every number written here is a direct float64 computation — none invented.

Usage:
  python generate_qwen_moe_decomposition_reference.py <model_dir> [out_dir]

  <model_dir>  a Qwen1.5-MoE-A2.7B(-Chat) checkout (sharded safetensors + index)
  out_dir      defaults to this fixtures/moe directory

Outputs (committed, ~1 MB):
  qwen_moe_decomp_ref.safetensors  input[d_model] f64, router_logits[E] f64,
                                    expert_outputs[E, d_model] f64
  qwen_moe_decomp_ref.json         config + provenance + topk_indices +
                                    routing_margin + seed
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file


def f64(t):
    """Read a (possibly bf16) safetensors tensor as a float64 numpy array.

    The real Qwen-MoE weights are bf16 on disk (NumPy cannot represent bf16);
    we go bf16 -> torch.float64 -> numpy, the same bf16->double promotion the
    dense ADR-004 references use (`model.double()`)."""
    return t.to(torch.float64).numpy()

# Deterministic input seed. Recorded in the JSON so the harness is reproducible.
INPUT_SEED = 20260603
LAYER = 0


def silu_f64(x):
    # x * sigmoid(x), all float64.
    return x / (1.0 + np.exp(-x))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    model_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))

    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    d_model = int(cfg["hidden_size"])
    d_ff = int(cfg["moe_intermediate_size"])
    n_experts = int(cfg["num_experts"])
    top_k = int(cfg["num_experts_per_tok"])
    norm_topk = bool(cfg.get("norm_topk_prob", False))

    # Locate the shard holding layer-0 mlp tensors via the index.
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    prefix = f"model.layers.{LAYER}.mlp"
    router_name = f"{prefix}.gate.weight"
    shard = index[router_name]
    shard_path = os.path.join(model_dir, shard)
    print(f"[decomp-ref] model={model_dir}")
    print(f"[decomp-ref] d_model={d_model} d_ff={d_ff} experts={n_experts} top_k={top_k} "
          f"norm_topk_prob={norm_topk}")
    print(f"[decomp-ref] layer-{LAYER} mlp shard: {shard}")

    # Deterministic input. Rounded to float32 FIRST, then promoted back to
    # float64 for the reference math, so the value Atenia consumes (the stored
    # f32 `input`) is bit-identical to the value the f64 reference was computed
    # from — no input mismatch contaminates the per-expert drift.
    rng = np.random.default_rng(INPUT_SEED)
    x_f32 = rng.standard_normal(d_model).astype(np.float32)
    x = x_f32.astype(np.float64)

    expert_outputs = np.zeros((n_experts, d_model), dtype=np.float64)

    with safe_open(shard_path, framework="pt") as f:
        # Router logits in float64: W_gate[E, d_model] @ x.
        w_router = f64(f.get_tensor(router_name))
        assert w_router.shape == (n_experts, d_model), w_router.shape
        router_logits = w_router @ x  # [E]

        # Per-expert SwiGLU, ONE EXPERT AT A TIME (decomposition: never all in F64).
        for e in range(n_experts):
            g = f64(f.get_tensor(f"{prefix}.experts.{e}.gate_proj.weight"))  # [d_ff,d_model]
            u = f64(f.get_tensor(f"{prefix}.experts.{e}.up_proj.weight"))    # [d_ff,d_model]
            d = f64(f.get_tensor(f"{prefix}.experts.{e}.down_proj.weight"))  # [d_model,d_ff]
            h = silu_f64(g @ x) * (u @ x)   # [d_ff]
            expert_outputs[e] = d @ h       # [d_model]
            del g, u, d, h

    # Top-k by logit, ties broken by LOWER index (matches Atenia's top_k_routing).
    order = sorted(range(n_experts), key=lambda i: (-router_logits[i], i))
    topk_indices = sorted(order[:top_k])
    # Routing margin: gap between the k-th and (k+1)-th largest logit.
    sorted_desc = [router_logits[i] for i in order]
    routing_margin = float(sorted_desc[top_k - 1] - sorted_desc[top_k])

    # Stored as float32 so Atenia's SafetensorsReader (no F64 target dtype) can
    # consume them. The MATH above is float64; storing f32 keeps >=7 sig digits
    # — immaterial against the 0.5 gate and the ~1e-3 expected drift, and it is
    # exactly how the dense ADR-004 F64 references are consumed (cast to f32 at
    # compare time). `input` is already f32-valued (see above).
    os.makedirs(out_dir, exist_ok=True)
    st_path = os.path.join(out_dir, "qwen_moe_decomp_ref.safetensors")
    save_file(
        {
            "input": x_f32,
            "router_logits": router_logits.astype(np.float32),
            "expert_outputs": expert_outputs.astype(np.float32),
        },
        st_path,
    )

    meta = {
        "milestone": "MOE-CERT-2",
        "obligations": ["C1 per-expert", "C2 router"],
        "model": "Qwen1.5-MoE-A2.7B-Chat",
        "source_arch": cfg.get("architectures", ["Qwen2MoeForCausalLM"])[0],
        "layer": LAYER,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_experts": n_experts,
        "experts_per_token": top_k,
        "norm_topk_prob": norm_topk,
        "input_seed": INPUT_SEED,
        "oracle": "NumPy float64 (one expert at a time; no full-model F64)",
        "topk_indices": topk_indices,
        "routing_margin": routing_margin,
        "reference_file": "qwen_moe_decomp_ref.safetensors",
        "note": "Real trained layer-0 weights. ADR-007 C1/C2. C3 (attention) is "
                "covered by the existing Qwen-MoE full-forward cert, not here.",
    }
    json_path = os.path.join(out_dir, "qwen_moe_decomp_ref.json")
    json.dump(meta, open(json_path, "w"), indent=2)

    print(f"[decomp-ref] wrote {st_path}")
    print(f"[decomp-ref] wrote {json_path}")
    print(f"[decomp-ref] topk_indices={topk_indices} routing_margin={routing_margin:.6f}")
    print(f"[decomp-ref] |x|inf={np.max(np.abs(x)):.4f} "
          f"max|expert_out|={np.max(np.abs(expert_outputs)):.4f}")


if __name__ == "__main__":
    main()
