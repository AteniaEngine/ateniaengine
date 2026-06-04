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
  python generate_qwen_moe_decomposition_reference.py <model_dir> [out_dir] [--all]

  <model_dir>  a Qwen1.5-MoE-A2.7B(-Chat) checkout (sharded safetensors + index)
  out_dir      defaults to this fixtures/moe directory
  --all        MOE-CERT-2-ext: generate references for ALL layers (one shared
               probe input, one expert at a time, per-layer router). Default
               (no flag) is the MOE-CERT-2 layer-0-only reference.

Outputs (default, layer 0, ~0.5 MB):
  qwen_moe_decomp_ref.safetensors  input[d_model], router_logits[E],
                                   expert_outputs[E, d_model]  (all f32-stored)
  qwen_moe_decomp_ref.json         config + provenance + topk_indices +
                                   routing_margin + seed

Outputs (--all, all L layers, ~11 MB):
  qwen_moe_decomp_ref_all_layers.safetensors  input[d_model],
      router_logits[L, E], expert_outputs[L, E, d_model]  (all f32-stored)
  qwen_moe_decomp_ref_all_layers.json  per-layer topk_indices + routing_margin

Both stay decomposition-faithful: a single probe input is applied to each
layer's real weights, experts computed ONE AT A TIME in float64 — the full model
is never materialised in F64.
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


class ShardCache:
    """Lazily opens the shard each tensor lives in (via the index) and caches
    open handles, so a layer whose mlp spans two shards is read transparently.
    Never loads a whole shard into RAM (safetensors mmaps; we pull per-tensor)."""

    def __init__(self, model_dir, index):
        self.model_dir = model_dir
        self.index = index
        self.handles = {}

    def f64(self, name):
        shard = self.index[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(os.path.join(self.model_dir, shard), framework="pt")
        return f64(self.handles[shard].get_tensor(name))


def compute_layer(cache, layer, d_model, d_ff, n_experts, top_k, x):
    """Decomposition reference for one layer: router logits (f64) + per-expert
    SwiGLU outputs (f64, one expert at a time). Returns (router_logits[E],
    expert_outputs[E, d_model], topk_indices, routing_margin)."""
    prefix = f"model.layers.{layer}.mlp"
    w_router = cache.f64(f"{prefix}.gate.weight")
    assert w_router.shape == (n_experts, d_model), (layer, w_router.shape)
    router_logits = w_router @ x  # [E]

    expert_outputs = np.zeros((n_experts, d_model), dtype=np.float64)
    for e in range(n_experts):
        g = cache.f64(f"{prefix}.experts.{e}.gate_proj.weight")  # [d_ff,d_model]
        u = cache.f64(f"{prefix}.experts.{e}.up_proj.weight")    # [d_ff,d_model]
        d = cache.f64(f"{prefix}.experts.{e}.down_proj.weight")  # [d_model,d_ff]
        h = silu_f64(g @ x) * (u @ x)   # [d_ff]
        expert_outputs[e] = d @ h       # [d_model]
        del g, u, d, h

    order = sorted(range(n_experts), key=lambda i: (-router_logits[i], i))
    topk_indices = sorted(order[:top_k])
    sorted_desc = [router_logits[i] for i in order]
    routing_margin = float(sorted_desc[top_k - 1] - sorted_desc[top_k])
    return router_logits, expert_outputs, topk_indices, routing_margin


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    all_layers = "--all" in sys.argv[1:]
    model_dir = args[0]
    out_dir = args[1] if len(args) > 1 else os.path.dirname(os.path.abspath(__file__))

    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    d_model = int(cfg["hidden_size"])
    d_ff = int(cfg["moe_intermediate_size"])
    n_experts = int(cfg["num_experts"])
    top_k = int(cfg["num_experts_per_tok"])
    norm_topk = bool(cfg.get("norm_topk_prob", False))
    n_layers = int(cfg["num_hidden_layers"])

    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    cache = ShardCache(model_dir, index)
    print(f"[decomp-ref] model={model_dir}")
    print(f"[decomp-ref] d_model={d_model} d_ff={d_ff} experts={n_experts} top_k={top_k} "
          f"norm_topk_prob={norm_topk} layers={n_layers} mode={'ALL' if all_layers else 'layer0'}")

    # Deterministic probe input. Rounded to float32 FIRST, then promoted back to
    # float64 for the reference math, so the value Atenia consumes (the stored
    # f32 `input`) is bit-identical to the value the f64 reference was computed
    # from — no input mismatch contaminates the per-expert drift. One shared
    # probe is applied to every layer's real weights.
    rng = np.random.default_rng(INPUT_SEED)
    x_f32 = rng.standard_normal(d_model).astype(np.float32)
    x = x_f32.astype(np.float64)
    os.makedirs(out_dir, exist_ok=True)

    if not all_layers:
        # MOE-CERT-2 layer-0 reference (unchanged format/filenames).
        rl, eo, topk, margin = compute_layer(cache, LAYER, d_model, d_ff, n_experts, top_k, x)
        save_file(
            {"input": x_f32, "router_logits": rl.astype(np.float32),
             "expert_outputs": eo.astype(np.float32)},
            os.path.join(out_dir, "qwen_moe_decomp_ref.safetensors"),
        )
        meta = {
            "milestone": "MOE-CERT-2", "obligations": ["C1 per-expert", "C2 router"],
            "model": "Qwen1.5-MoE-A2.7B-Chat",
            "source_arch": cfg.get("architectures", ["Qwen2MoeForCausalLM"])[0],
            "layer": LAYER, "d_model": d_model, "d_ff": d_ff, "num_experts": n_experts,
            "experts_per_token": top_k, "norm_topk_prob": norm_topk, "input_seed": INPUT_SEED,
            "oracle": "NumPy float64 (one expert at a time; no full-model F64)",
            "topk_indices": topk, "routing_margin": margin,
            "reference_file": "qwen_moe_decomp_ref.safetensors",
            "note": "Real trained layer-0 weights. ADR-007 C1/C2. C3 (attention) is "
                    "covered by the existing Qwen-MoE full-forward cert, not here.",
        }
        json.dump(meta, open(os.path.join(out_dir, "qwen_moe_decomp_ref.json"), "w"), indent=2)
        print(f"[decomp-ref] layer0 topk={topk} routing_margin={margin:.6f}")
        return

    # MOE-CERT-2-ext: ALL layers. One expert at a time, one layer at a time.
    router_logits = np.zeros((n_layers, n_experts), dtype=np.float64)
    expert_outputs = np.zeros((n_layers, n_experts, d_model), dtype=np.float64)
    per_layer = []
    for l in range(n_layers):
        rl, eo, topk, margin = compute_layer(cache, l, d_model, d_ff, n_experts, top_k, x)
        router_logits[l] = rl
        expert_outputs[l] = eo
        per_layer.append({"layer": l, "topk_indices": topk, "routing_margin": margin})
        print(f"[decomp-ref] layer {l:2d}: topk={topk} margin={margin:.6f} "
              f"max|out|={np.max(np.abs(eo)):.3f}")

    st_path = os.path.join(out_dir, "qwen_moe_decomp_ref_all_layers.safetensors")
    save_file(
        {"input": x_f32, "router_logits": router_logits.astype(np.float32),
         "expert_outputs": expert_outputs.astype(np.float32)},
        st_path,
    )
    meta = {
        "milestone": "MOE-CERT-2-ext", "obligations": ["C1 per-expert (all layers)",
                                                        "C2 router (all layers)"],
        "model": "Qwen1.5-MoE-A2.7B-Chat",
        "source_arch": cfg.get("architectures", ["Qwen2MoeForCausalLM"])[0],
        "num_layers": n_layers, "d_model": d_model, "d_ff": d_ff, "num_experts": n_experts,
        "experts_per_token": top_k, "norm_topk_prob": norm_topk, "input_seed": INPUT_SEED,
        "oracle": "NumPy float64 (one expert at a time, one layer at a time; no full-model F64)",
        "per_layer": per_layer,
        "reference_file": "qwen_moe_decomp_ref_all_layers.safetensors",
        "note": "Real trained weights, ALL layers. ADR-007 C1/C2 exhaustive. C3 "
                "(attention) reused from the existing Qwen-MoE full-forward cert.",
    }
    json.dump(meta, open(os.path.join(out_dir, "qwen_moe_decomp_ref_all_layers.json"), "w"), indent=2)
    print(f"[decomp-ref] wrote {st_path} ({os.path.getsize(st_path)} bytes)")
    print(f"[decomp-ref] all {n_layers} layers; "
          f"min routing_margin={min(p['routing_margin'] for p in per_layer):.6f}")


if __name__ == "__main__":
    main()
