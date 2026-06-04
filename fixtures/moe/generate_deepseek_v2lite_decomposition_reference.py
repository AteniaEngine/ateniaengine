#!/usr/bin/env python3
"""MLA-1 (C1+C2) — per-expert + router F64 decomposition reference for the REAL
DeepSeek-V2-Lite. Mirrors the Qwen MOE-CERT-2-ext generator (the expert/router
tensor names are identical: `mlp.gate.weight`, `mlp.experts.{e}.{gate,up,down}_proj`)
with two DeepSeek specifics:
  * **dense-first layers are skipped** (`first_k_dense_replace`): only the MoE
    layers carry experts;
  * config keys are `n_routed_experts` / `num_experts_per_tok`.

Decomposition-faithful: a single probe input is applied to each MoE layer's real
weights, experts computed ONE AT A TIME in float64 (never the full model in F64).
Reads bf16 -> torch.float64. C3 (MLA attention) is NOT here; it is covered by the
DeepSeek MLA cert + MLA-0. No certification is performed; this only emits the
reference the Rust harness gates against.

Usage:
  python generate_deepseek_v2lite_decomposition_reference.py <model_dir> [out_dir]

Outputs (~14 MB):
  deepseek_v2lite_decomp_ref.safetensors  input[d_model], router_logits[Lmoe,E],
                                           expert_outputs[Lmoe,E,d_model] (f32-stored)
  deepseek_v2lite_decomp_ref.json          per-layer (real layer idx + topk + margin)
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

INPUT_SEED = 20260604


def f64(t):
    return t.to(torch.float64).numpy()


def silu_f64(x):
    return x / (1.0 + np.exp(-x))


class ShardCache:
    """Opens the shard each tensor lives in (via the index), caches handles; a
    layer whose mlp spans two shards is read transparently. Never loads a whole
    shard into RAM (safetensors mmaps; per-tensor pulls)."""

    def __init__(self, model_dir, index):
        self.model_dir = model_dir
        self.index = index
        self.handles = {}

    def f64(self, name):
        shard = self.index[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(os.path.join(self.model_dir, shard), framework="pt")
        return f64(self.handles[shard].get_tensor(name))


def compute_layer(cache, layer, d_model, n_experts, top_k, x):
    prefix = f"model.layers.{layer}.mlp"
    w_router = cache.f64(f"{prefix}.gate.weight")
    assert w_router.shape == (n_experts, d_model), (layer, w_router.shape)
    router_logits = w_router @ x  # [E]
    expert_outputs = np.zeros((n_experts, d_model), dtype=np.float64)
    for e in range(n_experts):
        g = cache.f64(f"{prefix}.experts.{e}.gate_proj.weight")  # [d_ff,d_model]
        u = cache.f64(f"{prefix}.experts.{e}.up_proj.weight")    # [d_ff,d_model]
        d = cache.f64(f"{prefix}.experts.{e}.down_proj.weight")  # [d_model,d_ff]
        h = silu_f64(g @ x) * (u @ x)
        expert_outputs[e] = d @ h
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
    model_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))

    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    d_model = int(cfg["hidden_size"])
    d_ff = int(cfg["moe_intermediate_size"])
    n_experts = int(cfg["n_routed_experts"])
    top_k = int(cfg["num_experts_per_tok"])
    n_layers = int(cfg["num_hidden_layers"])
    first_k_dense = int(cfg.get("first_k_dense_replace", 0))
    norm_topk = bool(cfg.get("norm_topk_prob", False))

    moe_layers = list(range(first_k_dense, n_layers))  # dense-first layers skipped
    n_moe = len(moe_layers)
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    cache = ShardCache(model_dir, index)
    print(f"[ds-decomp] model={model_dir}")
    print(f"[ds-decomp] d_model={d_model} d_ff={d_ff} experts={n_experts} top_k={top_k} "
          f"norm_topk_prob={norm_topk} layers={n_layers} first_k_dense={first_k_dense} "
          f"MoE_layers={n_moe} -> total experts={n_moe*n_experts}")

    rng = np.random.default_rng(INPUT_SEED)
    x_f32 = rng.standard_normal(d_model).astype(np.float32)
    x = x_f32.astype(np.float64)

    router_logits = np.zeros((n_moe, n_experts), dtype=np.float64)
    expert_outputs = np.zeros((n_moe, n_experts, d_model), dtype=np.float64)
    per_layer = []
    for mi, l in enumerate(moe_layers):
        rl, eo, topk, margin = compute_layer(cache, l, d_model, n_experts, top_k, x)
        router_logits[mi] = rl
        expert_outputs[mi] = eo
        per_layer.append({"moe_index": mi, "layer": l, "topk_indices": topk,
                          "routing_margin": margin})
        print(f"[ds-decomp] moe {mi:2d} (layer {l:2d}): topk={topk} margin={margin:.6f} "
              f"max|out|={np.max(np.abs(eo)):.3f}")

    os.makedirs(out_dir, exist_ok=True)
    st = os.path.join(out_dir, "deepseek_v2lite_decomp_ref.safetensors")
    save_file({"input": x_f32, "router_logits": router_logits.astype(np.float32),
               "expert_outputs": expert_outputs.astype(np.float32)}, st)
    meta = {
        "milestone": "MLA-1 (C1+C2)", "obligations": ["C1 per-expert (all MoE layers)",
                                                       "C2 router (all MoE layers)"],
        "model": "DeepSeek-V2-Lite",
        "source_arch": cfg.get("architectures", ["DeepseekV2ForCausalLM"])[0],
        "num_hidden_layers": n_layers, "first_k_dense_replace": first_k_dense,
        "num_moe_layers": n_moe, "moe_layer_indices": moe_layers,
        "d_model": d_model, "d_ff": d_ff, "num_experts": n_experts,
        "experts_per_token": top_k, "norm_topk_prob": norm_topk, "input_seed": INPUT_SEED,
        "total_experts": n_moe * n_experts,
        "oracle": "NumPy/torch float64 (one expert at a time, one MoE layer at a time; "
                  "no full-model F64; dense-first layers skipped)",
        "per_layer": per_layer,
        "reference_file": "deepseek_v2lite_decomp_ref.safetensors",
        "note": "Real DeepSeek-V2-Lite weights, all MoE layers. ADR-007 C1/C2. C3 (MLA "
                "attention) covered by the DeepSeek MLA cert + MLA-0, not here.",
    }
    json.dump(meta, open(os.path.join(out_dir, "deepseek_v2lite_decomp_ref.json"), "w"), indent=2)
    print(f"[ds-decomp] wrote {st} ({os.path.getsize(st)} bytes)")
    print(f"[ds-decomp] {n_moe} MoE layers x {n_experts} experts = {n_moe*n_experts} experts; "
          f"min routing_margin={min(p['routing_margin'] for p in per_layer):.6f}")


if __name__ == "__main__":
    main()
