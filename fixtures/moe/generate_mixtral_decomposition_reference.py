#!/usr/bin/env python3
"""MIXTRAL-CERT-1 (C1+C2) — per-expert + router F64 decomposition reference for the
REAL Mixtral-8x7B-v0.1. Mirrors the Qwen / DeepSeek decomposition generators with
the Mixtral specifics:
  * router/experts live under `block_sparse_moe` (not `mlp`);
  * classic expert layout `experts.{e}.{w1,w3,w2}` (w1=gate, w3=up, w2=down);
  * **no shared expert**, **no dense-first** layers (all layers are MoE);
  * config keys `num_local_experts` / `num_experts_per_tok` / `intermediate_size`.

Decomposition-faithful: one deterministic probe input is applied to each layer's
real weights; experts computed ONE AT A TIME in float64 (never the full model in
F64). Reads bf16 -> torch.float64. C3 (attention) is NOT here. No certification is
performed; this only emits the reference the Rust harness gates against.

Usage:
  python generate_mixtral_decomposition_reference.py <model_dir> [out_dir]

Outputs (~4 MB):
  mixtral_decomp_ref.safetensors  input[d_model], router_logits[L,E],
                                  expert_outputs[L,E,d_model] (f32-stored)
  mixtral_decomp_ref.json         per-layer (layer idx + topk + margin)
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

INPUT_SEED = 20260606


def f64(t):
    return t.to(torch.float64).numpy()


def silu_f64(x):
    return x / (1.0 + np.exp(-x))


class ShardCache:
    """Opens the shard each tensor lives in (via the index), caches handles; a
    layer whose block_sparse_moe spans two shards is read transparently."""

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
    prefix = f"model.layers.{layer}.block_sparse_moe"
    w_router = cache.f64(f"{prefix}.gate.weight")
    assert w_router.shape == (n_experts, d_model), (layer, w_router.shape)
    router_logits = w_router @ x  # [E]
    expert_outputs = np.zeros((n_experts, d_model), dtype=np.float64)
    for e in range(n_experts):
        w1 = cache.f64(f"{prefix}.experts.{e}.w1.weight")  # gate [d_ff,d_model]
        w3 = cache.f64(f"{prefix}.experts.{e}.w3.weight")  # up   [d_ff,d_model]
        w2 = cache.f64(f"{prefix}.experts.{e}.w2.weight")  # down [d_model,d_ff]
        h = silu_f64(w1 @ x) * (w3 @ x)
        expert_outputs[e] = w2 @ h
        del w1, w3, w2, h
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
    d_ff = int(cfg["intermediate_size"])  # Mixtral: per-expert FFN width
    n_experts = int(cfg["num_local_experts"])
    top_k = int(cfg["num_experts_per_tok"])
    n_layers = int(cfg["num_hidden_layers"])

    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    cache = ShardCache(model_dir, index)
    print(f"[mixtral-decomp] model={model_dir}")
    print(f"[mixtral-decomp] d_model={d_model} d_ff={d_ff} experts={n_experts} top_k={top_k} "
          f"layers={n_layers} (all MoE, no shared) -> total experts={n_layers*n_experts}")

    rng = np.random.default_rng(INPUT_SEED)
    x_f32 = rng.standard_normal(d_model).astype(np.float32)
    x = x_f32.astype(np.float64)

    # **Resumable checkpoints** — save each layer immediately so an interrupted run
    # (e.g. the PC sleeping while idle) loses at most the in-flight layer and resumes.
    ckpt_dir = os.path.join(out_dir, ".mixtral_decomp_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    for l in range(n_layers):
        cp = os.path.join(ckpt_dir, f"layer_{l:02d}.npz")
        if os.path.exists(cp):
            print(f"[mixtral-decomp] layer {l:2d}: SKIP (checkpoint exists)", flush=True)
            continue
        rl, eo, topk, margin = compute_layer(cache, l, d_model, n_experts, top_k, x)
        np.savez(cp + ".tmp.npz", router_logits=rl, expert_outputs=eo,
                 topk=np.array(topk, dtype=np.int64), margin=np.float64(margin))
        os.replace(cp + ".tmp.npz", cp)  # atomic: a complete checkpoint or none
        print(f"[mixtral-decomp] layer {l:2d}: topk={topk} margin={margin:.6f} "
              f"max|out|={np.max(np.abs(eo)):.3f} [saved]", flush=True)

    # Gather all per-layer checkpoints into the final reference.
    router_logits = np.zeros((n_layers, n_experts), dtype=np.float64)
    expert_outputs = np.zeros((n_layers, n_experts, d_model), dtype=np.float64)
    per_layer = []
    for l in range(n_layers):
        cp = os.path.join(ckpt_dir, f"layer_{l:02d}.npz")
        d = np.load(cp)
        router_logits[l] = d["router_logits"]
        expert_outputs[l] = d["expert_outputs"]
        per_layer.append({"layer": l, "topk_indices": [int(t) for t in d["topk"]],
                          "routing_margin": float(d["margin"])})

    os.makedirs(out_dir, exist_ok=True)
    st = os.path.join(out_dir, "mixtral_decomp_ref.safetensors")
    save_file({"input": x_f32, "router_logits": router_logits.astype(np.float32),
               "expert_outputs": expert_outputs.astype(np.float32)}, st)
    meta = {
        "milestone": "MIXTRAL-CERT-1 (C1+C2)",
        "obligations": ["C1 per-expert (all layers)", "C2 router (all layers)"],
        "model": "Mixtral-8x7B-v0.1",
        "source_arch": cfg.get("architectures", ["MixtralForCausalLM"])[0],
        "num_hidden_layers": n_layers, "num_moe_layers": n_layers,
        "d_model": d_model, "d_ff": d_ff, "num_experts": n_experts,
        "experts_per_token": top_k, "input_seed": INPUT_SEED,
        "total_experts": n_layers * n_experts,
        "oracle": "NumPy/torch float64 (one expert at a time, one layer at a time; "
                  "no full-model F64). Mixtral classic experts w1/w3/w2; no shared expert.",
        "per_layer": per_layer,
        "reference_file": "mixtral_decomp_ref.safetensors",
        "note": "Real Mixtral-8x7B-v0.1 weights, all 32 layers. ADR-007 C1/C2. C3 "
                "(attention) and C4/C5 are not here.",
    }
    json.dump(meta, open(os.path.join(out_dir, "mixtral_decomp_ref.json"), "w"), indent=2)
    print(f"[mixtral-decomp] wrote {st} ({os.path.getsize(st)} bytes)")
    print(f"[mixtral-decomp] {n_layers} layers x {n_experts} experts = {n_layers*n_experts} experts; "
          f"min routing_margin={min(p['routing_margin'] for p in per_layer):.6f}")


if __name__ == "__main__":
    main()
