#!/usr/bin/env python3
"""MOE-FULL-13 — derive a CLASSIC-layout Mixtral checkpoint from the committed
packed `full_mixtral.safetensors` (MOE-FULL-6).

OFFLINE reproducibility artifact. NOT run in CI, NOT imported by Rust.

Current transformers saves Mixtral experts FUSED (`mlp.experts.gate_up_proj`).
The original Mixtral on-disk layout is CLASSIC (`block_sparse_moe.experts.{e}.
w1/w3/w2` + `block_sparse_moe.gate`). To certify the classic expert-binding path
end to end WITHOUT a second HF forward, we mechanically re-pack the SAME numeric
weights into the classic layout (gate = gate_up[:d_ff], up = gate_up[d_ff:],
down). The HF f64 reference is therefore identical to `full_mixtral.json`.

Writes:
  - fixtures/moe/mixtral_classic.safetensors
  - fixtures/moe/mixtral_classic_config.json  (eos_token_id=20, matches the
        full_mixtral greedy reference [17,20,...])
"""

import json
import os

import numpy as np
from safetensors.numpy import load_file, save_file

OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    j = json.load(open(os.path.join(OUT, "full_mixtral.json")))
    hidden = j["hidden_size"]
    d_ff = j["intermediate_size"]
    ne = j["num_local_experts"]
    n_layers = j["num_hidden_layers"]

    src = load_file(os.path.join(OUT, "full_mixtral.safetensors"))
    out = {}
    for name, t in src.items():
        if ".mlp.experts.gate_up_proj" in name or ".mlp.experts.down_proj" in name:
            continue
        if name.endswith(".mlp.gate.weight"):
            # router → classic router name
            out[name.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")] = t
        else:
            out[name] = t

    for l in range(n_layers):
        gu = src[f"model.layers.{l}.mlp.experts.gate_up_proj"]  # [ne, 2*d_ff, hidden]
        dn = src[f"model.layers.{l}.mlp.experts.down_proj"]      # [ne, hidden, d_ff]
        assert gu.shape == (ne, 2 * d_ff, hidden), gu.shape
        assert dn.shape == (ne, hidden, d_ff), dn.shape
        for e in range(ne):
            w1 = gu[e, :d_ff, :]        # gate
            w3 = gu[e, d_ff:2 * d_ff, :]  # up
            w2 = dn[e]                   # down
            p = f"model.layers.{l}.block_sparse_moe.experts.{e}"
            out[f"{p}.w1.weight"] = np.ascontiguousarray(w1)
            out[f"{p}.w3.weight"] = np.ascontiguousarray(w3)
            out[f"{p}.w2.weight"] = np.ascontiguousarray(w2)

    save_file(out, os.path.join(OUT, "mixtral_classic.safetensors"))

    cfg = dict(
        model_type="mixtral", architectures=["MixtralForCausalLM"],
        vocab_size=j["vocab_size"], hidden_size=hidden, intermediate_size=d_ff,
        num_hidden_layers=n_layers, num_attention_heads=j["num_attention_heads"],
        num_key_value_heads=j["num_key_value_heads"], head_dim=j["head_dim"],
        num_local_experts=ne, num_experts_per_tok=j["num_experts_per_tok"],
        rope_theta=j["rope_theta"], rms_norm_eps=j["rms_norm_eps"],
        tie_word_embeddings=False, eos_token_id=20,
    )
    json.dump(cfg, open(os.path.join(OUT, "mixtral_classic_config.json"), "w"), indent=2)

    print("WROTE mixtral_classic.safetensors tensors=", len(out))
    print("classic expert names sample:",
          [k for k in out if "experts.0" in k][:3])


if __name__ == "__main__":
    main()
