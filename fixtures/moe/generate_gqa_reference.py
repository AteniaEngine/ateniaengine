#!/usr/bin/env python3
"""MOE-FULL-9 — generate a tiny GQA Mixtral checkpoint + HF f64 reference.

OFFLINE reproducibility artifact. NOT run in CI, NOT imported by Rust.

Like generate_full_forward_reference.py (MOE-FULL-6) but with **GQA**:
num_key_value_heads (2) != num_attention_heads (4). Validates that Atenia's
MoE full-forward reproduces HuggingFace when the K/V heads are tiled by
kv_groups = num_attention_heads / num_key_value_heads = 2.

Writes:
  - fixtures/moe/gqa_mixtral.safetensors : all weights F32 (what Atenia reads)
  - fixtures/moe/gqa_mixtral.json        : config + input_ids + HF f64 logits
"""

import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import MixtralConfig, MixtralForCausalLM

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 0x46554C39  # "FUL9"


def main():
    torch.manual_seed(SEED)
    p = dict(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: kv_groups = 4 / 2 = 2
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        num_local_experts=4,
        num_experts_per_tok=2,
        output_router_logits=False,
        tie_word_embeddings=False,
        attention_bias=False,
        sliding_window=None,
    )
    cfg = MixtralConfig(**p)
    model = MixtralForCausalLM(cfg)
    model.eval()
    model.double()

    seq = 5
    rng = np.random.default_rng(SEED)
    input_ids = rng.integers(0, cfg.vocab_size, size=(1, seq)).astype(np.int64)

    with torch.no_grad():
        out = model(torch.tensor(input_ids))
        logits = out.logits.reshape(-1).detach().numpy().astype(np.float64)

    sd = model.state_dict()
    fixture = {}
    for name, t in sd.items():
        if t.dtype.is_floating_point:
            fixture[name] = t.detach().float().numpy().astype(np.float32)
    st_path = os.path.join(OUT, "gqa_mixtral.safetensors")
    save_file({k: np.ascontiguousarray(v) for k, v in fixture.items()}, st_path)

    sidecar = dict(
        source="synthetic MixtralForCausalLM (tiny config; GQA n_kv=2 != n_heads=4)",
        transformers_note="real HF Mixtral forward, f64 reference",
        vocab_size=p["vocab_size"],
        hidden_size=p["hidden_size"],
        num_hidden_layers=p["num_hidden_layers"],
        num_attention_heads=p["num_attention_heads"],
        num_key_value_heads=p["num_key_value_heads"],
        head_dim=p["head_dim"],
        intermediate_size=p["intermediate_size"],
        num_local_experts=p["num_local_experts"],
        num_experts_per_tok=p["num_experts_per_tok"],
        rope_theta=p["rope_theta"],
        rms_norm_eps=p["rms_norm_eps"],
        tie_word_embeddings=p["tie_word_embeddings"],
        seq=seq,
        input_ids=input_ids.reshape(-1).tolist(),
        hf_logits=logits.tolist(),
    )
    json.dump(sidecar, open(os.path.join(OUT, "gqa_mixtral.json"), "w"))

    print("WROTE", st_path, "bytes=", os.path.getsize(st_path))
    print("k_proj shape:", tuple(fixture["model.layers.0.self_attn.k_proj.weight"].shape))
    print("q_proj shape:", tuple(fixture["model.layers.0.self_attn.q_proj.weight"].shape))


if __name__ == "__main__":
    main()
