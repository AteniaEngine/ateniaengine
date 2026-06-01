#!/usr/bin/env python3
"""MOE-FULL-11 — tiny DeepSeek-MoE **MoE-block** HF f64 reference.

OFFLINE reproducibility artifact. NOT run in CI, NOT imported by Rust.

DeepSeek-V2/V3 use MLA attention (a different architecture, out of scope for the
experimental runtime), so we do NOT certify an end-to-end forward. Instead we
certify the **MoE block** (router + packed routed experts + shared expert),
which is where the family differs and which Atenia's `RealMoeLayer` already
models. To make the HF MoE block reduce to the certified top-k softmax + renorm
+ ungated-shared convention we configure simple routing:
  topk_method='greedy', n_group=1, topk_group=1, routed_scaling_factor=1.0,
  scoring_func='softmax', norm_topk_prob=True, n_shared_experts=1.

Writes:
  - fixtures/moe/deepseek_block.safetensors : layer-0 MoE tensors (F32)
  - fixtures/moe/deepseek_block.json        : probe input + HF f64 MoE output
"""

import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 0xDEE75EE6


def main():
    torch.manual_seed(SEED)
    hidden = 16
    moe_ff = 16
    cfg = DeepseekV2Config(
        vocab_size=32, hidden_size=hidden, intermediate_size=32,
        moe_intermediate_size=moe_ff, num_hidden_layers=1,
        num_attention_heads=4, n_routed_experts=4, n_shared_experts=1,
        num_experts_per_tok=2, first_k_dense_replace=0, moe_layer_freq=1,
        topk_method="greedy", n_group=1, topk_group=1,
        routed_scaling_factor=1.0, scoring_func="softmax", norm_topk_prob=True,
        q_lora_rank=None, kv_lora_rank=8, qk_rope_head_dim=4,
        qk_nope_head_dim=4, v_head_dim=4, max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    model = DeepseekV2ForCausalLM(cfg)
    model.eval()
    model.double()

    mlp = model.model.layers[0].mlp  # DeepseekV2MoE
    rng = np.random.default_rng(SEED)
    probe = torch.tensor(rng.standard_normal((1, 1, hidden)), dtype=torch.float64)
    with torch.no_grad():
        out = mlp(probe).reshape(-1).numpy().astype(np.float64)  # [hidden]

    sd = model.state_dict()
    fixture = {}
    for k, t in sd.items():
        if "layers.0.mlp" in k and t.dtype.is_floating_point:
            fixture[k] = t.detach().float().numpy().astype(np.float32)
    save_file({k: np.ascontiguousarray(v) for k, v in fixture.items()},
              os.path.join(OUT, "deepseek_block.safetensors"))

    sidecar = dict(
        source="DeepseekV2MoE block (simple routing) f64 reference",
        hidden=hidden, moe_intermediate_size=moe_ff,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        probe=probe.reshape(-1).numpy().astype(np.float64).tolist(),
        mlp_output=out.tolist(),
    )
    json.dump(sidecar, open(os.path.join(OUT, "deepseek_block.json"), "w"))

    print("WROTE deepseek_block.safetensors")
    print("tensors:", sorted(fixture.keys()))


if __name__ == "__main__":
    main()
