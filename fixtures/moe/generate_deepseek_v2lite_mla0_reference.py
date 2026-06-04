#!/usr/bin/env python3
"""MLA-0 — tiny DeepSeek-V2-Lite-like reference (YaRN + dense-first layer).

OFFLINE reproducibility artifact. NOT run in CI, NOT imported by Rust.

Builds a real `DeepseekV2ForCausalLM` configured like DeepSeek-V2-Lite for the
three features MLA-0 adds, so Atenia's experimental MLA path can be validated:
  - **YaRN** rope_scaling active (so inv_freq is reparametrised AND the attention
    softmax scale carries mscale^2 — at *every* position, not just long context);
    mscale == mscale_all_dim so the cos/sin _mscale cancels to 1.0 (the V2-Lite
    case), keeping the change to inv_freq + softmax-scale only.
  - **first_k_dense_replace = 1** → layer 0 is a DENSE SwiGLU MLP, layers 1.. are MoE.
  - **q_lora_rank = None** (the V2-Lite variant) + **norm_topk_prob = False**
    (no top-k renormalisation; ungated shared expert).

Writes:
  deepseek_v2lite_mla0.safetensors  all weights F32
  deepseek_v2lite_mla0_config.json  HF config (Atenia reads this)
  deepseek_v2lite_mla0.json         input_ids + f64 logits + greedy ids + eos
"""

import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 0x2117E0A5

# YaRN parameters (mscale == mscale_all_dim → cos/sin _mscale cancels to 1.0).
ROPE_SCALING = {
    "type": "yarn",
    "factor": 8,
    "original_max_position_embeddings": 16,
    "mscale": 0.707,
    "mscale_all_dim": 0.707,
    "beta_fast": 32,
    "beta_slow": 1,
}

CFG = dict(
    vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=16,
    num_hidden_layers=3, num_attention_heads=4,
    n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
    first_k_dense_replace=1, moe_layer_freq=1,
    topk_method="greedy", n_group=1, topk_group=1, routed_scaling_factor=1.0,
    scoring_func="softmax", norm_topk_prob=False,
    q_lora_rank=None, kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
    max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-5,
    attention_bias=False, tie_word_embeddings=False,
)


def main():
    torch.manual_seed(SEED)
    cfg = DeepseekV2Config(rope_scaling=ROPE_SCALING, **CFG)
    model = DeepseekV2ForCausalLM(cfg)
    model.eval()
    model.double()

    seq = 5
    rng = np.random.default_rng(SEED)
    input_ids = rng.integers(0, cfg.vocab_size, size=(1, seq)).astype(np.int64)

    with torch.no_grad():
        logits = model(torch.tensor(input_ids)).logits.reshape(-1).numpy().astype(np.float64)

    # Greedy (full recompute) for EOS demo.
    sequence = input_ids.reshape(-1).tolist()
    greedy = []
    with torch.no_grad():
        for _ in range(4):
            row = model(torch.tensor([sequence])).logits[0, -1].numpy().astype(np.float64)
            t = int(np.argmax(row))
            greedy.append(t)
            sequence.append(t)
    eos = greedy[1]

    sd = model.state_dict()
    fixture = {k: t.detach().float().numpy().astype(np.float32)
               for k, t in sd.items() if t.dtype.is_floating_point}
    save_file({k: np.ascontiguousarray(v) for k, v in fixture.items()},
              os.path.join(OUT, "deepseek_v2lite_mla0.safetensors"))

    hf_config = dict(
        model_type="deepseek_v2", architectures=["DeepseekV2ForCausalLM"],
        eos_token_id=eos, rope_scaling=ROPE_SCALING, **CFG,
    )
    json.dump(hf_config, open(os.path.join(OUT, "deepseek_v2lite_mla0_config.json"), "w"), indent=2)

    sidecar = dict(
        source="DeepseekV2ForCausalLM (MLA + YaRN + first_k_dense_replace=1) f64 reference",
        seq=seq, vocab_size=cfg.vocab_size, hidden=cfg.hidden_size,
        input_ids=input_ids.reshape(-1).tolist(),
        hf_logits=logits.tolist(),
        greedy_ids=greedy, eos_token_id=eos,
    )
    json.dump(sidecar, open(os.path.join(OUT, "deepseek_v2lite_mla0.json"), "w"))

    print("WROTE deepseek_v2lite_mla0.safetensors bytes=",
          os.path.getsize(os.path.join(OUT, "deepseek_v2lite_mla0.safetensors")))
    print("greedy_ids=", greedy, "eos=", eos)
    print("=== layer-0 (dense) + layer-1 (moe) tensor names ===")
    for k in sorted(fixture):
        if "layers.0." in k or "layers.1.mlp" in k:
            print("  ", k, tuple(fixture[k].shape))


if __name__ == "__main__":
    main()
