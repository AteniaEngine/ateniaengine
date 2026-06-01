#!/usr/bin/env python3
"""MOE-FULL-11 — tiny Qwen2-MoE checkpoint + HF f64 reference + config.json.

OFFLINE reproducibility artifact. NOT run in CI, NOT imported by Rust.

Builds a real `Qwen2MoeForCausalLM` (GQA, Q/K/V bias, packed experts, shared
expert with sigmoid gate, norm_topk_prob=False) and writes:
  - fixtures/moe/qwen_moe_tiny.safetensors : all weights F32
  - fixtures/moe/qwen_moe_tiny_config.json : HF config (eos_token_id set to a
        token actually emitted by greedy decoding, so the runtime demonstrates
        load -> generate -> EOS)
  - fixtures/moe/qwen_moe_tiny.json        : input_ids + f64 logits + greedy ids
"""

import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 0x5157454E  # "QWEN"


def main():
    torch.manual_seed(SEED)
    cfg = Qwen2MoeConfig(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,  # every layer is MoE
        mlp_only_layers=[],
        norm_topk_prob=False,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    model = Qwen2MoeForCausalLM(cfg)
    model.eval()
    model.double()

    seq = 5
    rng = np.random.default_rng(SEED)
    input_ids = rng.integers(0, cfg.vocab_size, size=(1, seq)).astype(np.int64)

    with torch.no_grad():
        logits = model(torch.tensor(input_ids)).logits.reshape(-1).numpy().astype(np.float64)

    # Greedy (full recompute) for the EOS demo.
    sequence = input_ids.reshape(-1).tolist()
    greedy = []
    with torch.no_grad():
        for _ in range(4):
            row = model(torch.tensor([sequence])).logits[0, -1].numpy().astype(np.float64)
            t = int(np.argmax(row))
            greedy.append(t)
            sequence.append(t)
    eos = greedy[1]  # emitted at step 1 → runtime stops there

    sd = model.state_dict()
    fixture = {k: t.detach().float().numpy().astype(np.float32) for k, t in sd.items() if t.dtype.is_floating_point}
    save_file({k: np.ascontiguousarray(v) for k, v in fixture.items()},
              os.path.join(OUT, "qwen_moe_tiny.safetensors"))

    hf_config = dict(
        model_type="qwen2_moe",
        architectures=["Qwen2MoeForCausalLM"],
        vocab_size=48, hidden_size=32, intermediate_size=32,
        moe_intermediate_size=16, shared_expert_intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
        norm_topk_prob=False, max_position_embeddings=64, rope_theta=10000.0,
        rms_norm_eps=1e-5, tie_word_embeddings=False, eos_token_id=eos,
    )
    json.dump(hf_config, open(os.path.join(OUT, "qwen_moe_tiny_config.json"), "w"), indent=2)

    sidecar = dict(
        source="synthetic Qwen2MoeForCausalLM (GQA, qkv bias, packed experts, shared expert)",
        seq=seq, vocab_size=48,
        input_ids=input_ids.reshape(-1).tolist(),
        hf_logits=logits.tolist(),
        greedy_ids=greedy,
        eos_token_id=eos,
    )
    json.dump(sidecar, open(os.path.join(OUT, "qwen_moe_tiny.json"), "w"))

    print("WROTE qwen_moe_tiny.safetensors bytes=",
          os.path.getsize(os.path.join(OUT, "qwen_moe_tiny.safetensors")))
    print("greedy_ids=", greedy, "eos=", eos)
    print("q_proj.bias present:", "model.layers.0.self_attn.q_proj.bias" in fixture)


if __name__ == "__main__":
    main()
