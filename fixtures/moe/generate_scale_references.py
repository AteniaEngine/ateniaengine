#!/usr/bin/env python3
"""MOE-FULL-15 — topology-representative SCALE certification fixtures.

OFFLINE reproducibility artifact. NOT run in CI, NOT imported by Rust.

Real large MoE checkpoints (Mixtral-8x7B ~47 GB, Qwen2-57B, DeepSeek-V2) cannot
be downloaded or committed. Instead we certify the **real topology** of each
family — expert count, top-k routing, GQA ratio, shared experts, MLA — at a
reduced hidden dim, end to end vs a HuggingFace f64 forward. This certifies that
the runtime handles the real routing/topology (NOT the 47 GB weights).

  - Mixtral 8x7B topology: 8 experts, top-2, GQA 4:1 (n_heads=8, n_kv=2).
  - Qwen-MoE scale: 16 experts, top-4, shared expert (sigmoid gate), GQA, qkv bias.
  - DeepSeek scale: 16 routed, top-6, 2 shared experts, MLA.

Writes, per family: <name>.safetensors, <name>_config.json (HF), <name>.json
(input_ids, f64 logits, greedy ids, eos).
"""

import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import (
    DeepseekV2Config, DeepseekV2ForCausalLM,
    MixtralConfig, MixtralForCausalLM,
    Qwen2MoeConfig, Qwen2MoeForCausalLM,
)

OUT = os.path.dirname(os.path.abspath(__file__))


def dump(model, cfg_hf, name, seq=5, vocab=32, extra=None):
    model.eval()
    model.double()
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    input_ids = rng.integers(0, vocab, size=(1, seq)).astype(np.int64)
    with torch.no_grad():
        logits = model(torch.tensor(input_ids)).logits.reshape(-1).numpy().astype(np.float64)
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
    fixture = {k: t.detach().float().numpy().astype(np.float32) for k, t in sd.items() if t.dtype.is_floating_point}
    save_file({k: np.ascontiguousarray(v) for k, v in fixture.items()}, os.path.join(OUT, f"{name}.safetensors"))
    cfg_hf["eos_token_id"] = eos
    json.dump(cfg_hf, open(os.path.join(OUT, f"{name}_config.json"), "w"), indent=2)
    side = dict(seq=seq, vocab_size=vocab, input_ids=input_ids.reshape(-1).tolist(),
                hf_logits=logits.tolist(), greedy_ids=greedy, eos_token_id=eos)
    if extra:
        side.update(extra)
    json.dump(side, open(os.path.join(OUT, f"{name}.json"), "w"))
    print(f"WROTE {name}: experts/greedy", greedy, "eos", eos,
          "bytes", os.path.getsize(os.path.join(OUT, f"{name}.safetensors")))


def mixtral_scale():
    torch.manual_seed(0x4D495838)
    c = MixtralConfig(vocab_size=32, hidden_size=64, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=64, rope_theta=10000.0, rms_norm_eps=1e-5,
                      num_local_experts=8, num_experts_per_tok=2, tie_word_embeddings=False)
    hf = dict(model_type="mixtral", vocab_size=32, hidden_size=64, intermediate_size=32,
              num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2, head_dim=8,
              num_local_experts=8, num_experts_per_tok=2, rope_theta=10000.0, rms_norm_eps=1e-5,
              tie_word_embeddings=False)
    dump(MixtralForCausalLM(c), hf, "mixtral_scale")


def qwen_scale():
    torch.manual_seed(0x5157454E)
    c = Qwen2MoeConfig(vocab_size=32, hidden_size=64, intermediate_size=32, moe_intermediate_size=16,
                       shared_expert_intermediate_size=24, num_hidden_layers=2, num_attention_heads=8,
                       num_key_value_heads=2, num_experts=16, num_experts_per_tok=4, decoder_sparse_step=1,
                       mlp_only_layers=[], norm_topk_prob=False, max_position_embeddings=64,
                       rope_theta=10000.0, rms_norm_eps=1e-5, tie_word_embeddings=False)
    hf = dict(model_type="qwen2_moe", vocab_size=32, hidden_size=64, intermediate_size=32,
              moe_intermediate_size=16, shared_expert_intermediate_size=24, num_hidden_layers=2,
              num_attention_heads=8, num_key_value_heads=2, num_experts=16, num_experts_per_tok=4,
              norm_topk_prob=False, rope_theta=10000.0, rms_norm_eps=1e-5, tie_word_embeddings=False)
    dump(Qwen2MoeForCausalLM(c), hf, "qwen_scale")


def deepseek_scale():
    torch.manual_seed(0xDEE12FF0)
    c = DeepseekV2Config(vocab_size=32, hidden_size=64, intermediate_size=32, moe_intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=8, n_routed_experts=16, n_shared_experts=2,
                         num_experts_per_tok=6, first_k_dense_replace=0, moe_layer_freq=1,
                         topk_method="greedy", n_group=1, topk_group=1, routed_scaling_factor=1.0,
                         scoring_func="softmax", norm_topk_prob=True, q_lora_rank=None, kv_lora_rank=16,
                         qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, max_position_embeddings=64,
                         rope_theta=10000.0, rms_norm_eps=1e-5, attention_bias=False, tie_word_embeddings=False)
    model = DeepseekV2ForCausalLM(c)
    hf = dict(model_type="deepseek_v2", vocab_size=32, hidden_size=64, intermediate_size=32,
              moe_intermediate_size=16, num_hidden_layers=2, num_attention_heads=8,
              n_routed_experts=16, n_shared_experts=2, num_experts_per_tok=6, first_k_dense_replace=0,
              q_lora_rank=None, kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
              norm_topk_prob=True, routed_scaling_factor=1.0, rope_theta=10000.0, rms_norm_eps=1e-5,
              attention_bias=False, tie_word_embeddings=False)
    dump(model, hf, "deepseek_scale")


def main():
    mixtral_scale()
    qwen_scale()
    deepseek_scale()


if __name__ == "__main__":
    main()
