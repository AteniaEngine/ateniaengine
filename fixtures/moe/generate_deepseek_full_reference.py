#!/usr/bin/env python3
"""MOE-FULL-12 — tiny DeepSeek-V2 (MLA) full reference + config.json.

OFFLINE reproducibility artifact. NOT run in CI, NOT imported by Rust.

Builds a real `DeepseekV2ForCausalLM` (MLA attention, low-rank KV, decoupled
interleaved RoPE; MoE with simple routing so the block reduces to the certified
top-k softmax + renorm + ungated-shared convention) and writes:
  - fixtures/moe/deepseek_full.safetensors  : all weights F32
  - fixtures/moe/deepseek_full_config.json  : HF config (eos = greedy step-1 tok)
  - fixtures/moe/deepseek_full.json         : input_ids, f64 logits, greedy ids,
        and a layer-0 self-attention oracle (post-input-LN input + output) to
        validate the MLA attention in isolation.
"""

import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 0xDE125EE2


def main():
    torch.manual_seed(SEED)
    cfg = DeepseekV2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=4,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        first_k_dense_replace=0, moe_layer_freq=1,
        topk_method="greedy", n_group=1, topk_group=1, routed_scaling_factor=1.0,
        scoring_func="softmax", norm_topk_prob=True,
        q_lora_rank=None, kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
        max_position_embeddings=64, rope_theta=10000.0, rms_norm_eps=1e-5,
        attention_bias=False, tie_word_embeddings=False,
    )
    model = DeepseekV2ForCausalLM(cfg)
    model.eval()
    model.double()

    seq = 5
    rng = np.random.default_rng(SEED)
    input_ids = rng.integers(0, cfg.vocab_size, size=(1, seq)).astype(np.int64)

    # Capture layer-0 self-attention input (post input_layernorm) + output.
    captured = {}

    def hook(module, args, kwargs, out):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        captured["attn_in"] = hs.detach().reshape(-1).numpy().astype(np.float64)
        o = out[0] if isinstance(out, tuple) else out
        captured["attn_out"] = o.detach().reshape(-1).numpy().astype(np.float64)

    h = model.model.layers[0].self_attn.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        logits = model(torch.tensor(input_ids)).logits.reshape(-1).numpy().astype(np.float64)
    h.remove()

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
    fixture = {k: t.detach().float().numpy().astype(np.float32) for k, t in sd.items() if t.dtype.is_floating_point}
    save_file({k: np.ascontiguousarray(v) for k, v in fixture.items()},
              os.path.join(OUT, "deepseek_full.safetensors"))

    hf_config = dict(
        model_type="deepseek_v2", architectures=["DeepseekV2ForCausalLM"],
        vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=4,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        first_k_dense_replace=0,
        q_lora_rank=None, kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
        norm_topk_prob=True, routed_scaling_factor=1.0,
        max_position_embeddings=64, rope_theta=10000.0, rms_norm_eps=1e-5,
        attention_bias=False, tie_word_embeddings=False, eos_token_id=eos,
    )
    json.dump(hf_config, open(os.path.join(OUT, "deepseek_full_config.json"), "w"), indent=2)

    sidecar = dict(
        source="DeepseekV2ForCausalLM (MLA) f64 reference",
        seq=seq, vocab_size=32, hidden=16,
        input_ids=input_ids.reshape(-1).tolist(),
        hf_logits=logits.tolist(),
        greedy_ids=greedy, eos_token_id=eos,
        attn0_in=captured["attn_in"].tolist(),
        attn0_out=captured["attn_out"].tolist(),
    )
    json.dump(sidecar, open(os.path.join(OUT, "deepseek_full.json"), "w"))

    print("WROTE deepseek_full.safetensors bytes=", os.path.getsize(os.path.join(OUT, "deepseek_full.safetensors")))
    print("greedy_ids=", greedy, "eos=", eos)
    attn_keys = sorted(k for k in fixture if "layers.0.self_attn" in k)
    for k in attn_keys:
        print("  ", k, tuple(fixture[k].shape))


if __name__ == "__main__":
    main()
