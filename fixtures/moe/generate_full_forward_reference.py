#!/usr/bin/env python3
"""MOE-FULL-6 — generate a tiny FULL Mixtral checkpoint + HF f64 reference.

OFFLINE reproducibility artifact. NOT run in CI, NOT imported by Rust.

Builds a *real* `MixtralForCausalLM` with a tiny config (small vocab/hidden so
the fixture is a few KB), runs the HF forward in **f64** (`model.double()`) on a
fixed input token sequence, and writes:
  - fixtures/moe/full_mixtral.safetensors : all weights, F32 (what Atenia reads)
  - fixtures/moe/full_mixtral.json        : config + input_ids + HF f64 logits

Scope simplification (documented): num_key_value_heads == num_attention_heads
(MHA, no GQA) to keep Atenia's experimental full-forward graph minimal. RoPE,
causal mask, multi-token, MoE experts, final norm and lm_head are all real.
"""

import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import MixtralConfig, MixtralForCausalLM

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 0x46554C36  # "FUL6"


def main():
    torch.manual_seed(SEED)
    # Capture the hyperparameters locally; do NOT read them back from the HF
    # config object afterwards (attribute names vary across transformers
    # versions, e.g. rope_theta may live under rope_parameters).
    p = dict(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,  # MHA (no GQA) — documented simplification
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
        logits = out.logits.reshape(-1).detach().numpy().astype(np.float64)  # [seq*vocab]

    # Export every weight as F32 (what Atenia consumes). Mixtral on this
    # transformers version may store experts packed (gate_up_proj/down_proj)
    # or classic (w1/w3/w2) — we just dump whatever state_dict has.
    sd = model.state_dict()
    fixture = {}
    for name, t in sd.items():
        if t.dtype.is_floating_point:
            fixture[name] = t.detach().float().numpy().astype(np.float32)
    st_path = os.path.join(OUT, "full_mixtral.safetensors")
    save_file({k: np.ascontiguousarray(v) for k, v in fixture.items()}, st_path)

    sidecar = dict(
        source="synthetic MixtralForCausalLM (tiny config; MHA, no GQA)",
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
        hf_logits=logits.tolist(),  # length seq*vocab, row-major [seq, vocab]
    )
    json.dump(sidecar, open(os.path.join(OUT, "full_mixtral.json"), "w"))

    # Report tensor names + sizes for the audit.
    names = sorted(fixture.keys())
    total = sum(v.size * 4 for v in fixture.values())
    print("WROTE", st_path, "bytes=", os.path.getsize(st_path))
    print("WROTE", os.path.join(OUT, "full_mixtral.json"))
    print("num_tensors=", len(names), "weight_bytes=", total)
    l0 = [n for n in names if "layers.0." in n] + [
        n for n in names if n in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
    ]
    for n in sorted(set(l0)):
        print("  ", n, tuple(fixture[n].shape))


if __name__ == "__main__":
    main()
