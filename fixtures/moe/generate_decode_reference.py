#!/usr/bin/env python3
"""MOE-FULL-7 — generate an offline HuggingFace f64 GREEDY reference for the
experimental MoE decode loop.

OFFLINE reproducibility artifact. NOT run in CI, NOT imported by Rust.

Reuses the **already-committed** tiny Mixtral weights
(`fixtures/moe/full_mixtral.safetensors`, MOE-FULL-6) — loaded into a real
`MixtralForCausalLM` and run in **f64** (`model.double()`) — so the reference
is bound to exactly the weights Atenia consumes (no re-init / seed drift).

Greedy decoding is computed by the most robust possible oracle: a **full
recompute every step** (run the whole prefix through the model, argmax the last
position, append, repeat). This is mathematically identical to KV-cached greedy
decoding, so Atenia's prefill+decode loop must reproduce it. Writes:

  - fixtures/moe/full_mixtral_gen.json : prompt_ids, generated_ids, and the
    per-step f64 logits row that produced each generated token.
"""

import json
import os

import numpy as np
import torch
from safetensors.numpy import load_file
from transformers import MixtralConfig, MixtralForCausalLM

OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    j = json.load(open(os.path.join(OUT, "full_mixtral.json")))
    cfg = MixtralConfig(
        vocab_size=j["vocab_size"],
        hidden_size=j["hidden_size"],
        intermediate_size=j["intermediate_size"],
        num_hidden_layers=j["num_hidden_layers"],
        num_attention_heads=j["num_attention_heads"],
        num_key_value_heads=j["num_key_value_heads"],
        head_dim=j["head_dim"],
        max_position_embeddings=64,
        rope_theta=j["rope_theta"],
        rms_norm_eps=j["rms_norm_eps"],
        num_local_experts=j["num_local_experts"],
        num_experts_per_tok=j["num_experts_per_tok"],
        output_router_logits=False,
        tie_word_embeddings=j["tie_word_embeddings"],
        attention_bias=False,
        sliding_window=None,
    )
    model = MixtralForCausalLM(cfg)

    # Load the committed weights (F32 on disk) into the model, then go f64.
    weights = load_file(os.path.join(OUT, "full_mixtral.safetensors"))
    sd = {k: torch.from_numpy(v.astype(np.float32)) for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected tensors: {unexpected}"
    # Only the rotary inv_freq buffers may be 'missing' (recomputed from config).
    assert all("rotary" in m or "inv_freq" in m for m in missing), f"missing: {missing}"
    model.eval()
    model.double()

    # Greedy generation by full recompute (the KV-cache oracle).
    prompt = [int(x) for x in j["input_ids"][:3]]  # 3-token prompt
    max_new = 4
    seq = list(prompt)
    generated = []
    step_logits = []  # one f64 vocab row per generated token
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(torch.tensor([seq])).logits  # [1, len, vocab]
            row = logits[0, -1].detach().numpy().astype(np.float64)  # [vocab]
            tok = int(np.argmax(row))
            step_logits.append(row.tolist())
            generated.append(tok)
            seq.append(tok)

    sidecar = dict(
        source="greedy f64 full-recompute on committed full_mixtral weights",
        note="MOE-FULL-7 decode reference; KV-cached decode must reproduce this",
        vocab_size=j["vocab_size"],
        prompt_ids=prompt,
        max_new_tokens=max_new,
        generated_ids=generated,
        step_logits=[v for row in step_logits for v in row],  # flat [max_new*vocab]
    )
    json.dump(sidecar, open(os.path.join(OUT, "full_mixtral_gen.json"), "w"))

    print("WROTE", os.path.join(OUT, "full_mixtral_gen.json"))
    print("prompt_ids =", prompt)
    print("generated_ids =", generated)
    print("max_new =", max_new, "vocab =", j["vocab_size"])


if __name__ == "__main__":
    main()
