#!/usr/bin/env python3
"""MIXTRAL-CERT-3 / C5 — active-path F64 reference for Mixtral-8x7B-v0.1.

ADR-007 C5 certifies the REAL model, as actually run, end-to-end on a canonical
input vs an external reference over the active subgraph. A global-F64 forward is
infeasible (the whole model in F64 is ~374 GB), so we compute the reference **one
decoder layer at a time in float64**, reusing HuggingFace's OWN trusted Mixtral
layer module (GQA attention + RoPE + the sparse MoE block). Peak RAM is one layer
in F64 (~few GB), never the whole model -> NOT L4.

Mirrors generate_deepseek_v2lite_c5_reference.py, swapping DeepseekV2 for Mixtral:
standard GQA attention (no MLA, no bias), standard RoPE (theta 1e6, no YaRN, no
sliding window for v0.1), classic experts (block_sparse_moe.experts.{e}.{w1,w3,w2})
converted to the packed gate_up_proj/down_proj that transformers 5.x expects.

**Resumable** — the running hidden state is checkpointed after each layer (atomic),
so a reaped run (HDD / ~60-min background reaping) resumes from the last layer.

Modes:
  python generate_mixtral_c5_reference.py tiny
      Validate the driver vs the committed tiny mixtral_scale fixture's HF f64 logits.
  python generate_mixtral_c5_reference.py real <model_dir> [out_dir]
      Compute the C5 reference for the real Mixtral and write
      mixtral_c5_ref.{safetensors,json}.
"""

import gc
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralRMSNorm,
    MixtralRotaryEmbedding,
)

CANONICAL_INPUT_IDS = [1, 100, 200, 300]


class Weights:
    """Lazily reads (possibly sharded, bf16) safetensors tensors as f64 torch."""

    def __init__(self, model_dir, single_file=None):
        self.model_dir = model_dir
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        self.handles = {}
        if single_file is not None:
            self.index = None
            self.single = safe_open(os.path.join(model_dir, single_file), framework="pt")
        elif os.path.exists(idx_path):
            self.index = json.load(open(idx_path))["weight_map"]
            self.single = None
        else:
            self.index = None
            self.single = safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt")

    def _h(self, name):
        if self.single is not None:
            return self.single
        shard = self.index[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(os.path.join(self.model_dir, shard), framework="pt")
        return self.handles[shard]

    def f64(self, name):
        return self._h(name).get_tensor(name).to(torch.float64)

    def has(self, name):
        if self.single is not None:
            return name in self.single.keys()
        return name in self.index


# The real Mixtral-8x7B-v0.1 checkpoint uses the classic names
# `block_sparse_moe.gate` + `block_sparse_moe.experts.{e}.{w1,w3,w2}`; the tiny
# mixtral_scale fixture (saved by transformers 5.x) uses `mlp.gate` + packed
# `mlp.experts.gate_up_proj/down_proj`. These helpers read either. For the REAL
# model the classic-per-expert branch is taken (one expert at a time -> no OOM);
# the packed branch is only hit by the tiny fixture (trivially small).
def moe_gate(w, base):
    for n in (f"{base}.block_sparse_moe.gate.weight", f"{base}.mlp.gate.weight"):
        if w.has(n):
            return w.f64(n)
    raise KeyError(f"no MoE gate for {base}")


def expert_proj(w, base, e, which):
    """which in {w1 (gate), w3 (up), w2 (down)} -> that expert's f64 weight."""
    classic = f"{base}.block_sparse_moe.experts.{e}.{which}.weight"
    if w.has(classic):
        return w.f64(classic)
    # packed fallback (tiny fixture): mlp.experts.gate_up_proj [E,2*inter,hidden],
    # down_proj [E,hidden,inter]; gate = first half, up = second half.
    if which in ("w1", "w3"):
        gup = w.f64(f"{base}.mlp.experts.gate_up_proj")
        inter = gup.shape[1] // 2
        return gup[e, :inter] if which == "w1" else gup[e, inter:]
    return w.f64(f"{base}.mlp.experts.down_proj")[e]


@torch.no_grad()
def forward_logits_f64(model_dir, cfg, input_ids, single_file=None, ckpt_dir=None):
    cfg._attn_implementation = "eager"
    w = Weights(model_dir, single_file=single_file)
    seq = len(input_ids)
    ids = torch.tensor(input_ids, dtype=torch.long)

    # Resume: find the highest completed-layer checkpoint (running hidden state).
    start_layer = 0
    h = None
    if ckpt_dir is not None and os.path.isdir(ckpt_dir):
        done = sorted(
            int(f[2:-4]) for f in os.listdir(ckpt_dir)
            if f.startswith("h_") and f.endswith(".npy") and f[2:-4].isdigit()
        )
        if done:
            start_layer = done[-1] + 1
            h = torch.tensor(np.load(os.path.join(ckpt_dir, f"h_{done[-1]:02d}.npy")), dtype=torch.float64)
            print(f"[mixtral-c5] resume from layer {start_layer} (loaded h_{done[-1]:02d})", flush=True)

    if h is None:
        embed = w.f64("model.embed_tokens.weight")
        h = embed[ids].unsqueeze(0)  # [1, seq, hidden]
        del embed
        gc.collect()

    rotary = MixtralRotaryEmbedding(cfg)
    position_ids = torch.arange(seq).unsqueeze(0)
    pos_emb = rotary(h.to(torch.float32), position_ids)  # (cos, sin)
    pos_emb = (pos_emb[0].to(torch.float64), pos_emb[1].to(torch.float64))
    mask = torch.triu(torch.full((seq, seq), float("-inf"), dtype=torch.float64), diagonal=1)[None, None]

    hidden = cfg.hidden_size
    top_k = cfg.num_experts_per_tok
    silu = torch.nn.functional.silu
    for l in range(start_layer, cfg.num_hidden_layers):
        base = f"model.layers.{l}"
        # 1) input RMSNorm -> HF GQA attention -> residual. Memory-safe: HF attention
        #    loads only q/k/v/o proj (small), reproduces RoPE + GQA + softmax exactly.
        in_ln = MixtralRMSNorm(hidden, eps=cfg.rms_norm_eps).to(torch.float64)
        with torch.no_grad():
            in_ln.weight.copy_(w.f64(f"{base}.input_layernorm.weight"))
        attn = MixtralAttention(cfg, l).to(torch.float64).eval()
        attn.load_state_dict(
            {k: w.f64(f"{base}.self_attn.{k}") for k in attn.state_dict().keys()}, assign=True
        )
        attn_out, _ = attn(in_ln(h), position_embeddings=pos_emb, attention_mask=mask)
        x1 = h + attn_out
        del attn
        gc.collect()

        # 2) post-attn RMSNorm -> sparse MoE, computed over the ACTIVE subgraph only
        #    (ADR-007 "F64 over the active subgraph"): one routed expert at a time in
        #    f64, never the full 8-expert packed tensor (which OOMs for Mixtral's
        #    14336-wide experts). Router = softmax(float32) -> top-2 -> renorm (HF).
        post_ln = MixtralRMSNorm(hidden, eps=cfg.rms_norm_eps).to(torch.float64)
        with torch.no_grad():
            post_ln.weight.copy_(w.f64(f"{base}.post_attention_layernorm.weight"))
        h2 = post_ln(x1)[0]  # [seq, hidden]
        gate_w = moe_gate(w, base)  # [E, hidden]
        rw = torch.softmax(h2 @ gate_w.T, dim=-1, dtype=torch.float32)  # [seq, E] (HF float32)
        topw, topi = torch.topk(rw, top_k, dim=-1)                      # [seq, k]
        topw = (topw / topw.sum(-1, keepdim=True)).to(torch.float64)    # renorm -> f64
        moe = torch.zeros((seq, hidden), dtype=torch.float64)
        for e in sorted({int(x) for x in topi.flatten()}):
            w1 = expert_proj(w, base, e, "w1")  # gate [inter,hidden]
            w3 = expert_proj(w, base, e, "w3")  # up   [inter,hidden]
            w2 = expert_proj(w, base, e, "w2")  # down [hidden,inter]
            for t in range(seq):
                for kk in range(top_k):
                    if int(topi[t, kk]) == e:
                        y = silu(w1 @ h2[t]) * (w3 @ h2[t])
                        moe[t] = moe[t] + topw[t, kk] * (w2 @ y)
            del w1, w3, w2
            gc.collect()
        h = (x1[0] + moe).unsqueeze(0)

        if ckpt_dir is not None:
            tmp = os.path.join(ckpt_dir, f"h_{l:02d}_tmp")  # np.save appends .npy -> h_NN_tmp.npy
            np.save(tmp, h.numpy())
            os.replace(tmp + ".npy", os.path.join(ckpt_dir, f"h_{l:02d}.npy"))
        print(f"[mixtral-c5] layer {l:2d} done max|h|={float(h.abs().max()):.3f}", flush=True)

    norm = MixtralRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(torch.float64)
    with torch.no_grad():
        norm.weight.copy_(w.f64("model.norm.weight"))
    h = norm(h)
    lm = w.f64("lm_head.weight") if w.has("lm_head.weight") else w.f64("model.embed_tokens.weight")
    logits = (h @ lm.T)[0]
    return logits.numpy().astype(np.float64)


def load_cfg(model_dir):
    return MixtralConfig(**json.load(open(os.path.join(model_dir, "config.json"))))


def mode_tiny():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = MixtralConfig(**json.load(open(os.path.join(here, "mixtral_scale_config.json"))))
    ref = json.load(open(os.path.join(here, "mixtral_scale.json")))
    ids = ref["input_ids"]
    vocab = ref["vocab_size"]
    hf = np.array(ref["hf_logits"], dtype=np.float64).reshape(len(ids), vocab)
    got = forward_logits_f64(here, cfg, ids, single_file="mixtral_scale.safetensors")
    diff = float(np.max(np.abs(got - hf)))
    am_got = [int(x) for x in got.argmax(axis=1)]
    am_hf = [int(x) for x in hf.argmax(axis=1)]
    print(f"[mixtral-c5][VALIDATE tiny] max_abs_diff vs committed HF f64 = {diff:.3e}")
    print(f"[mixtral-c5][VALIDATE tiny] argmax driver={am_got} hf={am_hf} match={am_got == am_hf}")
    ok = diff < 1e-5 and am_got == am_hf
    print(f"[mixtral-c5][VALIDATE tiny] DRIVER {'VALIDATED' if ok else 'FAILED -- do not trust downstream'}")
    sys.exit(0 if ok else 1)


def mode_real(model_dir, out_dir):
    cfg = load_cfg(model_dir)
    ids = CANONICAL_INPUT_IDS
    ckpt_dir = os.path.join(out_dir, ".mixtral_c5_ref_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[mixtral-c5][REAL] {model_dir} layers={cfg.num_hidden_layers} experts={cfg.num_local_experts} "
          f"top_k={cfg.num_experts_per_tok} input_ids={ids}", flush=True)
    logits = forward_logits_f64(model_dir, cfg, ids, ckpt_dir=ckpt_dir)
    argmax = [int(x) for x in logits.argmax(axis=1)]
    os.makedirs(out_dir, exist_ok=True)
    st = os.path.join(out_dir, "mixtral_c5_ref.safetensors")
    save_file({"logits": logits.astype(np.float32)}, st)
    meta = {
        "milestone": "MIXTRAL-CERT-3 / C5",
        "obligation": "C5 active-path (F64, one layer at a time)",
        "model": "Mixtral-8x7B-v0.1",
        "oracle": "HuggingFace MixtralDecoderLayer in float64, ONE layer at a time "
                  "(trusted HF code: GQA + RoPE + sparse MoE; never the whole model in F64 -> not L4)",
        "input_ids": ids, "seq": len(ids), "vocab_size": cfg.vocab_size,
        "argmax_per_position": argmax, "reference_file": "mixtral_c5_ref.safetensors",
        "note": "Real Mixtral-8x7B-v0.1 weights, full end-to-end forward. ADR-007 C5 F64 form. "
                "Driver validated against the tiny mixtral_scale fixture before this run.",
    }
    json.dump(meta, open(os.path.join(out_dir, "mixtral_c5_ref.json"), "w"), indent=2)
    print(f"[mixtral-c5][REAL] wrote {st} ({os.path.getsize(st)} bytes)")
    print(f"[mixtral-c5][REAL] argmax_per_position={argmax} |logits|inf={float(np.max(np.abs(logits))):.4f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    if sys.argv[1] == "tiny":
        mode_tiny()
    elif sys.argv[1] == "real":
        model_dir = sys.argv[2]
        out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.abspath(__file__))
        mode_real(model_dir, out_dir)
    else:
        print(__doc__); sys.exit(2)


if __name__ == "__main__":
    main()
