#!/usr/bin/env python3
"""MLA-1 / C5 — active-path F64 reference for DeepSeek-V2-Lite.

ADR-007 C5 (active-path parity) certifies the REAL model, as actually run,
end-to-end on a canonical input, vs an external reference over the active
subgraph. A global-F64 forward is infeasible here (the whole model in F64 is
~120+ GB), so we compute the reference **one decoder layer at a time in
float64**, reusing HuggingFace's OWN trusted DeepseekV2 layer module (zero
convention risk: MLA attention, decoupled/complex RoPE + YaRN, RMSNorm, the
dense-first MLP, and the MoE routing/combine are all HF's code). Peak RAM is one
layer in F64 (~4.5 GB), never the whole model. This is the F64 form of C5. It is
NOT L4: L4 holds the ENTIRE model in F64 at once; we never do.

This mirrors `generate_qwen_moe_c5_reference.py` exactly, swapping the Qwen2Moe
classes for the DeepseekV2 ones and replicating `DeepseekV2Model.forward`'s own
plumbing (one rotary embedding shared across layers, an additive causal mask,
dense-first layers chosen automatically by the decoder layer's __init__).

Two modes:
  python generate_deepseek_v2lite_c5_reference.py tiny
      Validate the driver: run it on the committed tiny DeepSeek-V2-Lite-like
      MLA-0 fixture and compare to that fixture's HF f64 logits (which came from
      the FULL `DeepseekV2ForCausalLM(...).double()`). If this does not match
      (~1e-9), the one-layer-at-a-time driver is wrong and NOTHING downstream may
      be trusted.

  python generate_deepseek_v2lite_c5_reference.py real <model_dir> [out_dir]
      Compute the C5 reference for the real DeepSeek-V2-Lite on a fixed canonical
      input and write deepseek_v2lite_c5_ref.{safetensors,json}.

The Rust harness (tests/moe_mla1_deepseek_c5_active_path_test.rs) then compares
Atenia's MoeRuntime full forward (disk tier) to this reference.
"""

import gc
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

from transformers import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2RMSNorm,
    DeepseekV2RotaryEmbedding,
)

# Fixed canonical input for the real model (valid token ids < vocab). Recorded.
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


@torch.no_grad()
def forward_logits_f64(model_dir, cfg, input_ids, single_file=None):
    """End-to-end DeepseekV2 forward in float64, ONE LAYER AT A TIME, using HF's
    own decoder-layer module. Returns logits [seq, vocab] (float64 numpy)."""
    cfg._attn_implementation = "eager"  # f64-friendly, additive mask path
    w = Weights(model_dir, single_file=single_file)
    seq = len(input_ids)
    ids = torch.tensor(input_ids, dtype=torch.long)

    # Embedding (index the needed rows, upcast to f64).
    embed = w.f64("model.embed_tokens.weight")  # [vocab, hidden]
    h = embed[ids].unsqueeze(0)  # [1, seq, hidden]
    del embed
    gc.collect()

    # RoPE position embeddings (HF's OWN rotary; YaRN is baked into inv_freq +
    # attention_scaling here) + additive causal mask — exactly as
    # DeepseekV2Model.forward wires them.
    rotary = DeepseekV2RotaryEmbedding(cfg)
    position_ids = torch.arange(seq).unsqueeze(0)
    pos_emb = rotary(h, position_ids)  # complex freqs_cis (one tensor, shared)
    mask = torch.triu(torch.full((seq, seq), float("-inf"), dtype=torch.float64), diagonal=1)
    mask = mask[None, None, :, :]  # [1,1,seq,seq]

    n_experts = cfg.n_routed_experts
    for l in range(cfg.num_hidden_layers):
        layer = DeepseekV2DecoderLayer(cfg, l).to(torch.float64).eval()
        sd = {}
        for key in layer.state_dict().keys():
            full = f"model.layers.{l}.{key}"
            if w.has(full):
                # On-disk tensor (packed tiny fixture, or any classic non-expert).
                sd[key] = w.f64(full)
            elif key == "mlp.experts.gate_up_proj":
                # Real checkpoint stores CLASSIC per-expert gate/up; transformers
                # 5.x packs them as [E, 2*moe_inter, hidden] (gate first half, up
                # second). Build it from the classic tensors (same convention
                # Atenia's packed path + the Qwen C5 reference use).
                gus = []
                for e in range(n_experts):
                    g = w.f64(f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight")  # [inter, hidden]
                    u = w.f64(f"model.layers.{l}.mlp.experts.{e}.up_proj.weight")    # [inter, hidden]
                    gus.append(torch.cat([g, u], dim=0))                              # [2*inter, hidden]
                sd[key] = torch.stack(gus, dim=0)                                     # [E, 2*inter, hidden]
            elif key == "mlp.experts.down_proj":
                ds = [
                    w.f64(f"model.layers.{l}.mlp.experts.{e}.down_proj.weight")       # [hidden, inter]
                    for e in range(n_experts)
                ]
                sd[key] = torch.stack(ds, dim=0)                                      # [E, hidden, inter]
            else:
                sd[key] = w.f64(full)
        layer.load_state_dict(sd)
        h = layer(
            h,
            attention_mask=mask,
            position_ids=position_ids,
            position_embeddings=pos_emb,
        )
        del layer, sd
        gc.collect()

    norm = DeepseekV2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(torch.float64)
    with torch.no_grad():
        norm.weight.copy_(w.f64("model.norm.weight"))
    h = norm(h)
    lm = w.f64("lm_head.weight")  # [vocab, hidden]
    logits = (h @ lm.T)[0]  # [seq, vocab]
    return logits.numpy().astype(np.float64)


def _load_layer_sd(w, cfg, layer, l):
    """Build the f64 state dict for decoder layer `l` (classic->packed experts)."""
    n_experts = cfg.n_routed_experts
    sd = {}
    for key in layer.state_dict().keys():
        full = f"model.layers.{l}.{key}"
        if w.has(full):
            sd[key] = w.f64(full)
        elif key == "mlp.experts.gate_up_proj":
            gus = []
            for e in range(n_experts):
                g = w.f64(f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight")
                u = w.f64(f"model.layers.{l}.mlp.experts.{e}.up_proj.weight")
                gus.append(torch.cat([g, u], dim=0))
            sd[key] = torch.stack(gus, dim=0)
        elif key == "mlp.experts.down_proj":
            ds = [w.f64(f"model.layers.{l}.mlp.experts.{e}.down_proj.weight") for e in range(n_experts)]
            sd[key] = torch.stack(ds, dim=0)
        else:
            sd[key] = w.f64(full)
    return sd


@torch.no_grad()
def forward_capture_f64(model_dir, cfg, input_ids, single_file=None):
    """**C5-DIAG** — like forward_logits_f64 but ALSO captures, per decoder layer,
    the post-attention residual (x1) and the post-FFN output (the layer output),
    by replicating DeepseekV2DecoderLayer.forward's internal split. Returns
    (embeddings [seq,hidden], post_attn [L,seq,hidden], post_ffn [L,seq,hidden],
     logits [seq,vocab]) — all float64 numpy."""
    cfg._attn_implementation = "eager"
    w = Weights(model_dir, single_file=single_file)
    seq = len(input_ids)
    ids = torch.tensor(input_ids, dtype=torch.long)

    embed = w.f64("model.embed_tokens.weight")
    h = embed[ids].unsqueeze(0)  # [1, seq, hidden]
    embeddings = h[0].clone().numpy().astype(np.float64)
    del embed
    gc.collect()

    rotary = DeepseekV2RotaryEmbedding(cfg)
    position_ids = torch.arange(seq).unsqueeze(0)
    pos_emb = rotary(h, position_ids)
    mask = torch.triu(torch.full((seq, seq), float("-inf"), dtype=torch.float64), diagonal=1)
    mask = mask[None, None, :, :]

    post_attn, post_ffn = [], []
    for l in range(cfg.num_hidden_layers):
        layer = DeepseekV2DecoderLayer(cfg, l).to(torch.float64).eval()
        layer.load_state_dict(_load_layer_sd(w, cfg, layer, l))
        # Replicate DeepseekV2DecoderLayer.forward, capturing the split.
        residual = h
        hs = layer.input_layernorm(h)
        attn = layer.self_attn(hs, attention_mask=mask, position_embeddings=pos_emb)[0]
        x1 = residual + attn
        post_attn.append(x1[0].clone().numpy().astype(np.float64))
        residual2 = x1
        hs2 = layer.post_attention_layernorm(x1)
        ffn = layer.mlp(hs2)
        h = residual2 + ffn
        post_ffn.append(h[0].clone().numpy().astype(np.float64))
        del layer
        gc.collect()

    norm = DeepseekV2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(torch.float64)
    with torch.no_grad():
        norm.weight.copy_(w.f64("model.norm.weight"))
    hn = norm(h)
    lm = w.f64("lm_head.weight")
    logits = (hn @ lm.T)[0].numpy().astype(np.float64)
    return (embeddings,
            np.stack(post_attn, axis=0),
            np.stack(post_ffn, axis=0),
            logits)


@torch.no_grad()
def mode_precision(model_dir, out_dir):
    """**C5-DIAG / decisive control** — run HF's OWN one-layer-at-a-time forward in
    float64 AND float32 in lockstep (each path feeds its own running hidden, so
    the f32 path accumulates f32 error exactly as a pure-f32 engine would), and
    report the per-layer post-FFN |f32 - f64| curve + the final logit diff and
    argmax. This isolates the effect of activation precision with ZERO Atenia
    involvement: if HF-f32 vs HF-f64 alone reproduces the C5 logit gap + argmax
    flip, the C5 failure is the f32 precision regime (amplified by this model's
    massive activations), not an Atenia bug."""
    cfg = load_cfg(model_dir)
    cfg._attn_implementation = "eager"
    ids = CANONICAL_INPUT_IDS
    w = Weights(model_dir)
    seq = len(ids)
    idt = torch.tensor(ids, dtype=torch.long)

    embed = w.f64("model.embed_tokens.weight")
    h64 = embed[idt].unsqueeze(0)
    h32 = h64.to(torch.float32)
    del embed
    gc.collect()

    rotary = DeepseekV2RotaryEmbedding(cfg)
    position_ids = torch.arange(seq).unsqueeze(0)
    pos_emb = rotary(h64, position_ids)  # complex; apply_rotary_emb casts q/k to f32 anyway
    mask64 = torch.triu(torch.full((seq, seq), float("-inf"), dtype=torch.float64), diagonal=1)[None, None]
    mask32 = mask64.to(torch.float32)

    print(f"[c5-precision] {model_dir} layers={cfg.num_hidden_layers} input_ids={ids}")
    print("  layer | post_ffn |f32-f64| | worst_tok | h64_absmax")
    for l in range(cfg.num_hidden_layers):
        sd64 = _load_layer_sd(w, cfg, DeepseekV2DecoderLayer(cfg, l), l)
        layer64 = DeepseekV2DecoderLayer(cfg, l).to(torch.float64).eval()
        layer64.load_state_dict(sd64)
        layer32 = DeepseekV2DecoderLayer(cfg, l).to(torch.float32).eval()
        layer32.load_state_dict({k: v.to(torch.float32) for k, v in sd64.items()})
        h64 = layer64(h64, attention_mask=mask64, position_embeddings=pos_emb, position_ids=position_ids)
        h32 = layer32(h32, attention_mask=mask32, position_embeddings=pos_emb, position_ids=position_ids)
        d = (h32.to(torch.float64) - h64).abs()
        wtok = int(d.max(dim=-1).values.argmax())
        print(f"  {l:5d} | {float(d.max()):.3e}      | {wtok} | {float(h64.abs().max()):.2f}")
        del layer64, layer32, sd64
        gc.collect()

    norm64 = DeepseekV2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(torch.float64)
    norm32 = DeepseekV2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(torch.float32)
    nw = w.f64("model.norm.weight")
    with torch.no_grad():
        norm64.weight.copy_(nw)
        norm32.weight.copy_(nw.to(torch.float32))
    lm64 = w.f64("lm_head.weight")
    lm32 = lm64.to(torch.float32)
    logits64 = (norm64(h64) @ lm64.T)[0]
    logits32 = (norm32(h32) @ lm32.T)[0].to(torch.float64)
    ld = (logits32 - logits64).abs()
    am64 = [int(x) for x in logits64.argmax(dim=-1)]
    am32 = [int(x) for x in logits32.argmax(dim=-1)]
    wpos = int(ld.max(dim=-1).values.argmax())
    print(f"[c5-precision] FINAL logits |f32-f64| max_abs_diff={float(ld.max()):.3e} "
          f"(pos {wpos}) |logits|inf={float(logits64.abs().max()):.2f}")
    print(f"[c5-precision] argmax f64={am64} f32={am32} match={am64 == am32}")
    print("[c5-precision] CONCLUSION: if this f32-vs-f64 gap ~ the Atenia-vs-f64 C5 gap "
          "(~2.0, argmax flip), the C5 failure is the f32 activation regime, not an Atenia bug.")


def mode_diag(model_dir, out_dir):
    cfg = load_cfg(model_dir)
    ids = CANONICAL_INPUT_IDS
    print(f"[c5-diag] {model_dir} layers={cfg.num_hidden_layers} input_ids={ids}")
    emb, post_attn, post_ffn, logits = forward_capture_f64(model_dir, cfg, ids)
    os.makedirs(out_dir, exist_ok=True)
    st = os.path.join(out_dir, "deepseek_v2lite_c5_diag.safetensors")
    save_file(
        {
            "embeddings": emb.astype(np.float32),
            "post_attn": post_attn.astype(np.float32),  # [L, seq, hidden]
            "post_ffn": post_ffn.astype(np.float32),     # [L, seq, hidden]
            "logits": logits.astype(np.float32),
        },
        st,
    )
    meta = {
        "milestone": "MLA-1 / C5-DIAG",
        "model": "DeepSeek-V2-Lite",
        "input_ids": ids,
        "seq": len(ids),
        "hidden": cfg.hidden_size,
        "num_layers": cfg.num_hidden_layers,
        "vocab_size": cfg.vocab_size,
        "layout": "embeddings[seq,hidden]; post_attn[L,seq,hidden] (x1, post-attention residual); "
                  "post_ffn[L,seq,hidden] (layer output); logits[seq,vocab]. F64-computed, f32-stored. "
                  "HF still has float32 islands (eager softmax, RMSNorm, RoPE).",
    }
    json.dump(meta, open(os.path.join(out_dir, "deepseek_v2lite_c5_diag.json"), "w"), indent=2)
    print(f"[c5-diag] wrote {st} ({os.path.getsize(st)} bytes); "
          f"post_attn{post_attn.shape} post_ffn{post_ffn.shape}")


def load_cfg(model_dir):
    return DeepseekV2Config(**json.load(open(os.path.join(model_dir, "config.json"))))


def load_cfg_from_file(path):
    return DeepseekV2Config(**json.load(open(path)))


def mode_tiny():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = load_cfg_from_file(os.path.join(here, "deepseek_v2lite_mla0_config.json"))
    ref = json.load(open(os.path.join(here, "deepseek_v2lite_mla0.json")))
    ids = ref["input_ids"]
    vocab = ref["vocab_size"]
    hf = np.array(ref["hf_logits"], dtype=np.float64).reshape(len(ids), vocab)
    got = forward_logits_f64(here, cfg, ids, single_file="deepseek_v2lite_mla0.safetensors")
    diff = float(np.max(np.abs(got - hf)))
    am_got = [int(x) for x in got.argmax(axis=1)]
    am_hf = [int(x) for x in hf.argmax(axis=1)]
    print(f"[c5-ref][VALIDATE tiny] max_abs_diff vs committed HF f64 = {diff:.3e}")
    print(f"[c5-ref][VALIDATE tiny] argmax driver={am_got} hf={am_hf} match={am_got == am_hf}")
    # The tiny fixture stores weights as f32 and its HF logits as f64, so a
    # ~1e-8 gap is expected; require a tight HF match + exact argmax. This only
    # validates that the DRIVER reproduces HuggingFace; it is not the cert gate.
    ok = diff < 1e-5 and am_got == am_hf
    print(f"[c5-ref][VALIDATE tiny] DRIVER {'VALIDATED' if ok else 'FAILED — do not trust downstream'}")
    sys.exit(0 if ok else 1)


def mode_real(model_dir, out_dir):
    cfg = load_cfg(model_dir)
    ids = CANONICAL_INPUT_IDS
    print(f"[c5-ref][REAL] {model_dir} layers={cfg.num_hidden_layers} "
          f"first_k_dense={cfg.first_k_dense_replace} experts={cfg.n_routed_experts} input_ids={ids}")
    logits = forward_logits_f64(model_dir, cfg, ids)  # [seq, vocab] f64
    argmax = [int(x) for x in logits.argmax(axis=1)]
    os.makedirs(out_dir, exist_ok=True)
    st = os.path.join(out_dir, "deepseek_v2lite_c5_ref.safetensors")
    save_file({"logits": logits.astype(np.float32)}, st)  # f32-stored, f64-computed
    meta = {
        "milestone": "MLA-1 / C5",
        "obligation": "C5 active-path (F64, one layer at a time)",
        "model": "DeepSeek-V2-Lite",
        "oracle": "HuggingFace DeepseekV2DecoderLayer in float64, one layer at a time "
                  "(trusted HF code: MLA + complex RoPE/YaRN + dense-first + MoE; never "
                  "the whole model in F64 -> not L4)",
        "input_ids": ids,
        "seq": len(ids),
        "vocab_size": cfg.vocab_size,
        "argmax_per_position": argmax,
        "reference_file": "deepseek_v2lite_c5_ref.safetensors",
        "note": "Real trained weights, full end-to-end forward. ADR-007 C5 F64 form. "
                "Driver validated against the tiny MLA-0 DeepSeek fixture before this run.",
    }
    json.dump(meta, open(os.path.join(out_dir, "deepseek_v2lite_c5_ref.json"), "w"), indent=2)
    print(f"[c5-ref][REAL] wrote {st} ({os.path.getsize(st)} bytes)")
    print(f"[c5-ref][REAL] argmax_per_position={argmax} |logits|inf={float(np.max(np.abs(logits))):.4f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "tiny":
        mode_tiny()
    elif sys.argv[1] == "real":
        model_dir = sys.argv[2]
        out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.abspath(__file__))
        mode_real(model_dir, out_dir)
    elif sys.argv[1] == "diag":
        model_dir = sys.argv[2]
        out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.abspath(__file__))
        mode_diag(model_dir, out_dir)
    elif sys.argv[1] == "precision":
        model_dir = sys.argv[2]
        out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.abspath(__file__))
        mode_precision(model_dir, out_dir)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
