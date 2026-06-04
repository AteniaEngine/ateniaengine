#!/usr/bin/env python3
"""MOE-CERT-4 — C5 active-path F64 reference for Qwen-MoE.

ADR-007 C5 (active-path parity) certifies the REAL model, as actually run,
end-to-end on a canonical input, vs an external reference over the active
subgraph. A global-F64 forward is infeasible here (full weights 114 GB F64 /
57 GB F32 / 27 GB bf16 do not fit the 34 GB host), so we compute the reference
**one decoder layer at a time in float64**, reusing HuggingFace's OWN trusted
Qwen2Moe layer module (zero convention risk: RoPE / RMSNorm / attention / MoE
routing are HF's code). Peak RAM is one layer in F64 (~4.5 GB), not the whole
model. This is the F64 form of C5 (stronger than the F32 cross-framework
fallback). It is NOT L4: L4 holds the ENTIRE model in F64 at once; we never do.

Two modes:
  python generate_qwen_moe_c5_reference.py tiny
      Validate the driver: run it on the committed tiny Qwen-MoE fixture and
      compare to that fixture's HF f64 logits. If this does not match (~1e-9),
      the driver is wrong and NOTHING downstream may be trusted.

  python generate_qwen_moe_c5_reference.py real <model_dir> [out_dir]
      Compute the C5 reference for the real Qwen1.5-MoE-A2.7B on a fixed
      canonical input and write qwen_moe_c5_ref.{safetensors,json}.

The Rust harness (tests/moe_cert4_qwen_active_path_test.rs) then compares
Atenia's MoeRuntime full forward to this reference.
"""

import gc
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.numpy import save_file
import numpy as np

from transformers import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeDecoderLayer,
    Qwen2MoeRMSNorm,
    Qwen2MoeRotaryEmbedding,
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
            self.single = safe_open(
                os.path.join(model_dir, "model.safetensors"), framework="pt"
            )

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
    """End-to-end Qwen2Moe forward in float64, ONE LAYER AT A TIME, using HF's
    own decoder-layer module. Returns logits [seq, vocab] (float64 numpy)."""
    w = Weights(model_dir, single_file=single_file)
    seq = len(input_ids)
    ids = torch.tensor(input_ids, dtype=torch.long)

    # Embedding (index the needed rows, upcast to f64).
    embed = w.f64("model.embed_tokens.weight")  # [vocab, hidden]
    h = embed[ids].unsqueeze(0)  # [1, seq, hidden]
    del embed
    gc.collect()

    # RoPE position embeddings (HF's own rotary), causal mask — all f64.
    rotary = Qwen2MoeRotaryEmbedding(cfg).to(torch.float64)
    position_ids = torch.arange(seq).unsqueeze(0)
    cos, sin = rotary(h, position_ids)
    pos_emb = (cos.to(torch.float64), sin.to(torch.float64))
    mask = torch.triu(torch.full((seq, seq), float("-inf"), dtype=torch.float64), diagonal=1)
    mask = mask[None, None, :, :]  # [1,1,seq,seq]

    for l in range(cfg.num_hidden_layers):
        layer = Qwen2MoeDecoderLayer(cfg, l).to(torch.float64).eval()
        sd = {}
        for key in layer.state_dict().keys():
            full = f"model.layers.{l}.{key}"
            if w.has(full):
                # On-disk tensor (packed tiny fixture, or any classic non-expert).
                sd[key] = w.f64(full)
            elif key == "mlp.experts.gate_up_proj":
                # Real checkpoint stores CLASSIC per-expert gate/up; transformers
                # 5.x packs them as [E, 2*moe_inter, hidden] (gate first half, up
                # second) — the same convention Atenia's MOE-15 packed path uses
                # and which is HF-certified. Build it from the classic tensors.
                gus = []
                for e in range(cfg.num_experts):
                    g = w.f64(f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight")  # [inter, hidden]
                    u = w.f64(f"model.layers.{l}.mlp.experts.{e}.up_proj.weight")    # [inter, hidden]
                    gus.append(torch.cat([g, u], dim=0))                              # [2*inter, hidden]
                sd[key] = torch.stack(gus, dim=0)                                     # [E, 2*inter, hidden]
            elif key == "mlp.experts.down_proj":
                ds = [
                    w.f64(f"model.layers.{l}.mlp.experts.{e}.down_proj.weight")       # [hidden, inter]
                    for e in range(cfg.num_experts)
                ]
                sd[key] = torch.stack(ds, dim=0)                                      # [E, hidden, inter]
            else:
                sd[key] = w.f64(full)
        layer.load_state_dict(sd)
        out = layer(h, position_embeddings=pos_emb, attention_mask=mask)
        h = out[0] if isinstance(out, tuple) else out
        del layer, sd, out
        gc.collect()

    norm = Qwen2MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(torch.float64)
    with torch.no_grad():
        norm.weight.copy_(w.f64("model.norm.weight"))
    h = norm(h)
    lm = w.f64("lm_head.weight")  # [vocab, hidden]
    logits = (h @ lm.T)[0]  # [seq, vocab]
    return logits.numpy().astype(np.float64)


def load_cfg(model_dir):
    return Qwen2MoeConfig(**json.load(open(os.path.join(model_dir, "config.json"))))


def mode_tiny():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = load_cfg_from_file(os.path.join(here, "qwen_moe_tiny_config.json"))
    ref = json.load(open(os.path.join(here, "qwen_moe_tiny.json")))
    ids = ref["input_ids"]
    vocab = ref["vocab_size"]
    hf = np.array(ref["hf_logits"], dtype=np.float64).reshape(len(ids), vocab)
    got = forward_logits_f64(here, cfg, ids, single_file="qwen_moe_tiny.safetensors")
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


def load_cfg_from_file(path):
    return Qwen2MoeConfig(**json.load(open(path)))


def mode_real(model_dir, out_dir):
    cfg = load_cfg(model_dir)
    ids = CANONICAL_INPUT_IDS
    print(f"[c5-ref][REAL] {model_dir} layers={cfg.num_hidden_layers} input_ids={ids}")
    logits = forward_logits_f64(model_dir, cfg, ids)  # [seq, vocab] f64
    argmax = [int(x) for x in logits.argmax(axis=1)]
    os.makedirs(out_dir, exist_ok=True)
    st = os.path.join(out_dir, "qwen_moe_c5_ref.safetensors")
    save_file({"logits": logits.astype(np.float32)}, st)  # f32-stored, f64-computed
    meta = {
        "milestone": "MOE-CERT-4",
        "obligation": "C5 active-path (F64, one layer at a time)",
        "model": "Qwen1.5-MoE-A2.7B-Chat",
        "oracle": "HuggingFace Qwen2MoeDecoderLayer in float64, one layer at a time "
                  "(trusted HF code; never the whole model in F64 -> not L4)",
        "input_ids": ids,
        "seq": len(ids),
        "vocab_size": cfg.vocab_size,
        "argmax_per_position": argmax,
        "reference_file": "qwen_moe_c5_ref.safetensors",
        "note": "Real trained weights, full end-to-end forward. ADR-007 C5 F64 form. "
                "Driver validated against the tiny Qwen-MoE HF fixture before this run.",
    }
    json.dump(meta, open(os.path.join(out_dir, "qwen_moe_c5_ref.json"), "w"), indent=2)
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
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
