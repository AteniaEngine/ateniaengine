#!/usr/bin/env python3
"""MLA-1 / C5-DIAG — decisive root-cause probe for the YaRN mscale application.

Hypothesis: Atenia's `attend` multiplies the WHOLE q.k dot (nope + rope) by the
YaRN softmax scale `base * mscale^2`, whereas HF applies `mscale` only to the
decoupled-RoPE part (q_pe/k_pe via `attention_scaling` on cos/sin), leaving the
`nope` part unscaled. So Atenia over-scales `q_nope . k_nope` by `mscale^2`.

This probe replicates Atenia's layer-0 MLA attention in float64 TWO ways and
compares each to HF's committed `post_attn[0]` (residual + attn) from
`deepseek_v2lite_c5_diag.safetensors`:
  (a) ATENIA-CURRENT : score = base * mscale^2 * (nope_dot + rope_dot)
  (b) HF-STYLE FIX   : score = base * (nope_dot + mscale^2 * rope_dot)

If (b) ~ HF (tiny diff) and (a) ~ the observed isolation 0.055, the root cause is
confirmed: mscale^2 must apply ONLY to the decoupled-RoPE part. One layer only —
no full model load. Diagnostic; changes nothing.

  python fixtures/moe/diag_mscale_probe.py models/DeepSeek-V2-Lite
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open


def shard_handles(model_dir):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(name):
        shard = idx[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(model_dir, shard), framework="pt")
        return handles[shard].get_tensor(name).to(torch.float64).numpy()

    return get


def yarn_get_mscale(scale, mscale):
    return 0.1 * mscale * np.log(scale) + 1.0 if scale > 1.0 else 1.0


def yarn_find_correction_dim(num_rot, dim, base, max_pos):
    return (dim * np.log(max_pos / (num_rot * 2 * np.pi))) / (2 * np.log(base))


def rope_inv_freqs(rope_dim, base, factor, orig_max, beta_fast, beta_slow):
    half = rope_dim // 2
    extra = np.array([base ** (-(2.0 * i) / rope_dim) for i in range(half)])
    low = np.floor(yarn_find_correction_dim(beta_fast, rope_dim, base, orig_max))
    low = max(low, 0.0)
    high = np.ceil(yarn_find_correction_dim(beta_slow, rope_dim, base, orig_max))
    high = min(high, rope_dim - 1)
    denom = 0.001 if abs(high - low) < 1e-12 else (high - low)
    out = np.empty(half)
    for i in range(half):
        inter = extra[i] / factor
        ramp = np.clip((i - low) / denom, 0.0, 1.0)
        mask = 1.0 - ramp
        out[i] = inter * (1.0 - mask) + extra[i] * mask
    return out


def rope_interleaved(x, pos, inv_freqs):
    out = x.copy()
    for i, f in enumerate(inv_freqs):
        ang = pos * f
        c, s = np.cos(ang), np.sin(ang)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i] = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out


def rmsnorm(x, g, eps):
    ms = np.mean(x * x)
    return (x / np.sqrt(ms + eps)) * g


def main():
    model_dir = sys.argv[1]
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    H = cfg["hidden_size"]
    nh = cfg["num_attention_heads"]
    nope = cfg["qk_nope_head_dim"]
    rope = cfg["qk_rope_head_dim"]
    vd = cfg["v_head_dim"]
    kvl = cfg["kv_lora_rank"]
    qkh = nope + rope
    eps = cfg.get("rms_norm_eps", 1e-6)
    base = cfg.get("rope_theta", 10000.0)
    rs = cfg["rope_scaling"]
    factor = rs["factor"]
    orig = rs["original_max_position_embeddings"]
    mscale_all = rs.get("mscale_all_dim", rs.get("mscale", 1.0))
    mscale = yarn_get_mscale(factor, mscale_all)
    inv_freqs = rope_inv_freqs(rope, base, factor, orig, rs["beta_fast"], rs["beta_slow"])
    print(f"mscale={mscale:.6f} mscale^2={mscale**2:.6f} (factor={factor}, mscale_all_dim={mscale_all})")

    get = shard_handles(model_dir)
    p = "model.layers.0"
    w_q = get(f"{p}.self_attn.q_proj.weight")               # [nh*qkh, H]
    w_kv_a = get(f"{p}.self_attn.kv_a_proj_with_mqa.weight")  # [kvl+rope, H]
    kv_a_ln = get(f"{p}.self_attn.kv_a_layernorm.weight")     # [kvl]
    w_kv_b = get(f"{p}.self_attn.kv_b_proj.weight")           # [nh*(nope+vd), kvl]
    w_o = get(f"{p}.self_attn.o_proj.weight")                # [H, nh*vd]
    in_ln = get(f"{p}.input_layernorm.weight")               # [H]

    # input to layer 0 = embeddings (from the diag fixture), 4 tokens.
    here = os.path.dirname(os.path.abspath(__file__))
    f = safe_open(os.path.join(here, "deepseek_v2lite_c5_diag.safetensors"), framework="np")
    emb = f.get_tensor("embeddings").astype(np.float64)      # [seq, H]
    ref_post_attn = f.get_tensor("post_attn").astype(np.float64)[0]  # layer 0 [seq, H]
    seq = emb.shape[0]

    # Project all tokens (per-head q,k,v) with Atenia's exact math.
    qh = np.zeros((seq, nh, qkh)); kh = np.zeros((seq, nh, qkh)); vh = np.zeros((seq, nh, vd))
    for t in range(seq):
        h = rmsnorm(emb[t], in_ln, eps)
        q = w_q @ h                       # [nh*qkh]
        ckv = w_kv_a @ h                  # [kvl+rope]
        compressed = ckv[:kvl]
        k_pe = ckv[kvl:kvl + rope]
        cn = rmsnorm(compressed, kv_a_ln, eps)
        kv = w_kv_b @ cn                  # [nh*(nope+vd)]
        k_pe_r = rope_interleaved(k_pe, t, inv_freqs)
        for hd in range(nh):
            qslice = q[hd * qkh:(hd + 1) * qkh]
            q_pe = rope_interleaved(qslice[nope:qkh], t, inv_freqs)
            qh[t, hd, :nope] = qslice[:nope]
            qh[t, hd, nope:] = q_pe
            kh[t, hd, :nope] = kv[hd * (nope + vd):hd * (nope + vd) + nope]
            kh[t, hd, nope:] = k_pe_r
            vh[t, hd] = kv[hd * (nope + vd) + nope:hd * (nope + vd) + nope + vd]

    # --- Compare Atenia's YaRN inv_freq + mscale against HF's OWN rotary ---
    from transformers import DeepseekV2Config
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2RotaryEmbedding
    hfcfg = DeepseekV2Config(**cfg)
    rot = DeepseekV2RotaryEmbedding(hfcfg)
    hf_inv = rot.inv_freq.double().numpy()
    hf_scaling = float(rot.attention_scaling)
    print(f"HF attention_scaling={hf_scaling:.6f}  vs  Atenia mscale={mscale:.6f}")
    print(f"inv_freq  max_abs_diff(HF, Atenia)={np.abs(hf_inv - inv_freqs).max():.4e}  "
          f"len HF={len(hf_inv)} Atenia={len(inv_freqs)}")
    print(f"  HF inv_freq[:4]    ={hf_inv[:4]}")
    print(f"  Atenia inv_freq[:4]={inv_freqs[:4]}")
    print(f"  HF inv_freq[-4:]   ={hf_inv[-4:]}")
    print(f"  Atenia inv_freq[-4:]={inv_freqs[-4:]}")

    base_scale = 1.0 / np.sqrt(qkh)

    def attend(scale_mode):
        out = np.zeros((seq, nh * vd))
        for t in range(seq):
            for hd in range(nh):
                sc = np.zeros(t + 1)
                for s in range(t + 1):
                    nope_dot = np.dot(qh[t, hd, :nope], kh[s, hd, :nope])
                    rope_dot = np.dot(qh[t, hd, nope:], kh[s, hd, nope:])
                    asc = hf_scaling  # HF attention_scaling (cos/sin), here 1.0
                    if scale_mode == "atenia":        # (a) Atenia: mscale^2 on WHOLE
                        sc[s] = (nope_dot + rope_dot) * base_scale * mscale * mscale
                    elif scale_mode == "hf_fix":      # (b) mscale^2 on ROPE only
                        sc[s] = (nope_dot + mscale * mscale * rope_dot) * base_scale
                    else:                              # (c) HF-EXACT: attention_scaling^2 on rope
                        sc[s] = (nope_dot + asc * asc * rope_dot) * base_scale
                sc = np.exp(sc - sc.max()); sc /= sc.sum()
                for hd_i in range(vd):
                    out[t, hd * vd + hd_i] = np.dot(sc, vh[:t + 1, hd, hd_i])
        return out

    for mode in ("atenia", "hf_fix", "hf_exact"):
        ctx = attend(mode)
        x1 = emb + (ctx @ w_o.T)  # residual + o_proj
        d = np.abs(x1 - ref_post_attn).max()
        print(f"  mode={mode:8s}  post_attn(x1) max_abs_diff vs HF = {d:.4e}")


if __name__ == "__main__":
    main()
