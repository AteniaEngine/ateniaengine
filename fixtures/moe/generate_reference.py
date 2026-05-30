#!/usr/bin/env python3
"""MOE-16 — generate numerical-equivalence reference fixtures (OFFLINE).

This script is a *reproducibility artifact*. It is NOT run in CI and is NOT
imported by any Rust test. It reads tiny MoE checkpoints already present on a
local disk, extracts each model's **layer-0 MoE block** tensors into a small
F32 safetensors fixture, and computes two f64 references for a fixed input:

  * `atenia_ref`  — an independent f64 reimplementation of the EXACT MoE
                    operation Atenia performs (softmax -> top-k (lower-index
                    tiebreak) -> RENORMALISE selected -> SwiGLU experts ->
                    weighted sum -> + shared expert UNGATED). This is the
                    primary assertion target (ADR-004 style: f64 reference of
                    the defined operation; Rust f32 must match it).
  * `hf_ref`      — the real HuggingFace `transformers` MoE block forward in
                    f64 (`block.double()`). Informative ground truth; reveals
                    convention divergences (norm_topk_prob, sigmoid-gated
                    shared expert) that Atenia does not (yet) implement.

Everything originates from F32 weights + F32 input (what Atenia consumes),
upcast to f64 for both references, so the comparison is apples-to-apples.

Usage (offline only):
    python fixtures/moe/generate_reference.py

Edit MODELS below to point at local checkpoint directories.
"""

import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM

# name -> local checkpoint directory (machine-specific; not committed).
MODELS = {
    "qwen15_moe": r"D:\models\tiny-qwen15moe",
    "qwen2_moe": r"D:\models\tiny-qwen2moe",
    "mixtral": r"D:\models\tiny-mixtral",
    # Appended last so existing fixtures' shared RNG draws are unchanged.
    "qwen3_moe": r"D:\models\tiny-qwen3moe",
    "mixtral_titanml": r"D:\models\tiny-mixtral-titanml",
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 0x4D4F4531  # "MOE1"


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(v):
    m = v.max()
    e = np.exp(v - m)
    return e / e.sum()


def atenia_expert(x, gate, up, down):
    # gate/up: [d_ff, d_model]; down: [d_model, d_ff]; all f64.
    h = silu(gate @ x) * (up @ x)
    return down @ h


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def atenia_moe_block(x, router_w, experts, top_k, shared=None,
                     renormalize=True, shared_gate=None):
    """f64 MoE block. `renormalize`/`shared_gate` select the convention:
    - Atenia default: renormalize=True, shared_gate=None (ungated shared).
    - HF Qwen:        renormalize=False, shared_gate=[d_model] (sigmoid-gated).
    """
    logits = router_w @ x                      # [ne]
    w = softmax(logits)                        # over all experts
    # top-k by weight desc, tie -> lower index (matches Atenia).
    order = sorted(range(len(w)), key=lambda i: (-w[i], i))
    idx = sorted(order[:top_k])
    sel = np.array([w[i] for i in idx], dtype=np.float64)
    if renormalize:
        sel = sel / sel.sum()
    out = np.zeros_like(x)
    for j, e in enumerate(idx):
        g, u, d = experts[e]
        out = out + sel[j] * atenia_expert(x, g, u, d)
    if shared is not None:
        sg, su, sd = shared
        s = atenia_expert(x, sg, su, sd)
        if shared_gate is not None:
            s = sigmoid(float(shared_gate @ x)) * s
        out = out + s
    return out


def to_f32_f64(t):
    """bf16/f32 tensor -> f32 numpy (fixture) and f64 numpy (reference)."""
    a32 = t.detach().float().numpy().astype(np.float32)
    return a32, a32.astype(np.float64)


def extract_layer0(model, sd, name_prefix):
    """Return (fixture_tensors_f32, experts_f64, router_f64, shared_f64, dims)
    for layer 0, handling classic per-expert and packed/fused layouts."""
    fixture = {}
    keys = [k for k in sd.keys() if k.startswith(name_prefix)]

    router_name = f"{name_prefix}.gate.weight"
    router_t = sd[router_name]
    r32, r64 = to_f32_f64(router_t)
    fixture[router_name] = r32
    ne, d_model = r64.shape

    experts64 = []
    packed = any(".experts.gate_up_proj" in k for k in keys)
    if packed:
        gu = sd[f"{name_prefix}.experts.gate_up_proj"]
        dn = sd[f"{name_prefix}.experts.down_proj"]
        gu32, gu64 = to_f32_f64(gu)
        dn32, dn64 = to_f32_f64(dn)
        fixture[f"{name_prefix}.experts.gate_up_proj"] = gu32
        fixture[f"{name_prefix}.experts.down_proj"] = dn32
        # gu: [ne, 2*d_ff, d_model]; first d_ff rows = gate, next = up.
        two_dff = gu64.shape[1]
        d_ff = two_dff // 2
        for e in range(ne):
            g = gu64[e, :d_ff, :]
            u = gu64[e, d_ff:2 * d_ff, :]
            d = dn64[e]
            experts64.append((g, u, d))
    else:
        # classic per-expert. Two name schemes:
        #  - Qwen-MoE: gate_proj / up_proj / down_proj
        #  - Mixtral:  w1 (gate) / w3 (up) / w2 (down)
        mixtral_classic = f"{name_prefix}.experts.0.w1.weight" in sd
        if mixtral_classic:
            names_for = lambda e: (
                f"{name_prefix}.experts.{e}.w1.weight",
                f"{name_prefix}.experts.{e}.w3.weight",
                f"{name_prefix}.experts.{e}.w2.weight",
            )
        else:
            names_for = lambda e: (
                f"{name_prefix}.experts.{e}.gate_proj.weight",
                f"{name_prefix}.experts.{e}.up_proj.weight",
                f"{name_prefix}.experts.{e}.down_proj.weight",
            )
        gn0, _, _ = names_for(0)
        d_ff = sd[gn0].shape[0]
        for e in range(ne):
            gn, un, dn = names_for(e)
            g, u, d = sd[gn], sd[un], sd[dn]
            for nm, t in [(gn, g), (un, u), (dn, d)]:
                a32, _ = to_f32_f64(t)
                fixture[nm] = a32
            _, g64 = to_f32_f64(g)
            _, u64 = to_f32_f64(u)
            _, d64 = to_f32_f64(d)
            experts64.append((g64, u64, d64))

    shared64 = None
    if any("shared_expert.gate_proj" in k for k in keys):
        sg = sd[f"{name_prefix}.shared_expert.gate_proj.weight"]
        su = sd[f"{name_prefix}.shared_expert.up_proj.weight"]
        sd_ = sd[f"{name_prefix}.shared_expert.down_proj.weight"]
        for nm, t in [
            (f"{name_prefix}.shared_expert.gate_proj.weight", sg),
            (f"{name_prefix}.shared_expert.up_proj.weight", su),
            (f"{name_prefix}.shared_expert.down_proj.weight", sd_),
        ]:
            a32, _ = to_f32_f64(t)
            fixture[nm] = a32
        _, sg64 = to_f32_f64(sg)
        _, su64 = to_f32_f64(su)
        _, sd64 = to_f32_f64(sd_)
        shared64 = (sg64, su64, sd64)

    # Shared-expert sigmoid gate (Qwen-MoE): [1, d_model]. Needed for the
    # MOE-17 HF-convention forward.
    shared_gate64 = None
    gate_name = f"{name_prefix}.shared_expert_gate.weight"
    if gate_name in sd:
        a32, g64 = to_f32_f64(sd[gate_name])
        fixture[gate_name] = a32
        shared_gate64 = g64.reshape(-1)  # [d_model]

    dims = dict(num_experts=int(ne), d_model=int(d_model), d_ff=int(d_ff),
                packed=bool(packed), has_shared=shared64 is not None,
                has_shared_gate=shared_gate64 is not None)
    return fixture, experts64, r64, shared64, shared_gate64, dims


def metrics(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = np.abs(a - b)
    return dict(
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        rmse=float(np.sqrt(((a - b) ** 2).mean())),
        argmax_match=bool(int(a.argmax()) == int(b.argmax())),
    )


def main():
    rng = np.random.default_rng(SEED)
    for name, path in MODELS.items():
        if not os.path.isdir(path):
            print(f"SKIP {name}: dir not found {path}")
            continue
        print(f"== {name} ==")
        cfg_path = os.path.join(path, "config.json")
        cfg = json.load(open(cfg_path))
        top_k = cfg.get("num_experts_per_tok", cfg.get("num_experts_per_token", 2))
        norm_topk = bool(cfg.get("norm_topk_prob", False))

        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        model.eval()
        sd = model.state_dict()
        prefix = "model.layers.0.mlp"
        # Some Mixtral variants name the block block_sparse_moe.
        if not any(k.startswith(prefix + ".") for k in sd.keys()):
            prefix = "model.layers.0.block_sparse_moe"

        fixture, experts64, router64, shared64, shared_gate64, dims = extract_layer0(model, sd, prefix)
        d_model = dims["d_model"]

        # Qwen3-MoE stores the router on disk as `mlp.router.weight`; HF's
        # state_dict renames it to `mlp.gate.weight`. Write the fixture under
        # the real on-disk name so the Rust test exercises Qwen3-MoE router
        # naming (and the MoE router-detection fix).
        if name == "qwen3_moe":
            gk = f"{prefix}.gate.weight"
            if gk in fixture:
                fixture[f"{prefix}.router.weight"] = fixture.pop(gk)

        x32 = (rng.standard_normal(d_model) * 0.5).astype(np.float32)
        x64 = x32.astype(np.float64)

        # Atenia default convention (renormalize + ungated shared).
        atenia_ref = atenia_moe_block(x64, router64, experts64, top_k, shared64)
        # Atenia under HF convention (no renorm + sigmoid-gated shared) — must
        # match hf_ref, validating our understanding of the HF block.
        atenia_hf_ref = atenia_moe_block(
            x64, router64, experts64, top_k, shared64,
            renormalize=norm_topk, shared_gate=shared_gate64,
        )

        # HF reference: real transformers block forward in f64.
        block = dict(model.named_modules())[prefix].double()
        with torch.no_grad():
            xt = torch.tensor(x64, dtype=torch.float64).reshape(1, 1, d_model)
            out = block(xt)
            hf_t = out[0] if isinstance(out, tuple) else out
            hf_ref = hf_t.reshape(-1).detach().numpy().astype(np.float64)

        # Save fixture safetensors (F32) + JSON sidecar.
        st_path = os.path.join(OUT_DIR, f"{name}_layer0.safetensors")
        save_file({k: np.ascontiguousarray(v) for k, v in fixture.items()}, st_path)

        sidecar = dict(
            model=name,
            source_repo={
                "qwen15_moe": "katuni4ka/tiny-random-qwen1.5-moe",
                "qwen2_moe": "hf-internal-testing/tiny-random-Qwen2MoeForCausalLM",
                "qwen3_moe": "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM",
                "mixtral": "hf-internal-testing/tiny-random-MixtralForCausalLM",
                "mixtral_titanml": "TitanML/tiny-mixtral",
            }[name],
            tensor_prefix=prefix,
            num_experts=dims["num_experts"],
            experts_per_token=int(top_k),
            d_model=dims["d_model"],
            d_ff=dims["d_ff"],
            packed=dims["packed"],
            has_shared=dims["has_shared"],
            has_shared_gate=dims["has_shared_gate"],
            norm_topk_prob=norm_topk,
            input=x32.tolist(),
            atenia_ref=atenia_ref.tolist(),
            atenia_hf_ref=atenia_hf_ref.tolist(),
            hf_ref=hf_ref.tolist(),
            metrics_atenia_ref_vs_hf=metrics(atenia_ref, hf_ref),
            metrics_atenia_hf_ref_vs_hf=metrics(atenia_hf_ref, hf_ref),
        )
        json.dump(sidecar, open(os.path.join(OUT_DIR, f"{name}_layer0.json"), "w"), indent=2)
        print("  dims:", dims, "top_k", top_k, "norm_topk", norm_topk)
        print("  atenia_ref vs hf_ref:", sidecar["metrics_atenia_ref_vs_hf"])
        print("  wrote", st_path)


if __name__ == "__main__":
    main()
