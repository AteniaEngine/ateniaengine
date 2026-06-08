#!/usr/bin/env python3
"""MOE-V3-ROUTE-1 — DeepSeek-V3-like routing L0 mechanism reference (OFFLINE).

Reproducibility artifact. NOT run in CI, NOT imported by Rust.

Builds a REDUCED-DIM HuggingFace `DeepseekV3MoE` block (random deterministic
weights) and dumps, in float64, the inputs + weights + the reference routing
decision and block output, so Atenia's `src/moe/v3_router.rs` mechanism can be
certified at L0 (mechanism/topology only — NOT real V3 weights, NOT L1/L2/L3,
NOT dense ADR-004 CERTIFIED).

The router algorithm certified here is HF's
`transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3MoE
.route_tokens_to_experts`:
  scores            = sigmoid(router_logits)
  scores_for_choice = scores + e_score_correction_bias        (selection only)
  group_score[g]    = sum of top-2 scores_for_choice in group g
  selected groups   = top-`topk_group` by group_score
  selected experts  = top-`top_k` scores_for_choice within selected groups
  combine weight    = scores (no bias), /=sum if norm_topk_prob, *= routed_scaling_factor

Experts are SwiGLU (`down(silu(gate·x) * (up·x))`); the shared expert is an
ungated MLP added to the routed combine. Source: transformers v5.6.2.

Writes: v3_route_ref.safetensors (weights + probes), v3_route_ref.json (config +
dense reference combine-weights + block outputs + per-token selected experts +
selection margins).
"""
import json
import os

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

OUT = os.path.dirname(os.path.abspath(__file__))

# Reduced-dim DeepSeek-V3 routing topology. Group-limiting is real: 8 experts in
# 4 groups of 2, pick 2 groups, then top-2 experts -> prunes experts outside the
# 2 selected groups.
HIDDEN = 16
INTER = 8
N_ROUTED = 8
N_GROUP = 4
TOPK_GROUP = 2
TOP_K = 2
N_SHARED = 1
SCALE = 2.5
NORM = True
TOKENS = 6


def main():
    torch.manual_seed(0x0DEE5933)
    cfg = DeepseekV3Config(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTER,
        intermediate_size=INTER,
        n_routed_experts=N_ROUTED,
        num_local_experts=N_ROUTED,
        n_shared_experts=N_SHARED,
        num_experts_per_tok=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        norm_topk_prob=NORM,
        routed_scaling_factor=SCALE,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        hidden_act="silu",
        first_k_dense_replace=0,
        num_hidden_layers=1,
        vocab_size=32,
    )
    moe = DeepseekV3MoE(cfg).double().eval()

    # Give the per-expert correction bias real (nonzero) values so bias-corrected
    # selection is genuinely exercised.
    with torch.no_grad():
        moe.gate.e_score_correction_bias.copy_(
            (torch.rand(N_ROUTED, dtype=torch.float64) - 0.5) * 0.6
        )

    rng = np.random.default_rng(0x0DEE5933)
    hidden = torch.tensor(rng.standard_normal((TOKENS, HIDDEN)), dtype=torch.float64)

    with torch.no_grad():
        router_logits = moe.gate(hidden)                       # [tokens, n_routed] (f32 inside)
        topk_idx, topk_w = moe.route_tokens_to_experts(router_logits)
        block_out = moe(hidden)                                # [tokens, hidden] f64

    topk_idx = topk_idx.cpu().numpy()
    topk_w = topk_w.double().cpu().numpy()
    # Dense [tokens, n_routed] combine weights (0 where not selected).
    dense_w = np.zeros((TOKENS, N_ROUTED), dtype=np.float64)
    for t in range(TOKENS):
        for slot in range(topk_idx.shape[1]):
            dense_w[t, int(topk_idx[t, slot])] += float(topk_w[t, slot])
    selected = [sorted(int(i) for i in topk_idx[t]) for t in range(TOKENS)]

    # Selection margin per token (lowest selected scores_for_choice − highest
    # not-selected-in-selected-groups), recomputed in f64 for reporting.
    scores = torch.sigmoid(router_logits.double()).cpu().numpy()
    bias = moe.gate.e_score_correction_bias.double().cpu().numpy()
    choice = scores + bias[None, :]
    epg = N_ROUTED // N_GROUP
    margins = []
    for t in range(TOKENS):
        gsum = [sorted(choice[t, g * epg:(g + 1) * epg], reverse=True)[:2] for g in range(N_GROUP)]
        gsum = [s[0] + s[1] for s in gsum]
        sel_groups = sorted(range(N_GROUP), key=lambda g: (-gsum[g], g))[:TOPK_GROUP]
        in_groups = [i for i in range(N_ROUTED) if (i // epg) in sel_groups]
        sel = set(selected[t])
        min_sel = min(choice[t, i] for i in sel)
        rest = [choice[t, i] for i in in_groups if i not in sel]
        margins.append(float(min_sel - max(rest)) if rest else float("inf"))

    # ---- weights -> safetensors (unpack packed experts to per-expert g/u/d) ----
    sd = {}
    gw = moe.gate.weight.detach().double().cpu().numpy().astype(np.float32)   # [n_routed, hidden]
    sd["router.weight"] = np.ascontiguousarray(gw)
    sd["router.bias"] = np.ascontiguousarray(bias.astype(np.float32))
    gate_up = moe.experts.gate_up_proj.detach().double().cpu().numpy()        # [n, 2*inter, hidden]
    down = moe.experts.down_proj.detach().double().cpu().numpy()              # [n, hidden, inter]
    for e in range(N_ROUTED):
        sd[f"expert.{e}.w_gate"] = np.ascontiguousarray(gate_up[e, :INTER, :].astype(np.float32))
        sd[f"expert.{e}.w_up"] = np.ascontiguousarray(gate_up[e, INTER:, :].astype(np.float32))
        sd[f"expert.{e}.w_down"] = np.ascontiguousarray(down[e].astype(np.float32))
    se = moe.shared_experts
    sd["shared.w_gate"] = np.ascontiguousarray(se.gate_proj.weight.detach().double().cpu().numpy().astype(np.float32))
    sd["shared.w_up"] = np.ascontiguousarray(se.up_proj.weight.detach().double().cpu().numpy().astype(np.float32))
    sd["shared.w_down"] = np.ascontiguousarray(se.down_proj.weight.detach().double().cpu().numpy().astype(np.float32))
    sd["hidden"] = np.ascontiguousarray(hidden.cpu().numpy().astype(np.float32))
    save_file(sd, os.path.join(OUT, "v3_route_ref.safetensors"))

    shared_inter = INTER * N_SHARED
    meta = dict(
        mechanism="DeepSeek-V3-like routing (sigmoid + e_score_correction_bias selection + "
                  "group-limited top-k + routed_scaling_factor), L0 mechanism only",
        source="transformers v5.6.2 DeepseekV3MoE.route_tokens_to_experts",
        hidden_size=HIDDEN, moe_intermediate_size=INTER, shared_intermediate_size=shared_inter,
        n_routed_experts=N_ROUTED, n_group=N_GROUP, topk_group=TOPK_GROUP, top_k=TOP_K,
        n_shared_experts=N_SHARED, routed_scaling_factor=SCALE, norm_topk_prob=NORM,
        scoring_func="sigmoid", tokens=TOKENS,
        dense_combine_weights=dense_w.reshape(-1).tolist(),
        block_out=block_out.double().cpu().numpy().reshape(-1).tolist(),
        selected_experts=selected,
        selection_margins=margins,
        min_selection_margin=float(min(m for m in margins if np.isfinite(m))) if any(np.isfinite(margins)) else float("inf"),
    )
    json.dump(meta, open(os.path.join(OUT, "v3_route_ref.json"), "w"))
    print("WROTE v3_route_ref: tokens", TOKENS, "selected", selected,
          "min_margin", meta["min_selection_margin"],
          "bytes", os.path.getsize(os.path.join(OUT, "v3_route_ref.safetensors")))


if __name__ == "__main__":
    main()
