# DeepSeek-V3 Gap Audit — DEEPSEEK-V3-GAP-AUDIT (analysis only)

> **Update (MOE-V3-ROUTE-1).** The **modern-router code gap is now closed at L0**: the
> DeepSeek-V3-like routing mechanism (sigmoid scoring + `e_score_correction_bias`
> selection + group-limited top-k + `routed_scaling_factor`) is implemented in
> `src/moe/v3_router.rs` and **certified L0 (mechanism)** vs a HuggingFace `DeepseekV3MoE`
> float64 reference (router set-equality 6/6, MoE-block `max_abs_diff` 3.891e-8 < 1e-3,
> deterministic; `tests/moe_v3_route_scale_cert_test.rs`, `docs/HANDOFF_MOE_V3_ROUTE_1.md`).
> This is **L0 mechanism only — not real-weight, not L1/L2/L3, not dense ADR-004 CERTIFIED**.
> Still pending for a real V3 forward: **Q-LoRA q-path, FP8-in-MoE, MTP, V3.2 DSA**, and the
> binding constraint below — **real V3 weights remain provisioning-blocked** (no small
> V3-family checkpoint). The "router math absent" rows below are superseded for the router.

**Audit + planning only — no code, no manifests, no downloads, no certification,
no commits beyond this doc.** Evidence-grounded in the current `src/moe/` code, the
committed docs, and the public DeepSeek-V2/V3/V3.2 + Kimi-K2 configs. Every claim
cites code or a public spec; nothing is assumed. Terminology is strict:
`MoE-certified Ln`, never the dense ADR-004 `CERTIFIED`; **L4 reserved/unreachable**.

## TL;DR

The **architecture gaps** between the certified DeepSeek-V2-Lite (L3) and
DeepSeek-V3/V3.2/Kimi-K2 are **moderate and well-scoped** (Q-LoRA q-path, the modern
sigmoid + aux-loss-free + group-limited router, `routed_scaling_factor`, FP8-in-MoE).
**But the binding constraint is not code — it is model scale/provisioning:** there is
**no small V3-family checkpoint** (V3 = 671 B, Kimi-K2 = ~1 T; no "V3-Lite"), so a
**real-weight** V3 certification (L1–L3) is **hardware-blocked** on a 32 GB-RAM /
~725 GB-NVMe laptop, independent of the code. **Recommendation: do Mixtral-8x7B
real-weight first** (feasible, low-risk, ~90 % tooling reuse → completes the
Qwen/DeepSeek/Mixtral L3 triad); treat DeepSeek-V3 as a **mechanism-only L0** target
(topology + router/Q-LoRA on a tiny random-weight fixture) until a smaller V3-family
model or larger hardware exists.

---

## FASE 1 — DeepSeek support inventory today (evidence)

| Model | Implemented | Validated | Certified | Notes (evidence) |
|---|---|---|---|---|
| **DeepSeek-V2-Lite** (15.7 B, `q_lora_rank=null`) | ✅ full MLA + YaRN + dense-first + MoE | ✅ real e2e (MLA-2 disk tier) | ✅ **MoE-certified L3** | C1 1664 experts + C2 + C4 + C5 active-path `2.587e-5` (`numcert/deepseek-v2-lite.moecert.json`, MLA-1/2/3) |
| **DeepSeek-V2** (236 B, `q_lora_rank=1536`) | ⚠️ MLA core reusable; **Q-LoRA q-path absent** | ❌ | ❌ | classified `DeepSeekMoe`, but `build_deepseek` loads `self_attn.q_proj.weight` (`runtime.rs:1438`); V2 ships `q_a_proj`/`q_b_proj` → fail-loud |
| **DeepSeek-V3 / V3.1** (671 B, FP8) | ❌ router + Q-LoRA + FP8-MoE gaps | ❌ | ❌ | detected as `DeepSeekMoe` (`family.rs:165`, `kv_a_proj_with_mqa`); `production.rs:114` flags `q_a_proj` as an **unsupported variant** → fail-loud |
| **DeepSeek-V3.2-Exp** (+ sparse attention / DSA) | ❌ + new sparse-attention mechanism | ❌ | ❌ | everything in V3 plus DeepSeek Sparse Attention (lightning indexer + top-k token select) — a new attention block |
| **Kimi-K2** (~1 T, V3 recipe) | ❌ same V3 gaps at larger scale | ❌ | ❌ | `DeepseekV3ForCausalLM` arch; MLA + Q-LoRA + sigmoid routing + FP8 |

**What is genuinely present (reusable):** MLA attention (low-rank KV, decoupled
interleaved RoPE, asymmetric head dims) — `mla.rs`; **YaRN with the correct general
`attention_scaling = get_mscale(factor,mscale)/get_mscale(factor,mscale_all_dim)`**
(MLA-3) — handles V3's `mscale==mscale_all_dim==1.0` (→ 1.0) too; **dense-first**
(`first_k_dense_replace`); the **MoE block** (`RealMoeLayer` + residency +
**disk-tier**, MLA-2); the **C5 methodology** (one-layer-at-a-time F64 reference +
disk-tier real forward); **norm_topk_prob renorm** convention. **FP8 weight decode
exists in the dense/AMG path** (`graph.rs` `DType::FP8`, `amm/fp8_manager`) but is
**not wired into the MoE/MLA source reader** (`build_deepseek` uses `source.get_f32`
= bf16→f32).

**What is absent (router math):** the router is **softmax-top-k only**
(`dense.rs::route` → `softmax(W·x)`); no sigmoid scoring, no aux-loss-free bias, no
group-limited routing, no `routed_scaling_factor` applied in the combine.

---

## FASE 2 — Architectural gaps V2-Lite (L3) → V3/V3.2 (public configs)

Public `DeepSeek-V3` config: `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope=128`,
`qk_rope=64`, `v=128`, `n_routed_experts=256`, `num_experts_per_tok=8`,
`n_shared_experts=1`, `first_k_dense_replace=3`, `topk_method="noaux_tc"`,
`scoring_func="sigmoid"`, `n_group=8`, `topk_group=4`, `routed_scaling_factor=2.5`,
`norm_topk_prob=true`, per-layer `e_score_correction_bias`, `num_nextn_predict_layers=1`
(MTP), `rope_scaling=yarn (mscale=mscale_all_dim=1.0)`, FP8 `weight_block_size=[128,128]`.

| Gap | V2-Lite | V3/V3.2 | In Atenia? | Evidence |
|---|---|---|---|---|
| **Q-LoRA query path** (`q_a_proj`→`q_a_layernorm`→`q_b_proj`) | `q_lora_rank=null` (direct `q_proj`) | `q_lora_rank=1536` | ❌ **absent** | `mla.rs::project_token` does `matvec(w_q,…)`; `build_deepseek` loads `q_proj` only; `production.rs:114` flags `q_a_proj` unsupported |
| **Sigmoid scoring** (`scoring_func="sigmoid"`) | softmax | sigmoid per-expert | ❌ absent | `dense.rs::route` = `softmax` only |
| **Aux-loss-free routing** (`e_score_correction_bias` added for *selection*, not for the combine weight) | none | per-layer bias tensor | ❌ absent | no `e_score_correction` anywhere in `src/` |
| **Group-limited routing** (`n_group`, `topk_group`, `noaux_tc`) | none | 8 groups, top-4 | ❌ absent | no group logic in `sparse.rs`/`layer.rs` |
| **`routed_scaling_factor`** (selected weights × factor) | 1.0 (no-op) | 2.5 | ❌ not applied | combine = Σ wᵢ·expertᵢ; no scale (irrelevant at 1.0) |
| **YaRN `attention_scaling`** | =1.0 | =1.0 (`mscale==mscale_all_dim`) | ✅ **handled** (MLA-3 general formula) | `mla.rs::attention_scaling` |
| **MLA core / decoupled RoPE / dense-first / norm_topk renorm** | ✅ | ✅ (same) | ✅ **reused** | `mla.rs`, MLA-2 disk tier |
| **FP8 block-scaled weights** in the MoE path | bf16 | FP8 (`[128,128]`) | ⚠️ dense path only | `graph.rs` FP8 exists; MoE reader is bf16→f32 |
| **MTP** (`num_nextn_predict_layers`) | none | 1 extra head | ❌ absent (**not needed for main-path cert**) | auxiliary prediction head |
| **V3.2 sparse attention (DSA)** | none | lightning indexer + token top-k | ❌ absent (V3.2 only) | new attention mechanism |
| **Scale** (experts × size) | 64 exp / 15.7 B | 256 exp / 671 B (Kimi ~1 T) | ⚠️ disk-tier scales RAM, **not provisioning** | `models/` would need ~671 GB FP8 |

---

## FASE 3 — Complexity per gap + MLA-3 reuse

| Gap | Complexity | MLA-3 / V2-Lite reuse |
|---|---|---|
| Q-LoRA q-path | **Small–Moderate** | High — mirrors the existing `kv_a`/`kv_b` low-rank pattern (add `q_a_proj`→RMSNorm→`q_b_proj` before the nope/rope split); config + loader add `q_lora_rank` |
| Sigmoid scoring | **Small** | High — swap `softmax` for per-expert `sigmoid` in the router (the selection/combine plumbing stays) |
| Aux-loss-free bias + group-limited (`noaux_tc`) | **Moderate** | Medium — a new selection routine (add `e_score_correction_bias` to the *score used for top-k only*, group-mask to `topk_group` of `n_group`, then take the **original** sigmoid scores of the selected, normalise, × `routed_scaling_factor`); HF `DeepseekV3` has the reference algorithm |
| `routed_scaling_factor` | **Trivial** | High — one multiply in the combine |
| FP8-in-MoE intake | **Moderate** | Medium — reuse the dense FP8 decode (`fp8_manager`) in the MoE source reader, or pre-convert FP8→bf16 offline |
| MTP | **Small (skippable)** | n/a — the main LM forward ignores the MTP head; not part of C1–C5 |
| V3.2 DSA sparse attention | **Very complex** | Low — a new attention algorithm + index kernels (separate track) |
| Scale / provisioning | **Very complex (hardware)** | n/a — 671 GB–1 TB download + storage; the disk-tier helps RAM, not disk/download |

**Net code effort for V3 *main-path math*: Moderate** (Q-LoRA + the V3 router +
`routed_scaling_factor` + FP8-MoE) — the MLA/YaRN/dense-first/MoE/disk-tier/C5
machinery is **reused**. V3.2 (DSA) and MTP are separate, optional.

---

## FASE 4 — Certifiability (what each gap affects)

- **Affects correctness (⇒ blocks any ADR-007 Ln until done):** Q-LoRA q-path,
  sigmoid scoring, aux-loss-free + group-limited selection, `routed_scaling_factor`,
  FP8-MoE decode. These change the **active path** / the **selected experts** — C1
  (per-expert) is unaffected (experts are plain SwiGLU), but **C2 (router set
  equality)** and **C5 (active path)** would fail until the V3 router is implemented.
- **Affects only memory:** model scale (experts count/size) → disk-tier extends RAM;
  the F64 one-layer-at-a-time reference is larger but still one layer at a time.
- **Affects only performance:** latent/absorb KV cache (perf, not correctness).
- **Affects provisioning (the real blocker):** 671 GB (V3) / ~1 TB (Kimi) FP8
  download + storage; no small V3-family model exists.
- **Not needed for main-path cert:** MTP head (auxiliary).
- **Separate mechanism:** V3.2 DSA sparse attention (correctness + new kernels).

---

## FASE 5 — Minimum roadmap to DeepSeek-V3 L1/L2/L3 (no implementation)

Two independent axes: **(I) code** (moderate, doable on this laptop) and **(II)
real weights** (hardware-blocked). A real-weight Ln needs both.

- **V3-MECH (L0, feasible now):** tiny **random-weight** V3-topology fixture (like
  `deepseek_scale`) exercising Q-LoRA + the V3 sigmoid/aux-loss-free/group router +
  `routed_scaling_factor`, validated vs HF `DeepseekV3` f64 end-to-end at reduced
  dim → **topology/mechanism L0** + a unit-level router-parity check. *No real
  weights, no download.* This is the cheap, high-value first step and de-risks all
  the code gaps.
- **V3-PROVISION (precondition for L1+):** obtain a real V3-family checkpoint that
  fits the host. **Today: none exists small enough** → this step is **blocked**
  (671 GB–1 TB). Revisit if a "V3-Lite", a distill with native V3 router, or larger
  hardware appears.
- **V3 L1** (C1 per-expert + C2 router-set + C3 attention, real weights): needs
  V3-MECH **and** V3-PROVISION. Reuses the MLA-1 C1/C2 decomposition harness.
- **V3 L2** (+ C4 topology): the V3-MECH fixture already supplies the topology cert.
- **V3 L3** (+ C5 active path): the C5 disk-tier methodology reused; gated entirely
  by V3-PROVISION (storage/RAM for a 671 B+ model) and a V3 reference driver.
- **L4** remains reserved/unreachable (global F64).

**Conclusion:** V3 **L0 (mechanism)** is reachable now; **L1–L3 are
provisioning-blocked**, not code-blocked.

---

## FASE 6 — Priority: (A) Mixtral real-weight vs (B) open DeepSeek-V3

| Axis | (A) Mixtral-8x7B real-weight → L1–L3 | (B) DeepSeek-V3 route |
|---|---|---|
| **Strategic impact** | Completes the **3-family L3 triad** (Qwen-MoE, DeepSeek-V2-Lite, Mixtral) — the classic + MLA pillars fully real-weight certified | V3 is the frontier model; high prestige — but the cert payoff is blocked |
| **Reuse** | ~90 %: softmax-top-k (already certified), classic experts, disk-tier (graph path, used by Qwen C5); only a **Mixtral C5 reference driver** (no shared expert, no qkv bias, `w1/w3/w2`) is new (already scoped in `MIXTRAL_L3_FEASIBILITY.md` / `MIXTRAL_CERT_ROADMAP.md`) | High for code (MLA/YaRN/dense-first/MoE/C5), but a **new V3 router + Q-LoRA + FP8-MoE** are required |
| **Effort** | **Low–moderate**: ~93 GB download (feasible on 725 GB NVMe) + a reference driver; no router/attention code change | **Code: moderate; real-weight cert: infeasible** (671 GB+ download/storage) |
| **Risk** | **Low**: the routing/expert math is already certified on Qwen/Mixtral topology | **High**: new router math + a model that cannot be provisioned here |
| **Value for Atenia** | A real, auditable **third MoE family at L3** now | V3 **L0 mechanism** is valuable + cheap; **L1–L3 cannot land** on this hardware |

**Recommendation — do (A) Mixtral first.** It is the only path that yields a **new
real-weight L3** on this hardware, at low risk and high reuse, and it closes the
classic-MoE story. Then, optionally, a **(B′) "DeepSeek-V3 L0 mechanism" milestone**
(Q-LoRA + V3 router on a tiny random-weight fixture, no download) to de-risk and
document the V3 architecture cheaply — but **full V3 L1–L3 should not be scheduled
until a smaller V3-family checkpoint or larger hardware is available** (it is
provisioning-blocked, not code-blocked). V3.2 (DSA) and Kimi are further out.

---

## FASE 7 — Validation (evidence basis)

- **Code:** `src/moe/mla.rs` (`project_token` direct `q_proj`; `attention_scaling`),
  `src/moe/runtime.rs::build_deepseek` (loads `q_proj`, parses `kv_lora_rank` not
  `q_lora_rank`), `src/moe/dense.rs::route` (`softmax` only), `src/moe/family.rs:165`
  (DeepSeek detection), `src/moe/production.rs:114` (`q_a_proj` unsupported-variant),
  `src/amg/graph.rs` + `src/amm/fp8_manager.rs` (FP8 in the dense path).
- **Docs:** `numcert/deepseek-v2-lite.moecert.json` (L3), `HANDOFF_MLA_3.md`,
  `MLA_COVERAGE_AUDIT.md`, `MIXTRAL_L3_FEASIBILITY.md`, `MIXTRAL_CERT_ROADMAP.md`,
  `FAMILY_COVERAGE_AUDIT.md` (DeepSeek-V3 "very high / excluded").
- **Public specs:** DeepSeek-V2/V3 configs (`q_lora_rank`, `topk_method=noaux_tc`,
  `scoring_func=sigmoid`, `n_group`/`topk_group`, `routed_scaling_factor`,
  `e_score_correction_bias`, `num_nextn_predict_layers`, FP8 `weight_block_size`);
  DeepSeek-V3.2-Exp (DSA); Kimi-K2 (`DeepseekV3ForCausalLM`, ~1 T).

No code, tests, manifests, or models were touched.

## Risks / caveats

- Public V3 config values are quoted from the released `deepseek-ai/DeepSeek-V3`
  config; a future point release could adjust router knobs.
- "Moderate" code effort assumes the V3 router matches the HF `DeepseekV3` reference
  exactly; the aux-loss-free `noaux_tc` selection has subtle bias/normalisation order
  that must be matched bit-for-bit (the same class of bug as the MLA-3 mscale).
- Real-weight V3/Kimi certification is **hardware-blocked here**; only a mechanism
  (L0) cert is honest to schedule on this notebook.
