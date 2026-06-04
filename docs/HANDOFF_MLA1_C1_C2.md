# HANDOFF — MLA-1 (C1+C2): DeepSeek-V2-Lite per-expert + router certification

Milestone: **MLA-1 / C1+C2** — certify **C1 (per-expert)** and **C2 (router
parity)** on the **real DeepSeek-V2-Lite** weights, reusing the Qwen MOE-CERT-2-ext
decomposition tooling. **No new features, no MLA/YaRN/runtime/loader/numerics/ATK
change** — test-only harness + a reference generator. Predecessors: MLA-0 (YaRN +
dense-first + routing convention), DeepSeek-V2-Lite provisioned (~31.4 GB local).

## What was done

The expert/router tensor names of DeepSeek-V2-Lite are **identical** to Qwen-MoE
(`mlp.gate.weight`, `mlp.experts.{e}.{gate,up,down}_proj`), so the Qwen
decomposition harness transfers almost verbatim. The only DeepSeek specifics:
**dense-first layer 0 is skipped** (`first_k_dense_replace=1`, no experts there),
**64 experts / top-6**, **`norm_topk_prob=false`** (the top-k *set* is convention-
independent), and **3 of the 26 MoE layers span two shards**.

- **Reference generator** (`fixtures/moe/generate_deepseek_v2lite_decomposition_reference.py`):
  a config-driven port of the Qwen generator — reads the real bf16 weights ->
  float64, **one expert at a time, one MoE layer at a time** (never the full model
  in F64), skipping dense layers; emits `deepseek_v2lite_decomp_ref.{safetensors,
  json}` (~13.6 MB: `input[2048]`, `router_logits[26,64]`, `expert_outputs[26,64,
  2048]` f32-stored, + per-layer topk + routing margin).
- **Harness** (`tests/moe_mla1_deepseek_decomposition_test.rs`,
  `mla1_real_deepseek_v2lite_c1_c2`, `#[ignore]` + env `DEEPSEEK_V2_LITE_DIR`):
  per MoE layer, opens the shard(s) holding it, `RealMoeLayer::assemble`, then
  **C1** per-expert `forward` vs the f64 reference (exhaustive) and **C2**
  `top_k_routing` set equality vs the reference + routing margin.

## Results (real, measured — 1664 experts, 26 MoE layers, ~505 s)

| Obligation | Result |
|---|---|
| **C1 per-expert** | **1664 experts** (26 × 64), **global worst `max_abs_diff` 1.907e-6** (layer 26, expert 57), gate `< 0.5` → PASS (~2.6e5× inside), **0 failures**, exhaustive |
| **C2 router** | top-6 **set equality** on **all 26 MoE layers**, **0 failures**, **min routing margin 0.011981** (layer 23) — `norm_topk_prob=false` does not affect the set (softmax monotonic) |
| **C3 attention** | reused at the **mechanism** level (DeepSeek MLA cert 9.999e-6 + MLA-0 V2-Lite-like full forward 9.072e-5) — same standard as Qwen-MoE C3 |
| **C4 / C5** | C4 (deepseek_scale topology 7.806e-3) **available, not folded**; C5 active-path **pending** |

## Does DeepSeek-V2-Lite reach partial L1?

**Yes — partial L1.** C1 + C2 are **certified on the real weights** across every
MoE layer (the substantive L1 work), and C3 is satisfied at the mechanism level
(the same way Qwen-MoE's C3 was). So DeepSeek-V2-Lite is at **ADR-007 L1 with the
C3-mechanism caveat** — NOT the dense ADR-004 `CERTIFIED`, and **NOT L2/L3/L4**
(C4 fold → L2 and C5 active-path → L3 are the next phases). Reported strictly per
ADR-007 (`docs/numcert/deepseek-v2-lite.moecert.json`, `schema_variant:
moe-decomposition`).

## Tooling reuse

~95% of the Qwen MOE-CERT-2-ext machinery transferred: the `ShardCache` +
`compute_layer` generator (identical SwiGLU + router math), the multi-shard
harness, the per-expert / top-k-set / routing-margin logic. DeepSeek-only deltas:
skip dense layers, config keys (`n_routed_experts`), k=6.

## Regression / scope

**No `src/` change** — the harness only *calls* `RealMoeLayer::assemble` +
per-expert `forward` + `top_k_routing` (the certified MoE primitives). MLA / YaRN /
runtime / loader / numerics / Adapter Toolkit untouched; ADR-004 gate not lowered.

## Files

- `fixtures/moe/generate_deepseek_v2lite_decomposition_reference.py` (new).
- `fixtures/moe/deepseek_v2lite_decomp_ref.{safetensors,json}` (new, ~13.6 MB).
- `tests/moe_mla1_deepseek_decomposition_test.rs` (new, `#[ignore]`).
- `docs/numcert/deepseek-v2-lite.moecert.json` (new, partial-L1 manifest).
- `docs/STATUS.md`, this handoff.

## Caveats / risks

- **C3 mechanism, not real-weight** (inherited; same as Qwen/Mixtral).
- **Single probe input** for C1/C2 (representative activation, not an inter-layer
  trace) — consistent with the Qwen approach; other inputs uncovered.
- **f32-vs-f64 DeepSeek drift** is looser than Qwen at full-forward (~1e-3), but
  per-expert C1 here is ~1.9e-6.

## MLA-1 (C4) update — folded → L2

C4 (assembly/topology) was folded into the manifest (manifest/docs only, no code).
**Primary C4 evidence = the MLA-0 V2-Lite-topology full-forward cert** vs HF f64 =
**9.072e-5** — a tiny real `DeepseekV2ForCausalLM` end-to-end with V2-Lite's *exact*
conventions (**YaRN + dense-first + no-renorm + MLA + q_lora=null**), argmax exact —
**corroborated** by `deepseek_scale` (DeepSeek-MoE 16-expert MLA topology, **7.806e-3**).
MLA-0 is preferred over deepseek_scale-alone because deepseek_scale uses *different*
conventions (renorm, no dense-first, no YaRN); MLA-0 carries V2-Lite's correct ones.
Both tests re-run green. Caveat: C4 is reduced-dim, **random-weight** topology — the
mechanism, not the real 64-expert trained-weight assembly; it transfers only with the
real-weight C1/C2 (met). Result: **DeepSeek-V2-Lite — MoE-certified L2 (whole model)
= L1 + C4** (`ladder_level_whole_model: L2`). Not dense ADR-004 `CERTIFIED`, not L3/L4.

## Next (toward L3)

- **MLA-1 / C5** — real-weight active-path full forward (MLA + YaRN + dense-first +
  MoE) vs F64 one-layer-at-a-time → **L3** (Qwen-scale, RAM-feasible on 32 GB).
- L4 (global F64 ~126 GB) reserved/unreachable.
