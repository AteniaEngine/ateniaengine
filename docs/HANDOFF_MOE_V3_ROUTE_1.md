# HANDOFF — MOE-V3-ROUTE-1: DeepSeek-V3-like routing mechanism → L0

Milestone: **MOE-V3-ROUTE-1** — implement the **DeepSeek-V3-like routing mechanism**
(modern router primitives) and certify it at **L0 (mechanism/topology only)** on a
reduced-dim fixture vs a HuggingFace `DeepseekV3MoE` float64 reference. **No real V3
weights, no downloads, no runtime/loader/Adapter-Toolkit change, no L1/L2/L3 claim.**

## Result

**DeepSeek-V3-like routing mechanism reaches L0 (mechanism).** A reduced-dim DeepSeek-V3
MoE block — sigmoid scoring + `e_score_correction_bias` selection + group-limited top-k +
`routed_scaling_factor` + SwiGLU experts + ungated shared expert — reproduced from raw
weights by Atenia's new router and compared end-to-end vs HF f64:

| Metric | Value |
|---|---|
| Router **set-equality** vs HF (selected experts) | **6/6 tokens exact** |
| Worst **combine-weight** diff vs HF | **1.192e-7** |
| Worst **MoE-block** `max_abs_diff` vs HF f64 | **3.891e-8** (gate `< 1e-3`) |
| Min selection margin | **5.56e-2** (comfortable; no near-ties) |
| Determinism | yes (router bit-identical on re-run) |

→ **DeepSeek-V3-like routing mechanism: L0 mechanism/topology only.** **Not** real-weight
certified, **not** L1/L2/L3, **not** the dense ADR-004 `CERTIFIED`; **L4** (global F64)
reserved/unreachable. Real-weight V3 Ln remains **provisioning-blocked** (no small
V3-family checkpoint; 671 GB–1 TB), as documented in `DEEPSEEK_V3_GAP_AUDIT.md` — this
milestone closes the **code/mechanism** gap for the router, not the model-scale gap.

## What was implemented

`src/moe/v3_router.rs` — a self-contained, fail-loud **reference router**, isolated like
`dense.rs`/`sparse.rs` (no runtime/loader/graph/CUDA/Adapter-Toolkit wiring). It computes,
from raw router logits + `e_score_correction_bias`, the selected experts + combine weights,
matching `transformers` v5.6.2 `DeepseekV3MoE.route_tokens_to_experts` exactly:

- `ScoringFunc::Sigmoid` (fail-loud on any other `scoring_func`, incl. `"softmax"` — that
  has its own certified path; this module is the V3 sigmoid mechanism only).
- `scores = sigmoid(logits)`; `scores_for_choice = scores + bias` (**bias used for
  selection only**).
- group score = **sum of the top-2** `scores_for_choice` per group; select `topk_group`
  groups; mask experts outside them to `-inf`; take top-`top_k` experts.
- combine weight = **original** `scores` (no bias) at the selected experts; `/= (Σ + 1e-20)`
  if `norm_topk_prob`; `*= routed_scaling_factor`.
- f64 internals; deterministic tie-break (lower index). Fail-loud validation:
  `n_routed % n_group`, `experts_per_group ≥ 2`, `topk_group ≤ n_group`,
  `top_k ≤ topk_group·experts_per_group`, length/finite checks.

Out of scope (explicitly NOT implemented): Q-LoRA query path, FP8-in-MoE, MTP, V3 real
weights, productive loader wiring, Adapter Toolkit. Qwen/DeepSeek-V2-Lite/Mixtral certs
untouched.

## Tooling

- **Reference generator** (`fixtures/moe/generate_v3_route_reference.py`): builds a
  reduced-dim HF `DeepseekV3MoE` (`hidden=16`, `inter=8`, 8 routed experts, 4 groups of 2,
  `topk_group=2`, `top_k=2`, 1 shared, `routed_scaling_factor=2.5`, `norm_topk_prob=true`,
  nonzero `e_score_correction_bias`) in float64; dumps weights (per-expert gate/up/down
  unpacked from the packed `gate_up_proj`/`down_proj`, router weight+bias, shared MLP) +
  6 probe hidden states + the f64 reference (dense combine weights, block outputs, selected
  experts, selection margins). Offline; not run in CI.
- **Harness** (`tests/moe_v3_route_scale_cert_test.rs`, `deepseek_v3_route_scale_certifies`):
  rebuilds the block from raw weights using `v3_router::v3_route` + `MoeDenseExpert`,
  gates router set-equality + combine-weight diff + block `max_abs_diff < 1e-3` +
  determinism. Runs in CI (`cargo test --lib` plus this `--test`).

## Validation

- New router unit tests (`src/moe/v3_router.rs`): 8 — sigmoid scoring, bias-affects-
  selection-not-combine, group-limiting exclusion, `routed_scaling_factor`, `norm_topk_prob`,
  determinism, fail-loud config/input cases.
- L0 cert test: PASS (set-equality 6/6, block 3.891e-8, deterministic).
- Regressions: `moe_scale_cert_test` (Mixtral/Qwen/DeepSeek) 3/3; full `cargo test --lib`
  green; the three certified families unchanged.

## What still blocks real DeepSeek-V3 (L1+)

- **Provisioning (the binding constraint):** no small V3-family checkpoint exists (V3 =
  671 B, Kimi-K2 ≈ 1 T); a real-weight C1/C2/C5 is hardware-blocked on a 32 GB host —
  independent of code.
- **Remaining code gaps for a full V3 forward** (not needed for the router L0, tracked for
  later): Q-LoRA query path (`q_a_proj`→RMSNorm→`q_b_proj`), FP8-in-MoE weight intake, MTP
  head (auxiliary, not on the main cert path), V3.2 DSA sparse attention (separate).

## Files

- `src/moe/v3_router.rs` (new), `src/moe/mod.rs` (+`pub mod v3_router`).
- `fixtures/moe/generate_v3_route_reference.py` (new), `fixtures/moe/v3_route_ref.{safetensors,json}` (new).
- `tests/moe_v3_route_scale_cert_test.rs` (new).
- docs: this handoff + `STATUS.md`, `DEEPSEEK_V3_GAP_AUDIT.md`, `MOE_COVERAGE_AUDIT.md`.

## Risks / caveats

- L0 mechanism only — random-weight reduced-dim fixture; transfers to real V3 **only in
  conjunction with** real-weight C1/C2/C5, which are provisioning-blocked.
- The HF router computes logits/sigmoid in float32; Atenia's router is f64 (more precise).
  Differences are ~1e-7 (combine weights) / ~4e-8 (block) and set-equality is exact at the
  fixture's margins (min 5.6e-2); a pathological near-tie (margin ≲ 1e-6) could in principle
  diverge — surfaced by the reported selection margin, not hidden.
- A future transformers/DeepSeek point release could adjust router knobs; the algorithm is
  pinned to v5.6.2 `DeepseekV3MoE` and cited in the module + generator.
