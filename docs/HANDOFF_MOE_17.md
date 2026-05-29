# HANDOFF — MOE-17: HF convention parity mode

Milestone: **MOE-17** (an optional, opt-in execution convention that makes
Atenia's Qwen-MoE block match the HuggingFace `transformers` reference, while
keeping Atenia's existing default unchanged). Correctness-first, CPU-only,
experimental. The MOE-2 fail-loud guard is **unchanged**. No CUDA, ROCm,
Metal, tier-planner, CLI, generation, Adapter Toolkit, batching, optimisation.
Predecessor: MOE-0..16 (`bdc09a0`).

## Conventions supported

`MoeExecutionConvention` (in `src/moe/layer.rs`):

- **`Atenia`** (default, unchanged): selected top-k softmax weights are
  **renormalised** to sum 1; the shared expert (if any) is added **ungated**.
  This also exactly matches Mixtral.
- **`HuggingFaceQwen`** (opt-in): selected top-k weights are **not**
  renormalised (`norm_topk_prob = false`); the shared expert is scaled by
  `sigmoid(shared_expert_gate · x)`.

`RealMoeLayer::forward(x)` is unchanged — it delegates to
`forward_with(x, MoeExecutionConvention::Atenia)`. The new
`forward_with(x, convention)` selects the convention. The Atenia default is
preserved as the only behaviour any existing caller sees.

## How the two differences are implemented

1. **Top-k probability mode** — `sparse::top_k_routing_with(weights, k,
   renormalize)` and `MoeDenseLayer::forward_sparse_with(x, k, renormalize)`.
   `renormalize = true` is the existing Atenia/Mixtral path (kept as the
   default of `top_k_routing` / `forward_sparse`); `false` keeps the raw
   softmax weights at the selected indices (HF Qwen).
2. **Shared-expert sigmoid gate** — `RealMoeLayer` now captures the optional
   `shared_expert_gate` weight (`[d_model]`, ignored by the Atenia default).
   Under `HuggingFaceQwen`, the shared output is scaled by
   `sigmoid(gate · x)` (f64 accumulation); with no gate tensor it falls back
   to ungated.

## Atenia vs HF — differences

| Aspect | Atenia default | HuggingFaceQwen |
|---|---|---|
| top-k weights | renormalised to sum 1 | raw softmax (no renorm) |
| shared expert | added ungated | scaled by `sigmoid(gate·x)` |
| Mixtral | exact match (Mixtral renormalises, no shared) | n/a (Mixtral uses Atenia) |

## Results — before / after (Atenia vs HuggingFace transformers f64)

`max_abs_diff` against the HF transformers block reference; argmax matched in
all cases, before and after.

| Model | Before (Atenia default) | After (HF convention) | Argmax |
|---|---|---|---|
| qwen15_moe | 2.369e-4 | **2.910e-11** | ✅ |
| qwen2_moe  | 3.647e-4 | **5.821e-11** | ✅ |
| mixtral    | 1.164e-10 | 1.164e-10 (Atenia default; HF==Atenia) | ✅ |

The HF convention drives the Qwen-MoE divergence from ~3e-4 down to ~1e-10 —
matching HuggingFace to machine precision. Mixtral is unchanged: its
convention already equals Atenia's, so the default already matches HF.

The Atenia default is also re-verified unchanged: `forward` (default) still
matches the MOE-16 primary f64 reference at < 1e-4 (~1e-10) for all three
models.

## What was NOT implemented

- No change to the Atenia default behaviour (it remains the default for every
  existing caller and test).
- No automatic convention selection from `config.json` (the caller chooses;
  config-driven selection is Adapter Toolkit territory, deferred).
- No full-model logits comparison, no transformer stack, no fail-loud lift, no
  Adapter Toolkit / loader load-path / CLI / generation change, no
  CUDA/ROCm/Metal, tier-planner, batching, optimisation.

## Confirmation

- **Fail-loud preserved** (`fail_loud_still_active`): the fixtures are real
  MoE tensors; `detect_moe` still fires; the productive loader is untouched.
- **Atenia remains the default** (`atenia_mode_preserves_existing_results`):
  `forward` is unchanged; the HF convention is strictly opt-in via
  `forward_with`.
- No Adapter Toolkit / CLI / CUDA / generation changes.

## Tests

- `tests/moe_hf_convention_test.rs` — 8 integration tests:
  `topk_prob_mode_changes_output`, `shared_gate_changes_output`,
  `atenia_mode_preserves_existing_results`, `hf_qwen_mode_matches_reference`,
  `mixtral_still_matches_reference`, `metrics_are_deterministic`,
  `fail_loud_still_active`, `dense_models_still_load`.
- Fixtures regenerated to include `shared_expert_gate.weight` and a diagnostic
  `atenia_hf_ref` (Atenia's f64 reimpl under the HF convention), which matches
  the HF block to ~1e-11 — confirming the convention is implemented correctly.

Local validation: `cargo test --lib --release -- --test-threads=1` →
**722 passed / 0 failed / 1 ignored**. HF-convention suite: 8/8. MOE-16
numerical suite still 7/7 (Atenia default unchanged).

## Files modified

* `src/moe/sparse.rs` — `top_k_routing_with`, `forward_sparse_with`
  (renormalise flag); existing functions delegate with `true`.
* `src/moe/layer.rs` — `MoeExecutionConvention`, `RealMoeLayer.shared_gate`,
  `resolve_shared_gate`, `forward_with`, `sigmoid_dot`; `forward` delegates to
  the Atenia convention.
* `src/moe/mod.rs` — re-exports.
* `fixtures/moe/generate_reference.py` — emit `shared_expert_gate.weight` +
  `atenia_hf_ref` diagnostic.
* `fixtures/moe/{qwen15_moe,qwen2_moe,mixtral}_layer0.{safetensors,json}` —
  regenerated (added the shared gate tensor + HF-convention diagnostic).
* `tests/moe_hf_convention_test.rs` — new (8 integration tests).
* `docs/HANDOFF_MOE_17.md` — this file.

No loader load-path, Adapter Toolkit, CLI, generation, CUDA, ROCm, Metal,
tier-planner, or graph changes. Fail-loud preserved; Atenia default preserved.
