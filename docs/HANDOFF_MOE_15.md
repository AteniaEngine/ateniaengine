# HANDOFF — MOE-15: packed / fused expert tensor support

Milestone: **MOE-15** (assemble MoE experts from the packed/fused tensor
layout used by modern Qwen2-MoE / Mixtral checkpoints, in addition to the
classic per-expert layout). Correctness-first, CPU-only, experimental,
opt-in. The MOE-2 fail-loud guard is **unchanged**. No CUDA, ROCm, Metal,
tier-planner, CLI, generation, Adapter Toolkit, batching, optimisation, model
download, or model committed. Predecessor: MOE-0..14 + docs `e0c8945`.

## Motivation

The MOE-14 real smoke runs showed that recent `transformers` checkpoints pack
all experts into 3-D tensors:

```text
model.layers.{L}.mlp.experts.gate_up_proj    # gate+up fused, all experts
model.layers.{L}.mlp.experts.down_proj       # down, all experts
```

instead of the classic per-expert layout
(`mlp.experts.{E}.{gate,up,down}_proj.weight`). Atenia detected them as MoE
but could not assemble them. MOE-15 closes that gap.

## Supported packed formats

The packed tensors are recognised by name (with or without a trailing
`.weight`):

- `…mlp.experts.gate_up_proj` → role `MoePackedGateUp`
- `…mlp.experts.down_proj` (no per-expert id) → role `MoePackedDown`

Classic per-expert tensors keep their existing roles unchanged.

## Assumed layout (verified internally + on real checkpoints)

```text
gate_up_proj : [num_experts, 2*d_ff, d_model]   # first d_ff rows = gate, next d_ff = up
down_proj    : [num_experts, d_model, d_ff]
```

Derivation: `num_experts = gate_up.shape[0]`, `d_ff = gate_up.shape[1] / 2`
(must be even), `d_model = gate_up.shape[2]`; `down_proj` is cross-checked as
`[num_experts, d_model, d_ff]`. If the shapes are inconsistent, binding
returns a `ShapeInconsistency` / `PackedBadRank` error rather than mis-slicing.

For expert `E` (row-major):
- gate = `gate_up[E][0 .. d_ff, :]`
- up   = `gate_up[E][d_ff .. 2*d_ff, :]`
- down = `down[E]`

The gate-first ordering follows the standard fused-`gate_up` convention. The
tests build packed tensors with exactly this layout and assert the sliced
per-expert weights equal an equivalent classic reference layer, and that the
packed-layer forward matches the classic reference forward within `1e-5`.
**Note:** this verifies the slice/split is *self-consistent* and dimensionally
correct; it does **not** assert bit-exact agreement with HuggingFace's own
forward (no F64 reference here). No full Mixtral / Qwen-MoE support is claimed.

## How gate/up/down are extracted

`build_packed_layer(map, layer_id, top_k, resolve)` (in `src/moe/binding.rs`):
resolves `gate_up_proj` + `down_proj` once, derives dims, slices each expert's
contiguous block, splits gate/up, builds a `MoeDenseExpert`, attaches the
router (`[num_experts, d_model]`), and returns the same `MoeDenseLayer` type
the classic path produces — so the stack / validation / smoke pipeline is
unchanged downstream.

## Format selection (classic vs packed)

`RealMoeLayer::assemble` now selects per layer:
- both classic and packed present → **`MixedExpertFormat` error** (ambiguous,
  refuse rather than guess);
- classic present → classic path (`build_real_layer`), unchanged;
- packed present → packed path (`build_packed_layer`);
- neither → the classic `NoExperts` error.

This preserves the classic Qwen1.5-MoE path that already passed in MOE-14.

## Real smoke re-runs (manual, out of CI)

Re-ran the MOE-14 opt-in smoke against the three tiny checkpoints still on the
local disk (no re-download):

| Checkpoint | Expert format | MOE-14 result | MOE-15 result |
|---|---|---|---|
| `hf-internal-testing/tiny-random-Qwen2MoeForCausalLM` | packed | forward INCOMPLETE | **SMOKE PASS** (layers=2, experts=8, shared=2, d_model=64) |
| `hf-internal-testing/tiny-random-MixtralForCausalLM` | packed | forward INCOMPLETE | **SMOKE PASS** (layers=2, experts=8, shared=0, d_model=64) |
| `katuni4ka/tiny-random-qwen1.5-moe` | classic | SMOKE PASS | **SMOKE PASS** (layers=4, experts=32, shared=4, d_model=32) |

A `SMOKE PASS` means the experimental path read the checkpoint, assembled a
stack, and ran a finite forward — NOT that the model is numerically correct,
supported, or production-ready.

## What is still missing for real Mixtral / Qwen-MoE

- Numerical equivalence vs a reference forward (no F64/HF comparison yet).
- Transformer structure (residuals, norms, attention, embeddings/lm_head,
  multi-token / KV cache).
- `config.json`-driven topology beyond the minimal MOE-14 fields; shared-
  expert gating; fail-loud lift behind an explicit opt-in; graph registry.
- Large real checkpoints (the harness materialises all experts in f32, so it
  only scales to small checkpoints today).

## What was NOT implemented

- No fail-loud lift, no Adapter Toolkit / loader load-path / CLI / generation
  change, no CUDA/ROCm/Metal, tier-planner, batching, optimisation.
- No model download in CI, no model committed to the repo.

## Confirmation

- **Fail-loud preserved**: packed checkpoints are still detected as MoE
  (`detect_moe` counts packed roles); the productive loader still refuses
  them. Re-asserted by `fail_loud_still_active`.
- Classic per-expert path unchanged (`classic_per_expert_still_passes`).
- Dense models unaffected (`dense_models_still_load`).

## Tests

- `tests/moe_packed_experts_test.rs` — 8 integration tests:
  `detects_packed_gate_up`, `packed_metadata_extracts_num_experts`,
  `packed_binding_extracts_gate_up_down` (vs classic reference, exact),
  `packed_layer_forward` (vs reference, 1e-5), `classic_per_expert_still_passes`,
  `ambiguous_classic_and_packed_reports_error`, `fail_loud_still_active`,
  `dense_models_still_load`.

Local validation: `cargo test --lib --release -- --test-threads=1` →
**715 passed / 0 failed / 1 ignored**. Packed suite: 8/8. Real smoke re-runs:
3/3 SMOKE PASS (see table).

## Files modified

* `src/moe/detect.rs` — `MoePackedGateUp` / `MoePackedDown` roles + classifier
  + detection counting.
* `src/moe/data_plane.rs` — `MoeLayerMap.packed_gate_up` / `packed_down`,
  `has_classic_experts()`, `has_packed_experts()`, routing in `from_tensors`.
* `src/moe/binding.rs` — `PackedExpertDims`, `packed_dims`,
  `build_packed_layer`, packed error variants.
* `src/moe/layer.rs` — format selection in `assemble`, `MixedExpertFormat`.
* `src/moe/validation.rs` — packed-aware config derivation + expert count.
* `src/moe/mod.rs` — re-exports.
* `tests/moe_packed_experts_test.rs` — new (8 integration tests).
* `docs/HANDOFF_MOE_15.md` — this file; `docs/HANDOFF_MOE_14.md` cross-note.

No loader load-path, Adapter Toolkit, CLI, generation, CUDA, ROCm, Metal,
tier-planner, or graph changes. Fail-loud preserved.
