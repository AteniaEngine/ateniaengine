# HANDOFF — QWEN-MOE-CERT-1: Qwen-MoE family certification

Goal: certify a **representative set of Qwen-MoE checkpoints** through the
experimental MoE path (not more generic infrastructure). Correctness-first,
CPU-only, experimental, fixture-based. The MOE-2 fail-loud guard is
**unchanged** — the productive loader still refuses MoE checkpoints. No CUDA,
CLI, generation, Adapter Toolkit changes. Predecessor: MOE-0..19 (`a008f00`).

## Models tested

| Model | Source repo | Arch | Experts | Shared | norm_topk_prob | Router name |
|---|---|---|---|---|---|---|
| qwen15_moe | `katuni4ka/tiny-random-qwen1.5-moe` | Qwen2Moe | classic per-expert | yes (+gate) | false | `mlp.gate.weight` |
| qwen2_moe | `hf-internal-testing/tiny-random-Qwen2MoeForCausalLM` | Qwen2Moe | packed/fused | yes (+gate) | false | `mlp.gate.weight` |
| qwen3_moe | `hf-internal-testing/tiny-random-Qwen3MoeForCausalLM` | Qwen3Moe | packed/fused | **no** | **true** | **`mlp.router.weight`** |

All three are **tiny / random-weight** test checkpoints (safetensors, tens of
MB). They are **not** real full models — certification here is of the MoE
*block math + family conventions*, not end-to-end generation.

## Models downloaded

- `qwen15_moe`, `qwen2_moe` — already on local disk (MOE-14..18).
- `qwen3_moe` — newly downloaded (`hf-internal-testing/tiny-random-
  Qwen3MoeForCausalLM`, ~tens of MB; f32 RAM < 100 MB). Not committed to the
  repo. RAM/disk verified first (14.3 GB free / 1.2 TB free).

A real full Qwen1.5-MoE-A2.7B (~14.3B params → ~57 GB f32) was **not**
downloaded: the current block-only harness materialises all experts in f32 and
has no full transformer path — infeasible and out of scope (documented blocker
in `docs/MOE_OVERVIEW.md`).

## Results — smoke (MOE-14 harness, real on-disk checkpoints)

| Model | layers | experts | shared layers | d_model | forward_pass_ok |
|---|---|---|---|---|---|
| qwen15_moe | 4 | 32 | 4 | 32 | ✅ true |
| qwen2_moe | 2 | 8 | 2 | 64 | ✅ true |
| qwen3_moe | 2 | 8 | 0 | 64 | ✅ true (after fix) |

## Results — numerical (vs HuggingFace transformers f64, argmax-matching)

`forward_auto` (auto-resolved convention) vs the HF block reference:

| Model | Resolved convention | MaxDiff | MeanDiff | RMSE | Argmax |
|---|---|---|---|---|---|
| qwen15_moe | HuggingFaceQwen | 2.910e-11 | 9.58e-12 | 1.22e-11 | ✅ |
| qwen2_moe | HuggingFaceQwen | 5.821e-11 | 1.21e-11 | 1.73e-11 | ✅ |
| qwen3_moe | Atenia | 5.821e-11 | 1.60e-11 | 2.31e-11 | ✅ |

All pass the ADR-004 gate (`max_abs_diff < 0.5`) with a ~10-order margin, and
all match the HF reference to ~1e-10. Qwen3-MoE (no shared expert,
`norm_topk_prob = true`) correctly auto-resolves to the **Atenia** convention,
which equals the HF Qwen3-MoE block — so the auto-resolver's Atenia branch is
validated against a real Qwen-MoE checkpoint.

## Errors found & fixes applied

**Error: Qwen3-MoE smoke failed — "layer 0 has no router tensor".**
Root cause: Qwen3-MoE stores the router on disk as `…mlp.router.weight`,
whereas Qwen2-MoE / Mixtral use `…mlp.gate.weight`. Atenia's
`is_moe_router_tensor` only recognised `block_sparse_moe.gate.` and
`.mlp.gate.`, so the Qwen3 router was unmapped. (The numerical fixture
initially masked this because HF's `state_dict` renames the router to
`mlp.gate.weight` on load.)

**Fix (bounded to MoE router detection, `src/moe/detect.rs`):** extend
`is_moe_router_tensor` to also match `.mlp.router.`. One-line, additive,
dense-safe (`mlp.router` never appears in dense checkpoints). The qwen3 fixture
was regenerated to carry the real on-disk name `mlp.router.weight` so the
CI numerical test exercises the fix. After the fix the qwen3 smoke is PASS and
the numerical parity is ~7e-11.

No other fixes were needed — packed experts (MOE-15), the Atenia/HF
conventions (MOE-17) and auto-selection (MOE-18) already covered the rest.

## GGUF audit

No Qwen-MoE GGUF checkpoint is present locally and Atenia has **no MoE GGUF
support**. GGUF MoE was **not** implemented in this milestone (out of scope).
Gap recorded; a dedicated **QWEN-MOE-GGUF-1** milestone is proposed if/when GGUF
MoE becomes a target (the GGUF reader would need MoE expert-block decoding +
the same detection/binding path).

## Limitations

- Tiny random-weight checkpoints only; no real full Qwen-MoE model run.
- MoE **block** certified (single layer-0 block), not a full transformer
  forward (no attention/norms/embeddings/KV cache/decode).
- Productive loader still **fails loud** on MoE; nothing wired into
  loader/runtime/Adapter Toolkit/CLI.
- The convention signal is tensor-name metadata (`shared_expert_gate`), not
  parsed `config.json`.

## Certification status

**Qwen-MoE family: partially certified (experimental).**

- ✅ MoE-block numerical parity with HuggingFace (~1e-10) certified for
  Qwen1.5-MoE, Qwen2-MoE and Qwen3-MoE conventions (classic + packed experts,
  with and without shared expert, both `norm_topk_prob` modes, both router
  namings).
- ❌ **Not** production-certified: no real full model, no full transformer
  path, no end-to-end generation, no numcert manifest, fail-loud still active.

"Partially certified" = the family's MoE math + conventions are proven correct
against HF on representative tiny checkpoints; full-family production
certification remains blocked on the items in `docs/MOE_OVERVIEW.md`.

## Tests

- `src/moe/detect.rs` — `qwen3_moe_router_is_detected` unit test (router fix).
- `tests/qwen_moe_cert_test.rs` — 6 integration tests: `qwen15_moe_certifies`,
  `qwen2_moe_certifies`, `qwen3_moe_certifies`, `qwen3_router_naming_is_detected`,
  `family_fail_loud_still_active`, `dense_models_still_load`.

Local validation: `cargo test --lib --release -- --test-threads=1` →
**727 passed / 0 failed / 1 ignored** (was 726). All prior MoE integration
suites still green (numerical 7/7, auto-convention 7/7, HF-convention 8/8,
packed 8/8, fail-loud 3/3).

## Files modified

* `src/moe/detect.rs` — recognise `mlp.router.weight` (Qwen3-MoE) + unit test.
* `fixtures/moe/generate_reference.py` — add qwen3_moe; use on-disk router name.
* `fixtures/moe/qwen3_moe_layer0.{safetensors,json}` — new reference fixture.
* `tests/qwen_moe_cert_test.rs` — new (6 cert tests).
* `docs/HANDOFF_QWEN_MOE_CERT_1.md` — this file.
* `docs/MODEL_FAMILY_VALIDATION.md`, `docs/MOE_OVERVIEW.md` — cross-notes.

No loader load-path, Adapter Toolkit, CLI, generation, CUDA, ROCm, Metal,
tier-planner, or graph changes. Fail-loud preserved.
