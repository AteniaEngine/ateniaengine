# HANDOFF — MOE-FULL-11: Qwen-MoE runtime + DeepSeek-MoE block certification

Milestone: **MOE-FULL-11** — turn the controlled MoE runtime (MOE-FULL-10,
Mixtral-only) into **real family support** for Qwen-MoE (end-to-end) and
DeepSeek-MoE (MoE-block certification), plus extended Mixtral validation and
robustness. **Experimental, CPU, opt-in (`ATENIA_EXPERIMENTAL_MOE=1`).** No
fail-loud lift of the dense loader, no general support declared, no CLI / VRAM /
batching / quant. Predecessor: MOE-FULL-10 (`339d5e4`).

## Audit finding (reported & resolved before coding)

Real attention differs by family: **Qwen2-MoE** has **Q/K/V bias** (standard
attention); **DeepSeek-V2/V3** use **MLA** (`kv_a_proj_with_mqa` / `kv_b_proj`,
no `k_proj`/`v_proj`). The MOE-FULL-6 graph models standard attention without
bias. Per the user's decision: **add attention bias** (Qwen end-to-end) and
**certify only the DeepSeek MoE block** (MLA is a new architecture, out of
scope). Documented, not faked.

## 1. Qwen-MoE — end-to-end runtime (opt-in)

The runtime is generalised from `MixtralRuntime` to a family-aware **`MoeRuntime`**
(`MixtralRuntime` kept as an alias). It enables **Mixtral** and **Qwen-MoE**:

- **Attention bias** (`full_forward.rs::QkvBias`, `add_proj_bias`): optional
  per-layer Q/K/V biases added post-projection in all three build functions
  (full forward, prefill, decode). The Q bias absorbs the `1/sqrt(head_dim)`
  score scale (mirrors the pre-scaled `w_q`); K/V biases are tiled to MHA shape
  by `gqa::to_mha_kv` exactly like the K/V weights. **`None` (Mixtral/MHA) is
  byte-identical to MOE-FULL-6** — every prior test stays green.
- **Per-family knobs** (the only divergence): Qwen → `has_shared_expert=true`,
  routed `d_ff = moe_intermediate_size`, attention bias; Mixtral → no shared, no
  bias, `d_ff = intermediate_size`. The shared expert (sigmoid-gated) +
  `norm_topk_prob=false` convention is auto-resolved by the certified
  `RealMoeLayer::forward_auto` (HuggingFaceQwen convention) — no new MoE math.
- **End-to-end HF f64 parity** (real tiny Qwen2-MoE, GQA n_kv=2):
  **max_abs_diff 5.960e-08**, per-position argmax match, `generate → EOS`
  (`[9, 31]` stop on eos=31), deterministic — all through the productive
  `MoeRuntime` (no test helpers).

## 2. DeepSeek-MoE — MoE-block certification (no end-to-end)

DeepSeek-V2/V3 use MLA → the runtime **refuses** them with a clear
`UnsupportedFamily` message. The **MoE block** (router + packed routed experts +
shared expert) is certified separately: a tiny `DeepseekV2MoE` block run in f64
with simple routing (`topk_method=greedy`, `n_group=1`,
`routed_scaling_factor=1.0`, `scoring_func=softmax`, `norm_topk_prob=True`) so it
reduces to the certified top-k softmax + renormalise + ungated-shared
convention, which `RealMoeLayer::forward_auto` reproduces.

- **MoE block vs HF: max_abs_diff 2.196e-04** (mean 9.510e-05). This is higher
  than Mixtral/Qwen (~6e-8) — attributable to f32-vs-f64 accumulation through
  the block on a unit-normal probe (vs token-embedding inputs elsewhere); well
  within the 1e-3 tolerance, argmax/structure consistent. **Drift documented,
  not hidden.**

## 3. Mixtral — extended validation

The productive runtime's full-sequence forward logits match the HF f64
reference: **max_abs_diff 7.451e-08** (ties the runtime to the MOE-FULL-6 logit
certification, not just greedy ids).

## 4. Robustness

`MoeRuntime::load_from_files` returns specific errors:
`dense → NotMoe`, `MLA → UnsupportedFamily`, `missing tensor → Load`,
`invalid config.json → Config`, `config/tensor expert-count mismatch →
ConfigInconsistent`. Plus a happy-path regression guard.

## Results by family

| Family | Path | HF parity | generate→EOS |
|---|---|---|---|
| Mixtral | end-to-end (FULL-10/11) | **7.451e-08** (full logits) | ✅ |
| Qwen-MoE | end-to-end (FULL-11) | **5.960e-08** (full logits) | ✅ |
| DeepSeek-MoE | **MoE block only** | **2.196e-04** (block) | ❌ (MLA, refused) |

## What stays protected by fail-loud

- The **dense loader** refuses every MoE checkpoint, always.
- The MoE runtime refuses **without** the opt-in.
- DeepSeek end-to-end refused (MLA). No other families. No CLI / VRAM / batching
  / quant / general support.

## Remaining risks / work (MOE-FULL-12)

- DeepSeek MLA attention (new architecture) for end-to-end DeepSeek.
- DeepSeek block drift (2e-4) is f32 vs f64; an f64 path would tighten it.
- VRAM expert tier, decode hot-path through residency+cache (perf), CLI, real
  large-checkpoint certification.

## Tests

- `src/moe/gqa.rs` (unchanged 5), `full_forward.rs` / `generate.rs` (bias
  plumbing; MHA path byte-identical).
- `tests/moe_qwen_runtime_test.rs` — 3 (family/config, full-forward vs HF,
  generate→EOS deterministic).
- `tests/moe_deepseek_block_test.rs` — 2 (family classification, block vs HF).
- `tests/moe_mixtral_runtime_test.rs` — +1 (`mixtral_runtime_forward_matches_hf`).
- `tests/moe_runtime_robustness_test.rs` — 6 (error paths + happy path).

Local validation (real output, exit 0): full lib suite
`cargo test --lib --release -- --test-threads=1` → **781 passed / 0 failed /
1 ignored**. MoE integration: qwen 3/3, deepseek 2/2, mixtral 3/3, robustness
6/6, decode 5/5, full_forward 7/7, gqa 3/3, residency 4/4, family-loader 4/4,
**moe_loader_failloud 3/3**. Dense regression: `llama_3_2_numerical_validation`
4/4.

## Files modified

* `src/moe/full_forward.rs` — `QkvBias` + `add_proj_bias`; optional bias in the
  attention graph (None = byte-identical to MOE-FULL-6).
* `src/moe/generate.rs` — bias applied in prefill + decode.
* `src/moe/runtime.rs` — generalised to `MoeRuntime` (family-aware: Mixtral +
  Qwen-MoE; refuses DeepSeek/MLA); `forward_logits`; `family()`; `MixtralRuntime`
  / `MixtralRuntimeError` aliases.
* `src/moe/mod.rs` — `MoeRuntime` / `MoeRuntimeError` re-exports.
* `tests/moe_qwen_runtime_test.rs`, `tests/moe_deepseek_block_test.rs`,
  `tests/moe_runtime_robustness_test.rs` — new.
* `tests/moe_mixtral_runtime_test.rs` — extended.
* `fixtures/moe/generate_qwen_moe_reference.py` + `qwen_moe_tiny.{safetensors,json}`
  + `qwen_moe_tiny_config.json` — new (~110 KB).
* `fixtures/moe/generate_deepseek_block_reference.py` +
  `deepseek_block.{safetensors,json}` — new (small).
* `docs/HANDOFF_MOE_FULL_11.md` — this file.
* `docs/MOE_FULL_PATH_AUDIT.md`, `docs/MOE_OVERVIEW.md`,
  `docs/MODEL_FAMILY_VALIDATION.md` — updated.

No dense-loader / CLI / runtime-dense / Adapter-Toolkit / CUDA change. Dense
models unaffected; fail-loud preserved except the explicit, opt-in,
Mixtral+Qwen-MoE runtime.
