# HANDOFF — MOE-FULL-12: DeepSeek-MoE MLA runtime

Milestone: **MOE-FULL-12** — add **MLA** (multi-head latent attention) so the
controlled runtime can run DeepSeek-MoE end-to-end (`load → generate → EOS`)
behind the opt-in. **Experimental, CPU, correctness-first, opt-in
(`ATENIA_EXPERIMENTAL_MOE=1`).** No fail-loud lift of the dense loader, no
optimisation, no batching/quant/VRAM/CLI, only DeepSeek **tiny**. Predecessor:
MOE-FULL-11 (`4df2c3e`).

## 1. MLA audit (the design driver)

DeepSeek-V2/V3 attention differs from MHA/GQA on every axis:

| Component | MLA |
|---|---|
| KV | low-rank: `kv_a_proj_with_mqa` → split `compressed_kv` / `k_pe` → `kv_a_layernorm` → `kv_b_proj` → per-head `k_nope` + `value` |
| RoPE | **decoupled, interleaved (GPT-J)** on the `qk_rope` part only; `k_pe` shared across heads. *Different* from the half-split (NeoX) RoPE used by Llama/Mixtral/Qwen (`gb.rope`). |
| head dims | `qk_head_dim = qk_nope + qk_rope` (scores) ≠ `v_head_dim` (value) |
| scale | `qk_head_dim ** -0.5` |

None of that maps onto the existing MHA graph.

## 2. Design — imperative MLA (correctness > performance)

Forcing MLA into the AMG graph (slice/concat/broadcast + interleaved-RoPE
gymnastics) would be high-risk and invasive. So MLA is implemented
**imperatively** in `src/moe/mla.rs` (f64 accumulation), reusing the certified
MoE block (`RealMoeLayer::forward_auto`):

- `project_token` — q_proj → split nope/pe; `kv_a` → compressed/k_pe →
  `kv_a_layernorm` → `kv_b` → per-head k_nope/value; **interleaved RoPE** on
  q_pe and the shared k_pe at the absolute position.
- `attend` — causal scaled dot-product over `qk_head_dim`, value over
  `v_head_dim`.
- `layer_step` — input-norm → MLA → +residual → post-norm → MoE block →
  +residual, appending to a per-layer **KV cache** (`MlaLayerCache`).
- `forward_prefill` / `forward_decode` / `generate_greedy_eos` — prefill seeds
  the cache, decode is incremental.

`MoeRuntime` gained a **dual backend** (`Backend::Graph` for Mixtral/Qwen,
`Backend::Mla` for DeepSeek) and the new `MoeFamily::DeepSeekMoe`
(detected via `kv_a_proj_with_mqa` / plural `shared_experts`). The MoE-assembly
+ residency + cache + config-validation pipeline is **shared** across all three
families; only the attention weights differ. Mixtral/Qwen paths are unchanged.

## 3. Results (HF f64 reference)

| Check | Value |
|---|---|
| MLA attention (layer 0, isolated) vs HF | **max_abs_diff 9.999e-06** |
| Full-transformer forward vs HF | **max_abs_diff 1.475e-03** |
| Per-position argmax | **match** |
| Greedy generation | **exact** (`[14, 9]`, stop on eos=9) |
| Determinism | ✅ |

### Drift (documented, not hidden)

The MLA attention is essentially exact (~1e-5). The full-forward drift
(**1.475e-03**) is **f32-vs-f64 accumulation dominated by the MoE block**
(~2e-04/layer, MOE-FULL-11) over 2 layers + lm_head — *not* an MLA error. The
**argmax is unaffected**: per-position argmax matches and the greedy ids match
HF exactly. An f64 weight path would tighten the bound; out of scope.

## 4. DeepSeek end-to-end

`MoeRuntime::load_from_files(deepseek_full_config.json, deepseek_full.safetensors)`
→ recognises `DeepSeekMoe`, assembles the MoE blocks + residency, builds the MLA
backend, and `generate([..],8) = [14, 9]` (stops at EOS) — load → generate → EOS,
deterministic, behind the opt-in. Without the opt-in it refuses; the dense
loader is unchanged.

## 5. Robustness

`malformed_deepseek_mla_reports_clear_error` (incomplete MLA checkpoint → clear
`Load`/`Config`/`ConfigInconsistent`), plus the existing matrix (dense → NotMoe,
missing tensor → Load, invalid config → Config, expert-count mismatch →
ConfigInconsistent). The `UnsupportedFamily` error (MOE-FULL-11's MLA refusal)
was removed — MLA is now supported.

## Remaining risks / work (MOE-FULL-13)

- Q-LoRA query compression + YaRN-scaled RoPE (this fixture uses
  `q_lora_rank=None`, default RoPE).
- f32-vs-f64 block drift (1.5e-3); a latent KV cache (the real MLA memory win);
  VRAM tier; decode hot-path through residency+cache; CLI; large-checkpoint
  certification.

## Tests

- `src/moe/mla.rs` — 3 unit (interleaved RoPE identity/rotation, RMSNorm).
- `tests/moe_deepseek_runtime_test.rs` — 4 (family, MLA attention vs HF, full
  forward vs HF, generate→EOS deterministic).
- `tests/moe_deepseek_block_test.rs` — updated (now `DeepSeekMoe` family).
- `tests/moe_runtime_robustness_test.rs` — updated (malformed MLA → clear error).

Local validation (real output, exit 0): full lib suite
`cargo test --lib --release -- --test-threads=1` → **784 passed / 0 failed /
1 ignored** (was 781; +3). MoE integration: deepseek-runtime 4/4, deepseek-block
2/2, qwen 3/3, mixtral 3/3, robustness 6/6, decode 5/5, full_forward 7/7, gqa
3/3, residency 4/4, family-loader 4/4, **loader_failloud 3/3**. Mixtral + Qwen
**no regression** (full HF parity unchanged).

## Files modified

* `src/moe/mla.rs` — new (MLA attention + imperative DeepSeek forward/decode/
  generate + 3 unit tests).
* `src/moe/runtime.rs` — dual `Backend` (Graph/Mla); DeepSeek loader
  (`build_deepseek`) + graph loader (`build_graph`) extracted; `debug_mla_attention`;
  removed `UnsupportedFamily`.
* `src/moe/family.rs` — `MoeFamily::DeepSeekMoe` + descriptor + classification.
* `src/moe/mod.rs` — `pub mod mla;`.
* `tests/moe_deepseek_runtime_test.rs` — new; `tests/moe_deepseek_block_test.rs`,
  `tests/moe_runtime_robustness_test.rs`, `tests/moe_mixtral_runtime_test.rs`
  — updated.
* `fixtures/moe/generate_deepseek_full_reference.py` +
  `deepseek_full.{safetensors,json}` + `deepseek_full_config.json` — new (~48 KB).
* `docs/HANDOFF_MOE_FULL_12.md` — this file.
* `docs/MOE_FULL_PATH_AUDIT.md`, `docs/MOE_OVERVIEW.md`,
  `docs/MODEL_FAMILY_VALIDATION.md` — updated.

No dense-loader / CLI / runtime-dense / Adapter-Toolkit / CUDA change. Dense and
graph (Mixtral/Qwen) paths unaffected; fail-loud preserved except the explicit,
opt-in, three-family runtime.
