# HANDOFF — MOE-FULL-13: production certification campaign

Milestone: **MOE-FULL-13** — expand **coverage** and formalise **certification**
for the three MoE families, **without new architecture / graph ops / families**.
Focus: robustness, certification, coverage, documentation. **Experimental, CPU,
opt-in (`ATENIA_EXPERIMENTAL_MOE=1`).** Predecessor: MOE-FULL-12 (`1565491`).

## Official MoE certification matrix

All references are offline HuggingFace **f64** forwards; drift is Atenia f32 vs
HF f64. "Status" is per-fixture numerical certification; the **family**
production status is **EXPERIMENTAL** (opt-in runtime, tiny fixtures only).

| Family | Fixture | Layout / attention | Status | Drift (max_abs_diff vs HF) | Runtime | Generate | EOS |
|---|---|---|---|---|---|---|---|
| Mixtral | `full_mixtral` | packed experts, MHA | **CERTIFIED** | **7.451e-08** | ✅ | ✅ | ✅ |
| Mixtral | `mixtral_classic` | classic `w1/w3/w2`, MHA | **CERTIFIED** | **7.451e-08** | ✅ | ✅ | ✅ |
| Mixtral | `gqa_mixtral` | packed, **GQA** (n_kv=2) | **CERTIFIED** | **5.960e-08** | ✅ | ✅ | ✅ |
| Qwen-MoE | `qwen_moe_tiny` | packed, GQA, **Q/K/V bias**, shared (sigmoid), `norm_topk_prob=false` | **CERTIFIED** | **5.960e-08** | ✅ | ✅ | ✅ |
| DeepSeek-MoE | `deepseek_full` | **MLA**, low-rank KV, interleaved RoPE, shared | **CERTIFIED†** | attn **9.999e-06** / full **1.475e-03** | ✅ | ✅ | ✅ |
| DeepSeek-MoE | `deepseek_block` | MoE block only | **CERTIFIED (block)** | **2.196e-04** | — | — | — |

† DeepSeek end-to-end: the MLA attention is exact (~1e-5); the full-forward drift
(1.475e-03) is f32-vs-f64 dominated by the MoE block (~2e-4/layer); **per-position
argmax and greedy ids are exact**.

### Variant gaps (UNSUPPORTED / not covered — honest)

| Item | Status | Reason |
|---|---|---|
| Mixtral-8x7B / large checkpoints | UNSUPPORTED | tiny fixtures only; no scale certification |
| Qwen3-MoE | UNSUPPORTED | adds **QK-norm** attention (different architecture) |
| DeepSeek Q-LoRA (`q_a_proj`/`q_b_proj`) | UNSUPPORTED | not implemented (`q_lora_rank=None` only) |
| DeepSeek YaRN-scaled RoPE | UNSUPPORTED | default RoPE only |
| Qwen `norm_topk_prob=true` + shared expert | LIMITATION | convention is auto-resolved from `shared_expert_gate` presence, not `norm_topk_prob`; would mismatch (use `forward_with` to pin) |

## Coverage added (MOE-FULL-13)

- **Mixtral classic layout** end-to-end through the runtime
  (`mixtral_classic.safetensors`, re-packed from the committed packed weights →
  same HF reference). Certifies the classic `build_real_layer` expert path E2E:
  **7.451e-08**, generate `[17,20]`.
- **Mixtral GQA** end-to-end through the runtime (`gqa_mixtral` + new config /
  greedy reference). Certifies GQA tiling through the productive runtime (was
  only graph-tested in MOE-FULL-9): **5.960e-08**, generate `[42,18]` → EOS.
- Mixtral now has **three real on-disk layouts** certified end-to-end:
  packed-MHA, classic-MHA, packed-GQA.

## Robustness added (FASE 5)

- **MLA shape validation** in `build_deepseek`: every MLA tensor length is
  checked against the config (q_proj, kv_a, kv_a_layernorm, kv_b, o_proj) → a
  corrupt MLA checkpoint or a mismatched `kv_lora_rank` fails with a clear
  `Load` error instead of a silent out-of-range panic in the imperative forward.
- New robustness tests: `corrupt_router_shape_reports_error` (router declares 5
  experts but 4 exist → clear error), `corrupt_mla_shape_reports_error` (real
  DeepSeek weights + a config lying about `kv_lora_rank` → clear `Load`).
- Existing matrix unchanged: dense → NotMoe, missing tensor → Load, invalid
  config → Config, expert-count mismatch → ConfigInconsistent, malformed MLA →
  clear error.

## Strategic review (FASE 8)

**Which families can be considered certified?** Numerically (tiny fixtures, vs
HF f64): **Mixtral** (3 layouts), **Qwen-MoE** (1 representative), **DeepSeek-MoE**
(1 end-to-end + block). All to f32 machine precision except the DeepSeek
full-forward (f32-vs-f64 block drift; argmax/greedy exact).

**What remains experimental?** Everything: the runtime is **opt-in** and runs
**tiny fixtures only**. No large-checkpoint certification, no productive load
path, no CLI, no numcert manifest, no VRAM tier, no scale.

**What is needed to drop the "experimental" label from MoE?**
1. Real large-checkpoint certification (Mixtral-8x7B, Qwen2-57B-A14B, …) with the
   MOE-1 partial/sub-reference methodology (full f64 is infeasible at scale).
2. Productive load path + CLI integration (currently a separate opt-in runtime).
3. A numcert manifest per family (like the dense families).
4. Cover the variant gaps: Qwen3-MoE (QK-norm), DeepSeek Q-LoRA + YaRN, and the
   `norm_topk_prob` convention edge case.
5. Tighten DeepSeek drift (f64 weight path) or document the bound as acceptable.
6. Throughput: decode-graph reuse, residency hot-path, VRAM expert tier.
7. A reviewed lift of the dense-loader fail-loud for the certified families.

## Tests

- `tests/moe_certification_test.rs` — 4 (Mixtral classic E2E, Mixtral GQA E2E,
  corrupt router, corrupt MLA shape).
- `src/moe/runtime.rs` — MLA shape validation in `build_deepseek`.

Local validation (real output, exit 0): full lib suite
`cargo test --lib --release -- --test-threads=1` → **784 passed / 0 failed /
1 ignored**. MoE integration (12 binaries): certification 4/4, deepseek-runtime
4/4, deepseek-block 2/2, qwen 3/3, mixtral 3/3, robustness 6/6, decode 5/5,
full_forward 7/7, gqa 3/3, residency 4/4, family-loader 4/4, loader_failloud
3/3. Dense/graph paths untouched.

## Files modified

* `src/moe/runtime.rs` — MLA tensor shape validation in `build_deepseek`.
* `tests/moe_certification_test.rs` — new (4 tests).
* `fixtures/moe/convert_mixtral_classic.py` + `mixtral_classic.safetensors` +
  `mixtral_classic_config.json` — new (classic layout, re-packed).
* `fixtures/moe/gqa_mixtral_config.json` + `gqa_mixtral_gen.json` — new (GQA
  runtime config + greedy reference).
* `docs/HANDOFF_MOE_FULL_13.md` — this file (incl. the official matrix).
* `docs/MOE_OVERVIEW.md`, `docs/MOE_FULL_PATH_AUDIT.md`,
  `docs/MODEL_FAMILY_VALIDATION.md` — updated with the certification matrix.

No new architecture, graph ops, or families. No dense-loader / CLI / CUDA
change. The only `src/` change is defensive MLA shape validation. Fail-loud
preserved except the explicit, opt-in, three-family runtime.
