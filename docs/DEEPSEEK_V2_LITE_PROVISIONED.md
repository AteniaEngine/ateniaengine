# DeepSeek-V2-Lite Provisioned — MLA-1 FASE 1 (download + validate + inventory)

**Provisioning only — model downloaded, validated read-only, inventoried. NO
certification, NO reference generation, NO C1–C5, NO forward/generate, no
runtime/loader/numerics/ATK change.** The model is now local and ready for the
MLA-1 certification milestone.

## Origin

- **Repo:** `deepseek-ai/DeepSeek-V2-Lite` (HuggingFace), license `other` (DeepSeek
  License Agreement), `gated: False`. Authorized + downloaded by the user.
- **Local path:** `models/DeepSeek-V2-Lite/` (new; no existing model moved/deleted).
- **Tool:** `huggingface_hub.snapshot_download` (`allow_patterns` = safetensors +
  index + config + tokenizer + tiny `*.py`/LICENSE; `max_workers=4`).
- **Provenance note:** a first attempt stalled on an internet drop (diagnosed: 0
  bytes/25 s, hung workers, 0/4 shards). On the user's instruction the partial
  `models/DeepSeek-V2-Lite/` (~4.6 GB) was **deleted clean** (no other model
  touched) and **re-downloaded from scratch** — so there are **no resumed/partial
  bytes**; every shard is a complete fresh download.

## Size & shard distribution

- **Total: ~31.4 GB** (4 safetensors shards + 480 KB index + ~5 MB small files).
- Shards (bf16): `model-00001-of-000004` 8.59 GB · `…00002…` 8.59 GB ·
  `…00003…` 8.59 GB · `…00004…` 5.64 GB. **No `.incomplete` left; no `.bin`/`.pt`.**

## Validations performed (read-only — no inference)

| Check | Result |
|---|---|
| Shard count | **4** (`model-0000{1..4}-of-000004.safetensors`) ✅ |
| Index present + valid | `model.safetensors.index.json` (480 KB), references **all 4 shards**, **5291 tensors** ✅ |
| No partial/incomplete leftovers | none ✅ |
| Safetensors headers readable (header-only) | ✅ — `embed_tokens` **BF16** `[102400, 2048]`; expert `down_proj` **BF16** `[2048, 1408]` |
| **MLA tensors present** (layer 5) | `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`, `q_proj`, `o_proj` — all PRESENT ✅ |
| **Dense-first layer** (`first_k_dense_replace=1`) | layer 0 has **dense** `mlp.{gate,up,down}_proj` + MLA attn, **no experts** ✅ |
| MoE layers (1..26) | layer 1 has `mlp.gate` + **64 experts** (`experts.{0..63}.{gate,up,down}_proj`, classic per-expert) + **shared_experts** (3 tensors) ✅ |
| Config matches the audited V2-Lite | ✅ (see inventory) — exactly matches `DEEPSEEK_V2_LITE_FEASIBILITY` |

**No forward, no generation, no certification was run.**

## Technical inventory (FASE 4)

| Field | Value |
|---|---|
| Architecture | `DeepseekV2ForCausalLM` (`model_type=deepseek_v2`) |
| Layers | **27** — layer 0 **dense**, layers 1–26 **MoE** (`first_k_dense_replace=1`, `moe_layer_freq=1`) |
| Hidden size | 2048 · vocab 102400 · `tie_word_embeddings=false` |
| Dense FFN width (layer 0) | `intermediate_size=10944` |
| **Routed experts** | **64** per MoE layer, **top-6** (`num_experts_per_tok=6`), classic per-expert layout |
| **Shared experts** | **2** (`n_shared_experts=2`, fused FFN, ungated) |
| Expert FFN width | `moe_intermediate_size=1408` (Qwen-scale → RAM-feasible) |
| Routing | `scoring_func=softmax`, `topk_method=greedy`, `n_group=1`, `topk_group=1`, **`norm_topk_prob=false`**, `routed_scaling_factor=1.0` |
| **MLA** | `q_lora_rank=None` (no Q-LoRA), `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`, 16 heads |
| **YaRN** | `type=yarn`, `factor=40`, `original_max_position_embeddings=4096`, `beta_fast=32`, `beta_slow=1`, `mscale=0.707`, `mscale_all_dim=0.707` |
| RoPE / norm | `rope_theta=10000`, `rms_norm_eps=1e-6` |
| dtype | **bf16** |
| Total params | ~15.7 B (≈2.4 B active) |

## Atenia readiness for MLA-1-cert

- **Every V2-Lite feature is covered by MLA-0** (validated on a tiny fixture):
  **YaRN** (factor 40, mscale 0.707), **dense-first layer**, **`norm_topk_prob=false`**
  routing convention, **`q_lora_rank=None`** MLA. The sharded loader handles the
  4-shard index; experts are **classic per-expert** (Atenia's certified
  `build_real_layer` path; the real model is classic, unlike the packed MLA-0
  fixture — both are supported).
- **Scale is Qwen-class** (`moe_intermediate=1408`, 64 small experts) → the
  MIXTRAL RAM wall does **not** apply; L1/L2/L3 are RAM-feasible on the 32 GB host
  per `DEEPSEEK_V2_LITE_FEASIBILITY`.

## Risks / caveats

- **License `other`** (DeepSeek License Agreement) — reviewed/accepted before
  download; not OSS-permissive. Recorded.
- **f32-vs-f64 block drift** (~1e-3, MLA-0 measured 9.072e-5 on the tiny fixture) —
  an honest looser bound than Qwen (~1e-7); argmax exact. Expected for DeepSeek.
- **Vocab 102400** (large lm_head) — small memory note for the C5 reference (still
  Qwen-class overall).
- Weights are **not** committed (gitignored `models/`); only this doc is tracked.

## Next steps (MLA-1 cert — NOT this milestone)

1. **MLA-1 / C1+C2** — per-expert + router parity on the real layers 1–26
   (1664 experts = 26×64) reusing the Qwen decomposition harness (classic experts,
   k=6, `norm_topk_prob=false`).
2. **MLA-1 / C3** — MLA attention (already MLA-0-validated; add a real-weight
   layer check) + the dense-first layer.
3. **MLA-1 / C4** — fold the existing DeepSeek scale-topology cert.
4. **MLA-1 / C5** — active-path full forward (HF `DeepseekV2DecoderLayer` one
   layer at a time, YaRN + dense-first) vs Atenia `MoeRuntime` (opt-in) → L1→L2→L3.

*Provisioning only — model downloaded + validated read-only + inventoried; no
forward/generate/certification, no source change.*
