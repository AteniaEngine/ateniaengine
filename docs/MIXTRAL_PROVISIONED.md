# Mixtral-8x7B-v0.1 — PROVISIONED (read-only validated)

**Status: provisioned + validated (read-only). Ready for MIXTRAL-CERT.** The real
Mixtral-8x7B-v0.1 weights are on disk, complete, and structurally validated. **No
forward, no generate, no certification done here.** Weights are git-ignored (not
committed).

## Source

- **Repo:** `mistralai/Mixtral-8x7B-v0.1` (HuggingFace), license **Apache-2.0**,
  gated acceptance done.
- **Tool:** `huggingface_hub.hf_hub_download`, **serial per-shard** (one file at a
  time, up to 5 retries each, verified after each), after the initial parallel
  `snapshot_download` was paused (see `MIXTRAL_DATA_PROVISION_PAUSED.md`).
- **Fetched:** the 19 `*.safetensors` shards + `model.safetensors.index.json` only.
  **Excluded:** `consolidated.*.pt` (~97 GB, redundant), `.bin` (none). Config +
  tokenizer were already present.

## Validation (read-only — no forward)

| Check | Result |
|---|---|
| Shards present | **19 / 19** |
| Each shard complete (file size == `8 + header_len + max data_offset`) | **PASS** (no truncation) |
| Dtype | **BF16** (all tensors) |
| `model.safetensors.index.json` parses | **PASS** |
| Index tensors | **995**, referencing **19** shards |
| Index → shard resolution | **all present, none missing** |
| Shard tensors not listed in index | **0** |
| Index `total_size` | **93,405,585,408 B = 86.99 GiB (93.4 GB)** |
| `.incomplete` partials | **0** |
| Download process alive | **no** |

**VALIDATION: PASS.**

## Technical inventory (from `config.json`)

| Field | Value |
|---|---|
| `model_type` / arch | `mixtral` / `MixtralForCausalLM` |
| `num_hidden_layers` | 32 |
| `hidden_size` | 4096 |
| `intermediate_size` (per expert FFN) | 14336 |
| `num_attention_heads` / `num_key_value_heads` | 32 / 8 (GQA 4:1) |
| `num_local_experts` | 8 |
| `num_experts_per_tok` (top-k) | 2 |
| `vocab_size` | 32000 |
| `rope_theta` / `max_position_embeddings` | 1,000,000 / 32768 |
| `torch_dtype` | bfloat16 |
| Shards / total size | 19 × safetensors / 86.99 GiB |
| Expert layout | **classic per-expert** `block_sparse_moe.experts.{e}.{w1,w3,w2}` |

## Local layout

```
models/Mixtral-8x7B-v0.1/
  config.json, generation_config.json, tokenizer.* (5 files)
  model-00001-of-00019.safetensors … model-00019-of-00019.safetensors   (19, BF16)
  model.safetensors.index.json
```
Git-ignored (verified via `git check-ignore`); only docs are committed.

## Readiness for MIXTRAL-CERT

**Ready.** Atenia's `MoeRuntime` already loads sharded classic Mixtral, and the
Mixtral topology is L0-certified (`mixtral_scale` 1.639e-7). The MoE machinery
(router softmax-top-k, classic experts, MLA-2 disk expert-tier, the one-layer-at-a-
time F64 C5 methodology) is reused from Qwen-MoE / DeepSeek-V2-Lite. Remaining for
MIXTRAL-CERT (per `MIXTRAL_CERT_ROADMAP.md` / `MIXTRAL_L3_FEASIBILITY.md`):

- **C1/C2 → L1:** per-expert + router-set decomposition harness (reuse Qwen/DeepSeek;
  Mixtral has **no shared expert**, **no qkv bias**, **renorm top-k**, `w1/w3/w2`).
- **C4 → L2:** already have the scale-topology cert.
- **C5 → L3:** a new **Mixtral** F64 reference driver (`MixtralDecoderLayer`,
  one layer at a time) + Atenia disk-tier real forward (RAM-feasible like DeepSeek).
- **L4** reserved/unreachable (global F64). Results would be `MoE-certified Ln`, never
  the dense ADR-004 `CERTIFIED`.

## Notes

- The earlier pause was a harness/measurement artifact, not a network or data fault
  (see `MIXTRAL_DATA_PROVISION_PAUSED.md`); the serial verified re-fetch completed
  cleanly and reused the 4 already-valid shards.
