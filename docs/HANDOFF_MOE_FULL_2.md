# HANDOFF — MOE-FULL-2: MoE config fields (parse-only)

Milestone: **MOE-FULL-2** (parse + normalize the MoE-specific fields of a
`config.json` into a stable struct). **Parse-only and inert** — no productive
path consumes it, no dense behaviour changes, the loader's MoE fail-loud guard
is unchanged. No runtime, graph, generation, loader, CUDA, CLI, or Adapter
Toolkit changes. Predecessor: MOE-FULL-1 (`fa1124a`).

## What was added

A new, **fully decoupled** module `src/nn/llama/moe_config.rs` exposing
`MoeConfig`. It is parsed independently from `LlamaConfig` — the dense config
parser (`src/nn/llama/config.rs`) is **not touched**, so dense models parse and
behave exactly as before.

```rust
pub struct MoeConfig {
    pub num_experts: Option<usize>,
    pub experts_per_token: Option<usize>,
    pub num_shared_experts: Option<usize>,
    pub has_shared_experts: bool,
    pub norm_topk_prob: Option<bool>,
    pub router_aux_loss_coef: Option<f32>,
    pub expert_intermediate_size: Option<usize>,
    pub shared_expert_intermediate_size: Option<usize>,
}
```

Entry points: `MoeConfig::from_json_str(&str)` and `MoeConfig::from_value(&serde_json::Value)`.
Helpers: `is_moe()`, `experts_per_token_or(default)` (clamped to `[1,
num_experts]`), `renormalize_topk_or(default)`.

## Fields parsed + alias normalization

| Normalized field | Source keys (first match wins) |
|---|---|
| `num_experts` | `num_experts`, `num_local_experts`, `n_routed_experts` |
| `experts_per_token` | `num_experts_per_tok`, `num_experts_per_token` |
| `num_shared_experts` | `n_shared_experts` |
| `shared_expert_intermediate_size` | `shared_expert_intermediate_size` |
| `expert_intermediate_size` | `moe_intermediate_size`, `expert_intermediate_size` |
| `norm_topk_prob` | `norm_topk_prob` |
| `router_aux_loss_coef` | `router_aux_loss_coef` |

`has_shared_experts` is derived: true if `n_shared_experts > 0` (DeepSeek count
style) OR a `shared_expert_intermediate_size` is present (Qwen-MoE size style).

`is_moe()` is true iff a routed-expert count was found — a dense config yields
an all-empty `MoeConfig` equal to `MoeConfig::default()`.

## Families covered

| Family | num_experts source | shared experts | norm_topk_prob | covered |
|---|---|---|---|---|
| Mixtral | `num_local_experts` | none | absent (block renorms by convention) | ✅ |
| Qwen2-MoE | `num_experts` | `shared_expert_intermediate_size` | `false` | ✅ |
| Qwen3-MoE | `num_experts` | none | `true` | ✅ |
| DeepSeek-MoE | `n_routed_experts` | `n_shared_experts` | `false` | ✅ (parse-only; not yet certified) |
| Dense (Qwen/Mistral/…) | — | — | — | ✅ → `is_moe() == false` |

DeepSeek field names are parsed for forward-compatibility, but no DeepSeek MoE
execution/convention is implied (still uncertified).

## What is NOT activated

- No productive path reads `MoeConfig`. It is exposed metadata only.
- No MoE execution, no MoE weight loading, no Mixtral/Qwen-MoE adapter.
- The loader still **fails loud** on MoE checkpoints
  (`LoaderError::MoeUnsupported`), unchanged.
- `LlamaConfig` and all dense parsing/behaviour are untouched.

## How this prepares MOE-FULL-3

MOE-FULL-3 (Mixtral adapter + tensor spec, gated load-only) can read
`MoeConfig` to know the expert count / top-k / shared-expert presence /
renormalization without re-parsing config, and cross-check it against the
`MoeWeightMap` tensor topology from `src/moe/`. It stays behind an opt-in flag;
the default loader keeps failing loud until MOE-FULL-6.

## Tests

`src/nn/llama/moe_config.rs` — 12 unit tests: dense stays non-MoE (Qwen +
Mistral), Mixtral/Qwen2-MoE/Qwen3-MoE/DeepSeek detection, experts-per-token +
num-experts + expert-intermediate-size alias normalization, `norm_topk_prob`
parsing, shared-expert fields, missing fields safe, top-k clamp, invalid JSON
errors.

Local validation: `cargo test --lib --release -- --test-threads=1` →
**739 passed / 0 failed / 1 ignored** (was 727; +12). MoE config suite: 12/12.

## Files modified

* `src/nn/llama/moe_config.rs` — new (parse-only `MoeConfig` + 12 tests).
* `src/nn/llama/mod.rs` — `pub mod moe_config;`.
* `docs/HANDOFF_MOE_FULL_2.md` — this file.
* `docs/MOE_FULL_PATH_AUDIT.md` — progress note.

No runtime, graph, generation, loader load-path, CUDA, CLI, or Adapter Toolkit
changes. Fail-loud preserved. Dense behaviour unchanged.
