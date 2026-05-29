# HANDOFF — MOE-2: detect expert tensors and fail loud

Milestone: **MOE-2** (data-plane preparation for MoE). Experimental,
isolated. **No** router, top-k, dispatch, combine, sparse MoE, MoE family,
graph/runtime/CUDA/tier-planner/adapter-toolkit/CLI/generation change.
Predecessors: MOE-0 audit, MOE-1 substrate (`f36d873`).

## Objective

Make Atenia **recognise** MoE tensor names and **fail loud** on a MoE
checkpoint instead of silently skipping the expert weights. Before MOE-2,
an unknown tensor was added to `outcome.skipped` and loading continued — so
a Mixtral / Qwen-MoE / DeepSeek-MoE checkpoint would have loaded as a
broken, half-dense model (all experts dropped) with no error. MOE-2 closes
that footgun. It does **not** execute MoE.

## What MoE patterns are detected

Detection lives in `src/moe/detect.rs` (pure functions). Recognised
families:

- **Mixtral-style**: `…block_sparse_moe.gate.weight` (router),
  `…block_sparse_moe.experts.{E}.w1|w2|w3.weight` (expert gate/down/up).
- **Qwen-MoE / DeepSeek-MoE-style**: `…mlp.gate.weight` (router),
  `…mlp.experts.{E}.gate_proj|up_proj|down_proj.weight` (expert
  gate/up/down), `…shared_expert(s).*` (shared expert).

The **fail-loud trigger** is the presence of *expert* tensors — names
containing `block_sparse_moe.experts.`, `.experts.`, or `shared_expert`.
These substrings never occur in dense checkpoints, so detection is precise.

## How layer / expert / role are extracted

`classify_tensor_name(name) -> TensorNameInfo { role, layer_id, expert_id }`:

- `role`: a `TensorRole` enum — `AttentionQ/K/V/O`, `MlpGate/Up/Down`,
  `MoeRouter`, `MoeExpertGate/Up/Down`, `MoeSharedExpert`, `Unknown`.
- `layer_id`: digits after `.layers.`.
- `expert_id`: digits after `.experts.`.

Mixtral mapping: `w1 → MoeExpertGate`, `w3 → MoeExpertUp`, `w2 →
MoeExpertDown` (SwiGLU: `down(silu(gate)·up)`). Qwen/DeepSeek mapping:
`gate_proj/up_proj/down_proj` map to the same three roles. A dense
`mlp.gate_proj.weight` is **not** a router (distinguished from
`mlp.gate.weight` by the `gate_proj` substring).

`detect_moe(names) -> MoeDetection { is_moe, expert_tensor_count,
router_tensor_count, shared_expert_tensor_count, max_expert_id }` aggregates
across a checkpoint and yields `implied_expert_count() = max_expert_id + 1`.

## How fail-loud works

`LoaderError` gained one variant: `MoeUnsupported(String)`. Both
safetensors shard-load entry points in `weight_mapper.rs`
(`load_one_shard_into_with_residency_plan` and `load_one_shard_into`) run a
guard **before** the per-tensor loop:

```rust
let moe = crate::moe::detect_moe(reader.iter().map(|e| e.name));
if moe.is_moe {
    return Err(LoaderError::MoeUnsupported(crate::moe::unsupported_message(&moe)));
}
```

The message is explicit and actionable:

> *MoE checkpoint detected, but MoE execution is not implemented yet
> (experts=8, expert_tensors=…, router_tensors=…, shared_expert_tensors=…).
> Atenia can load and run dense models only; loading this checkpoint as
> dense would silently drop the expert weights and produce a broken model.*

`src/v17/inference/infer.rs` got a **forced exhaustiveness arm** (the only
change outside loader/moe): the new variant is routed through the existing
`InferenceError::LoadFailed` — no behaviour change, just enum
exhaustiveness after extending `LoaderError`.

## Dense models that stay intact (verified)

The guard keys solely on expert substrings, so **all dense families are
unaffected**: Llama, Qwen 2/3 (dense), Mistral, Phi-3, Gemma 2/3, Falcon3,
SmolLM2, and crucially **DeepSeek-R1-Distill** (dense Llama/Qwen
derivatives — standard `mlp.*_proj` names, no experts). Unit tests assert
dense + DeepSeek-distill + Qwen-dense names do not trigger detection, and
the existing `m4_7_6_b_f16_decode_validation_test` (real dense load through
`load_into`) still passes. The pre-existing silent-skip behaviour for
genuinely-unknown *dense-extra* tensors (not expert tensors) is preserved.

## Scope notes / limitations

- **Safetensors path only.** The guard is wired into the safetensors shard
  loaders (where HF Mixtral/Qwen-MoE checkpoints live). The GGUF path uses
  different stacked-expert tensor names (`blk.N.ffn_*_exps`); GGUF MoE
  detection is deliberately **out of scope for MOE-2** (would risk
  false-positives and needs its own naming work) — noted for a future
  milestone.
- **Detection ≠ execution.** Nothing runs experts; this is pure
  recognition + refusal.
- **No adapter-toolkit / config change.** Expert-count and routing config
  parsing is future work (MOE-3+).

## How this prepares MOE-3

MOE-3 (dense MoE graph, certified on the synthetic fixture) can reuse
`TensorRole` + `classify_tensor_name` to drive a MoE weight mapping, and
will flip the fail-loud guard into an actual load path once the graph can
execute router + experts + combine.

## Tests

- `src/moe/detect.rs` — 10 unit tests: Mixtral/Qwen expert + router
  detection, layer/expert id extraction, dense names map normally,
  DeepSeek-distill / Qwen-dense do NOT trigger, shared-expert triggers,
  full-checkpoint detection, clear unsupported message.
- `src/moe/fixture.rs` — 11 MOE-1 tests (unchanged).
- `tests/moe_loader_failloud_test.rs` — 3 integration tests on a real
  `SafetensorsReader` built from synthetic buffers (Mixtral detected,
  Qwen-MoE+shared detected, dense not detected).

Local validation: `cargo test --lib --release` → **648 passed / 0 failed /
1 ignored** (was 638). Dense load integration test green.

## Files modified

* `src/moe/detect.rs` — new (detection + roles + 10 tests).
* `src/moe/mod.rs` — re-exports.
* `src/v17/loader/loader_errors.rs` — `MoeUnsupported` variant + Display.
* `src/v17/loader/weight_mapper.rs` — fail-loud guard in both safetensors
  shard loaders.
* `src/v17/inference/infer.rs` — forced exhaustiveness arm (no behaviour
  change).
* `tests/moe_loader_failloud_test.rs` — new integration test.
* `docs/HANDOFF_MOE_2.md` — this file.

No graph, runtime, AMG, CUDA, tier-planner, adapter-toolkit, CLI, or
generation changes.
