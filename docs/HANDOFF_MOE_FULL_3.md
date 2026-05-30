# HANDOFF — MOE-FULL-3: Mixtral adapter + tensor specification (load-only)

Milestone: **MOE-FULL-3** (teach Atenia to *recognize* a Mixtral checkpoint at
the tensor level and describe its MoE tensor topology). **Load-only / metadata
only.** No model execution, no runtime, no graph, no generation, no CUDA, no
CLI. The MOE-2 loader fail-loud guard is **unchanged**. Predecessor:
MOE-FULL-2 (`dbc4a47`).

## Scope decision

The Mixtral adapter lives in **`src/moe/mixtral_adapter.rs`** (the experimental
sandbox), **not** in the productive `src/model_adapters/` registry. This keeps
it metadata-only and guarantees it cannot enable execution or alter productive
behaviour — the productive loader still refuses MoE. Wiring a productive
adapter is a later, explicit step (MOE-FULL-5/6).

## Tensor inventory (verified on real tiny checkpoints, not assumed)

Two real on-disk Mixtral layouts (both validated in MIXTRAL-CERT-1):

```text
PACKED (hf-internal-testing/tiny-random-MixtralForCausalLM):
  model.layers.{L}.mlp.gate.weight                router  [E, d_model]
  model.layers.{L}.mlp.experts.gate_up_proj       [E, 2*d_ff, d_model]
  model.layers.{L}.mlp.experts.down_proj          [E, d_model, d_ff]

CLASSIC (TitanML/tiny-mixtral, original Mixtral layout):
  model.layers.{L}.block_sparse_moe.gate.weight            router [E, d_model]
  model.layers.{L}.block_sparse_moe.experts.{e}.w1.weight  gate   [d_ff, d_model]
  model.layers.{L}.block_sparse_moe.experts.{e}.w3.weight  up     [d_ff, d_model]
  model.layers.{L}.block_sparse_moe.experts.{e}.w2.weight  down   [d_model, d_ff]

Dense surround (reused, identical to a dense Mistral decoder):
  model.embed_tokens.weight, lm_head.weight, model.norm.weight,
  model.layers.{L}.input_layernorm.weight,
  model.layers.{L}.post_attention_layernorm.weight,
  model.layers.{L}.self_attn.{q,k,v,o}_proj.weight
```

Mixtral has **no shared expert** in either layout (confirmed on both real
checkpoints).

## Tensor specification

`MixtralExpertLayout { Packed, Classic }` + `MixtralTensorSpec` describe, per
layout: the router suffix, the expert suffixes (per-expert `w1/w3/w2` classic
or `gate_up_proj`/`down_proj` packed), and the dense suffixes reused unchanged.
This is pure metadata (`&'static str` suffixes) — no data, no execution.

## Adapter responsibilities (load-only)

`MixtralAdapter` (stateless):
- **`detect_family(tensors)`** — `true` iff the `(name, shape)` listing is MoE
  (via `detect::detect_moe`) AND a recognized Mixtral layout is present.
- **`recognize(tensors, &MoeConfig)`** — validates the inventory and builds
  `MixtralMetadata { layout, spec, weight_map, num_experts, num_moe_layers,
  has_shared_experts: false }`. Errors loud (load-only) on: not-MoE
  (`NotMoe`), MoE-but-not-Mixtral (`UnrecognizedLayout`), missing router
  (`MissingRouter`), missing experts (`MissingExperts`), bad packed shapes
  (`BadPackedShapes`).

It reuses existing pieces — `detect::classify_tensor_name`,
`data_plane::MoeWeightMap`, `binding::packed_dims`, and the MOE-FULL-2
`nn::llama::moe_config::MoeConfig`. It builds **no graph, no runtime, no
forward, loads no tensor bytes**.

## Limitations

- Recognition + metadata only. No weight bytes loaded; no execution.
- `recognize` accepts a `MoeConfig` for forward-compat (MOE-FULL-2) but does
  not yet cross-check/override the tensor-derived topology against it — that
  is a MOE-FULL-4/5 concern.
- Packed naming (`mlp.experts.gate_up_proj`) is shared with Qwen2/3-MoE; the
  adapter treats packed-without-classic as Mixtral packed because it is only
  *asked* about Mixtral by the caller. A family router that disambiguates
  Mixtral vs Qwen-MoE is future work.
- Not wired into the productive loader/adapter registry; fail-loud intact.

## How this prepares MOE-FULL-4

MOE-FULL-4 (MoE block as a graph op) can take a `MixtralMetadata` +
`SafetensorsReader` and: resolve real expert bytes (via `binding`) for the
recognized layout, build the certified `RealMoeLayer`, and wrap it as a single
AMG node — validated to match `src/moe/` within 1e-5 inside a one-layer graph.
The adapter gives MOE-FULL-4 a single, validated description of "where the
Mixtral router/experts are" so it never re-parses names.

## Tests

`src/moe/mixtral_adapter.rs` — 6 unit tests: `detect_mixtral_family`,
`validate_mixtral_tensor_spec`, `packed_experts_detected`,
`classic_experts_detected`, `missing_router_reports_error`,
`dense_models_unaffected`.

Local validation (real output, exit 0): `cargo test --lib --release
mixtral_adapter` → **6 passed**. Full suite `cargo test --lib --release --
--test-threads=1` → **744 passed / 0 failed / 1 ignored** (was 738; +6).

## Files modified

* `src/moe/mixtral_adapter.rs` — new (adapter + tensor spec + 6 tests).
* `src/moe/mod.rs` — `pub mod mixtral_adapter;` + re-exports.
* `docs/HANDOFF_MOE_FULL_3.md` — this file.
* `docs/MOE_FULL_PATH_AUDIT.md` — progress note.

No runtime, graph, generation, loader load-path, CUDA, CLI, or productive
adapter-registry changes. Fail-loud preserved.
