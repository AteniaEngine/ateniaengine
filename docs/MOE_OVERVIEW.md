# MoE Overview — Experimental Track (MOE-0 → MOE-18)

This document consolidates Atenia's Mixture-of-Experts (MoE) work and **closes
the experimental track** (MOE-19). It is the single entry point for "what MoE
is in Atenia today, what is proven, and what blocks production".

> **One-sentence status.** Atenia has a complete, numerically-validated,
> **experimental, CPU-only, opt-in** MoE compute + data plane that runs and
> matches HuggingFace on tiny real checkpoints — but MoE is **not** wired into
> the productive loader/runtime, and the productive loader still **fails loud**
> on MoE checkpoints. No MoE family is production-supported.

## Architecture

All MoE code lives in `src/moe/` and is reachable only by tests / explicit
opt-in callers. The productive loader, runtime, graph scheduler, Adapter
Toolkit, CLI and generation paths are untouched.

```
                checkpoint (safetensors, real or fixture)
                          │  (name, shape) listing
                          ▼
   detect.rs ── classify tensor names ──► fail-loud guard (loader refuses MoE)
                          │
                          ▼
   data_plane.rs ── MoeWeightMap (router / experts / shared / packed) metadata
                          │
                          ▼
   binding.rs ── resolve real bytes ──► MoeDenseExpert
                 (classic per-expert  OR  packed/fused gate_up+down)
                          │
                          ▼
   layer.rs ── RealMoeLayer (router + routed experts + optional shared)
                 forward / forward_with(convention) / forward_auto
                          │
                          ▼
   stack.rs ── RealMoeStack (sequential multi-layer composition)
                          │
                          ▼
   validation.rs / smoke.rs / numerical.rs
       metadata→config→stack→forward + metrics vs reference
```

Supporting modules: `dense.rs` (dense oracle + SwiGLU expert), `sparse.rs`
(top-k routing + sparse forward, renormalise flag), `fixture.rs` (MOE-1
certification substrate), `graph_op.rs` (fused/primitive/dynamic/conditional
graph ops over a process-global registry), `convention.rs` (auto convention
selection).

## Milestone index (MOE-0 → MOE-18)

| # | Milestone | Outcome |
|---|---|---|
| MOE-0 | Architecture audit | Data plane ~ready; compute plane was the blocker |
| MOE-1 | Certification substrate | `FixtureMoESpec`, F64-strategy model (`docs/MOE_CERTIFICATION_SUBSTRATE.md`) |
| MOE-2 | Detect + fail-loud | `detect_moe` + `LoaderError::MoeUnsupported` (loader refuses MoE) |
| MOE-3 | Dense oracle | All-experts reference forward (`MoeDenseLayer`) |
| MOE-4 | Sparse reference | top-k + renormalise, pinned to the dense oracle (1e-5) |
| MOE-5 | Fused graph op | `NodeType::MoeSparseReference` + layer registry |
| MOE-6 | Primitive graph ops | router softmax / top-k / combine nodes |
| MOE-7 | Dynamic dispatch | execute only selected experts |
| MOE-8 | Conditional subgraphs | per-expert conditional execution |
| MOE-9 | Real metadata plane | `MoeWeightMap` from real `(name, shape)` |
| MOE-10 | Real tensor binding | real bytes → `MoeDenseExpert`, real sparse forward |
| MOE-11 | Real layer assembly | router + experts + optional shared expert |
| MOE-12 | Multi-layer stack | sequential `RealMoeStack` |
| MOE-13 | Checkpoint validation harness | metadata→config→stack→forward→report |
| MOE-14 | Opt-in real local smoke | `#[ignore]` env-gated smoke against a local checkpoint |
| MOE-15 | Packed expert support | 3-D `gate_up_proj`/`down_proj` split into per-expert |
| MOE-16 | Numerical equivalence | f64 reference metrics (`max_abs_diff`/rmse/argmax) |
| MOE-17 | HF convention parity | `norm_topk_prob` + sigmoid-gated shared expert (opt-in) |
| MOE-18 | Automatic convention selection | resolve convention from `shared_expert_gate` signal |

Per-milestone detail: `docs/HANDOFF_MOE_2.md` … `docs/HANDOFF_MOE_18.md`.

## Real results

Three tiny **real** MoE checkpoints (downloaded locally, not committed; only
extracted layer-0 reference fixtures are committed) run end-to-end through the
experimental path:

| Checkpoint | Expert format | Shared expert | Smoke (MOE-14) |
|---|---|---|---|
| `katuni4ka/tiny-random-qwen1.5-moe` | classic per-expert | yes (+gate) | SMOKE PASS |
| `hf-internal-testing/tiny-random-Qwen2MoeForCausalLM` | packed/fused | yes (+gate) | SMOKE PASS (after MOE-15) |
| `hf-internal-testing/tiny-random-Qwen3MoeForCausalLM` | packed/fused | no | SMOKE PASS (after QWEN-MOE-CERT-1 router fix) |
| `hf-internal-testing/tiny-random-MixtralForCausalLM` | packed/fused | no | SMOKE PASS (after MOE-15) |

**Qwen-MoE family certification** (`docs/HANDOFF_QWEN_MOE_CERT_1.md`,
QWEN-MOE-CERT-1): Qwen1.5-MoE, Qwen2-MoE and Qwen3-MoE tiny checkpoints are
**partially certified (experimental)** — MoE-block numerical parity with
HuggingFace ~1e-10 across classic + packed experts, shared / no-shared, both
`norm_topk_prob` modes, both router namings (`mlp.gate` and Qwen3's
`mlp.router`). Not product-certified; fail-loud still active.

**Mixtral family certification** (`docs/HANDOFF_MIXTRAL_CERT_1.md`,
MIXTRAL-CERT-1): two real Mixtral checkpoints **partially certified
(experimental)** under the Atenia convention — `tiny-random-Mixtral` (packed,
4 experts, 1.164e-10, committed) and `TitanML/tiny-mixtral` (classic
`block_sparse_moe.w1/w3/w2`, 8 experts, d_model 1024, 1.49e-8, local-only;
~352 MB fixture too large to commit). Both on-disk Mixtral layouts validated
end-to-end on real data; no `src/` changes were needed. Not product-certified
(no Mixtral-8x7B); fail-loud still active. No MoE GGUF support.

"SMOKE PASS" = the experimental path discovered the shards, built a
`MoeWeightMap`, assembled a stack, and ran a finite forward. It does **not**
mean full-model support.

## Numerical results (MOE-16/17/18)

Layer-0 MoE block, Atenia vs an f64 reference, argmax matched in all cases:

| Model | vs Atenia-op f64 (primary) | vs HuggingFace f64 (parity) |
|---|---|---|
| qwen15_moe | 5.8e-11 | 2.9e-11 (HF convention) |
| qwen2_moe | 8.7e-11 | 5.8e-11 (HF convention) |
| mixtral | 1.2e-10 | 1.2e-10 (Atenia convention) |

`forward_auto` (MOE-18) reproduces these by resolving the convention from the
`shared_expert_gate` metadata signal — no caller input, no math change.

## Supported formats

- **Classic per-expert**: `…experts.{E}.{gate,up,down}_proj.weight` (Qwen-MoE)
  and `…block_sparse_moe.experts.{E}.{w1,w3,w2}.weight` (Mixtral classic).
- **Packed/fused**: `…experts.gate_up_proj` (3-D `[ne, 2*d_ff, d_model]`, gate
  first half / up second) + `…experts.down_proj` (3-D `[ne, d_model, d_ff]`).

## Supported conventions

- **Atenia / Mixtral**: renormalise top-k weights to sum 1; shared expert (if
  any) added ungated. Default.
- **HuggingFace Qwen**: no renormalisation (`norm_topk_prob = false`); shared
  expert scaled by `sigmoid(shared_expert_gate · x)`. Opt-in / auto-resolved.

## Limits (what is explicitly NOT done)

- **No production MoE support** for any family. The experimental path is the
  only way MoE executes.
- **Fail-loud still active**: the productive loader refuses every MoE
  checkpoint (`LoaderError::MoeUnsupported`); MOE-1..18 never lifted it.
- **No Adapter Toolkit integration**, **no CLI integration**, **no
  generation** wiring.
- **No full transformer path**: only the MoE block is executed — no attention,
  norms, embeddings, lm_head, KV cache, or multi-token decode.
- **No config.json parsing** in the productive sense (convention is inferred
  from tensor-name metadata only).
- **Large-model residency (experimental mitigation, MOE-FULL-8)**: the legacy
  harness materialises all experts in f32. `src/moe/residency.rs` now places
  expert weights in Atenia's real residency tiers (`SharedParam` F32/RAM or
  `Disk`/NVMe) and resolves only the **router-selected top-k** experts per
  token. Demonstrated: a 128-expert layer on NVMe holds ~router-only bytes in
  RAM (385× saving vs full materialisation) with bit-identical output. Still
  experimental/CPU+NVMe/test-only; not on the productive load path.

## Production readiness

| Area | Status | Evidence | Blocker to production |
|---|---|---|---|
| Tensor detection | ✅ Done | `detect.rs`, `moe_loader_failloud_test.rs` | — |
| Data plane | ✅ Done | `data_plane.rs`, `moe_data_plane_test.rs` | — |
| Expert binding (classic + packed) | ✅ Done (experimental) | `binding.rs`, `moe_real_binding_test.rs`, `moe_packed_experts_test.rs` | not on productive load path |
| Sparse execution | ✅ Done (experimental) | `sparse.rs`, `moe_real_layer_test.rs` | not wired to runtime |
| Graph ops | ✅ Done (experimental) | `graph_op.rs`, `moe_graph_op_test.rs`, `moe_primitive_ops_test.rs` | uses process-global registry, not the scheduler |
| Conditional dispatch | ✅ Done (experimental) | `moe_dynamic_dispatch_test.rs`, `moe_conditional_expert_test.rs` | not wired to runtime |
| Packed experts | ✅ Done (experimental) | MOE-15, `moe_packed_experts_test.rs` | layout assumed (gate-first), not config-confirmed |
| HF parity | ✅ Validated | MOE-16/17, `moe_numerical_equivalence_test.rs`, `moe_hf_convention_test.rs` | tiny checkpoints only |
| Convention selection | ✅ Done (experimental) | MOE-18, `moe_auto_convention_test.rs` | name-heuristic, not config-validated |
| Full transformer forward | ✅ Done (experimental) | MOE-FULL-6, `moe_full_forward_test.rs` (7.451e-08 vs HF) | tiny fixture, MHA-no-GQA |
| Generation (prefill+KV cache+decode) | ✅ Done (experimental) | MOE-FULL-7, `moe_decode_generation_test.rs` (4.470e-08 vs HF greedy) | greedy-only, tiny fixture |
| Expert residency (RAM/NVMe tiers) | ✅ Done (experimental) | MOE-FULL-8, `residency.rs`, `moe_residency_test.rs` (385× saving, top-k only) | CPU+NVMe, not on productive load path |
| GQA (grouped-query attention) | ✅ Done (experimental) | MOE-FULL-9, `gqa.rs`, `moe_gqa_test.rs` (5.960e-08 vs HF, n_kv≠n_heads) | tiny fixture |
| Expert cache (LRU/prefetch/reuse) | ✅ Done (experimental) | MOE-FULL-9, `residency.rs::ExpertCache` | not on productive path |
| Family recognition (Mixtral/Qwen-MoE) | ✅ Done (metadata) | MOE-FULL-9, `family.rs`, `moe_family_loader_test.rs` | recognise + validate only, no load |
| Loader (family-aware fail-loud) | ⚠️ Fail-loud (prepared) | MOE-FULL-9, loader emits "Family: …, Productive support not enabled" | dense loader still refuses MoE (by design) |
| Controlled Mixtral runtime (opt-in) | ✅ Done (experimental, Mixtral-only) | MOE-FULL-10, `runtime.rs`, `moe_mixtral_runtime_test.rs` (load→generate→EOS behind `ATENIA_EXPERIMENTAL_MOE=1`) | Mixtral only; no Qwen-MoE/DeepSeek/VRAM/CLI |
| Adapter Toolkit | ❌ Not integrated | — | **BLOCKER**: ATK has no MoE family/tensor spec |
| Product loader | ❌ Fail-loud | `weight_mapper.rs` guard | **BLOCKER**: must lift fail-loud behind a validated, opt-in path |
| CLI | ❌ Not integrated | — | **BLOCKER**: no MoE entry point |
| Full transformer | ❌ Not implemented | only MoE block runs | **BLOCKER**: no attention/norms/embeddings/KV cache around the MoE layer |
| Certification (real full models) | ❌ Not done | tiny smoke only | **BLOCKER**: ADR-004 F64 reference infeasible for full MoE (`docs/MOE_CERTIFICATION_SUBSTRATE.md`) |

## Production blockers (exact list)

1. **Adapter Toolkit MoE spec** — family/tensor descriptors for Qwen-MoE /
   Mixtral / DeepSeek-MoE (config: experts, top-k, shared-expert, packed-vs-
   classic, `norm_topk_prob`).
2. **Loader opt-in / fail-loud lift** — a validated, gated path that lets the
   productive loader carry MoE weights instead of refusing them.
3. **Full transformer path** — attention + norms + embeddings + lm_head +
   KV cache around the MoE layer; multi-token decode.
4. **Config parsing** — real `config.json` topology to confirm/override the
   tensor-name convention heuristic.
5. **Large-model memory strategy** — the current harness materialises all
   experts in f32; real MoE (14B–47B+) needs streaming / tiering.
6. **Certification for real full-family models** — an ADR-004-compatible
   reference strategy for full MoE checkpoints (full F64 is infeasible at
   scale; needs the partial/sub-reference methodology from MOE-1).

## Minimal experimental → production plan

The experimental infrastructure is complete; the next phase is **certification
and integration, not more infrastructure**:

1. Define the Adapter Toolkit MoE family/tensor spec (blocker 1).
2. Wire the MoE block into the full transformer path behind an opt-in flag
   (blocker 3) and lift fail-loud only for that validated path (blocker 2).
3. Parse config to drive convention + topology (blocker 4).
4. Certify one real small-but-full MoE end-to-end with the MOE-1 partial-F64
   strategy (blocker 6), then add a memory/tiering strategy for larger ones
   (blocker 5).

Until those land, MoE remains **experimental and CPU-only**, and the
productive loader continues to fail loud.

## Full transformer path audit (MOE-FULL-1)

`docs/MOE_FULL_PATH_AUDIT.md` maps what it takes for a MoE family to run the
*exact same full path* a dense family runs today. Key findings: ~70% of the
dense full-path stack (tokenizer, embeddings, attention, norms, residuals,
lm_head, generation, KV cache, graph executor, WeightStore) is **reusable
as-is**; the gaps are concentrated in 4 areas — MoE family adapter, MoE config
fields, MoE-block-as-graph bridge, and loader opt-in/fail-loud lift. The dense
path is **graph-based** (`src/nn/llama/builder.rs`) while the certified MoE
block is **imperative** (`src/moe/`), so the pivotal design choice is how the
MoE block enters the graph. Recommended first full-path family: **Mixtral** (a
dense Mistral decoder + one MoE FFN). Proposed incremental roadmap:
MOE-FULL-2 (config) → 3 (adapter, gated load) → 4 (MoE graph op) → 5 (one
decoder layer) → 6 (full tiny Mixtral + generation) → 7 (large-MoE residency).
