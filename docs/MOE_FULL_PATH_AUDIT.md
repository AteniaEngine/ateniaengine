# MOE-FULL-1 — Full Transformer Path Audit

**Audit only. No code, no behaviour change.** This document maps what it would
take for a MoE family to run the *exact same full path* a dense family runs
today (tokens → embeddings → attention → norm → MLP/MoE → residual → lm_head →
generation), identifies the real gaps, and proposes a minimal roadmap.

Predecessor state: MOE-0..19 + QWEN-MOE-CERT-1 + MIXTRAL-CERT-1 (`eaab399`).
The MoE block is numerically certified (~1e-10 vs HuggingFace) on tiny real
Qwen-MoE and Mixtral checkpoints, but runs only in an isolated experimental
path. Productive loader still fails loud on MoE.

---

## 1. Inventory

### 1a. Dense transformer path (productive, graph-based / AMG)

The dense decoder path is **graph-based** (`src/amg/`), built once then executed:

| Stage | File | Key symbol |
|---|---|---|
| Pipeline / load orchestration | `src/nn/llama/pipeline.rs` | `GenerationPipeline::from_model_dir_with_options()` |
| Config parsing | `src/nn/llama/config.rs` | `LlamaConfig::from_json_file()` |
| Tokenizer | `src/tokenizer/mod.rs` | `AteniaTokenizer::{encode,decode,apply_chat_template}` |
| Family adapters (v1) | `src/model_adapters/mod.rs` | `AteniaModelAdapter` trait; `ModelFamily` enum |
| Family adapters (v2, declarative) | `src/adapter_toolkit/{dsl,spec,registry}.rs` | `AdapterDsl`, `ResolvedAdapterSpec`, `AdapterRegistry` |
| Tensor name→role map | `src/model_adapters/tensor_spec.rs` | `FamilyTensorSpec`, `NameTable`, `TransformRule` |
| Weight load | `src/v17/loader/weight_mapper.rs` | `WeightMapper::load_into()` |
| Weight store | `src/amg/weight_store.rs` | `WeightStore` (Arc-shared CPU/BF16) |
| **Per-layer block** | `src/nn/llama/builder.rs` | `build_transformer_block_llama()` (L91-337) |
| **Per-layer block (store-backed)** | `src/nn/llama/builder_shared.rs` | `build_block_shared()` (L188+) |
| **Dense MLP (SwiGLU)** | `src/nn/llama/builder.rs` | L301-336 inside the block |
| Attention + RoPE + KV cache | `src/nn/llama/builder.rs` | L121-288 inside the block |
| RMSNorm + residuals | `src/nn/llama/builder.rs` | `gb.rms_norm` / `gb.add` (L110-336) |
| Embeddings + lm_head | `src/nn/llama/builder.rs` | L368-451 (`embed_tokens`, tied/untied head) |
| Generation loop | `src/nn/llama/generator.rs` | `generate_greedy()` (L195); prefill+decode, KV harvest |
| Graph execute | `src/amg/graph.rs` | `Graph::execute()` / `execute_inference()` |

### 1b. MoE components (experimental, imperative / `src/moe/`)

| Component | File | Form |
|---|---|---|
| Detection + fail-loud | `src/moe/detect.rs` | pure name classifier; drives loader guard |
| Metadata plane | `src/moe/data_plane.rs` | `MoeWeightMap` from `(name, shape)` |
| Tensor binding (classic+packed) | `src/moe/binding.rs` | real bytes → `MoeDenseExpert` |
| Dense oracle / SwiGLU expert | `src/moe/dense.rs` | imperative `Vec<f32>` |
| Sparse forward (top-k, renorm flag) | `src/moe/sparse.rs` | imperative |
| Real layer (router+experts+shared) | `src/moe/layer.rs` | `RealMoeLayer::forward_auto` |
| Multi-layer stack | `src/moe/stack.rs` | `RealMoeStack` (MoE layers only) |
| Convention auto-select | `src/moe/convention.rs` | from `shared_expert_gate` signal |
| Validation / smoke / numerical | `src/moe/{validation,smoke,numerical}.rs` | harness + metrics |
| Graph ops (fused/primitive/dyn/cond) | `src/moe/graph_op.rs` + `src/amg/nodes.rs` | `NodeType::Moe*` over a process-global registry |

### 1c. What is connected vs isolated

- **Connected (dense):** loader → adapter → graph builder → graph execute →
  generator → tokenizer. Fully wired, certified, shipping.
- **Isolated (MoE):** everything in `src/moe/` is reachable only from tests /
  explicit callers. It consumes a `SafetensorsReader` directly at the
  `reader.iter()` boundary, never through the productive load path.
- **The only productive MoE touch point** is the fail-loud guard
  (`src/v17/loader/weight_mapper.rs` → `LoaderError::MoeUnsupported`). The
  adapter/config/runtime layers have **zero** MoE awareness (no `num_experts`,
  `block_sparse_moe`, `router`, `shared_expert`, `norm_topk_prob` anywhere in
  `src/adapter_toolkit/`, `src/config/`, `src/model_adapters/`).

---

## 2. Dense vs MoE path comparison

```
DENSE (productive, graph):
  tokens
   → tokenizer.encode
   → GraphBuilder: index_select(embed_tokens)
   → per layer × N:  RMSNorm → attn(q,k,v,o; RoPE; KV cache) → +resid
                     → RMSNorm → SwiGLU(gate,up,down) → +resid     ← (MLP block)
   → final RMSNorm → lm_head matmul
   → Graph::execute → logits
   → greedy/sample → next token → loop (prefill + decode w/ KV cache)
   → tokenizer.decode

MoE (experimental, imperative, TODAY):
  checkpoint (reader)
   → MoeWeightMap (metadata)
   → RealMoeLayer per layer (router + experts + shared)
   → RealMoeStack: x → MoeLayer0 → MoeLayer1 → … → x'           ← (MLP blocks ONLY)
   → ValidationReport (finite check) / numerical metrics vs HF
```

**Divergences:**
1. **Substrate.** Dense = AMG graph nodes (`GraphBuilder`/`Graph::execute`).
   MoE = imperative `Vec<f32>` math. The MoE block is *not* expressed as graph
   nodes on the productive path (the `NodeType::Moe*` ops from MOE-5..8 exist
   but use a separate process-global registry, not the scheduler/WeightStore).
2. **Scope.** Dense path = whole model (embeddings, attention, norms, lm_head,
   KV cache, generation). MoE path = the FFN substitution only; no attention,
   norms, embeddings, lm_head, KV cache, or token loop around it.
3. **Config source.** Dense topology comes from `LlamaConfig`. MoE topology is
   inferred from tensor-name metadata (MOE-13/18), not parsed config.
4. **Memory.** Dense uses tiered `WeightStore` (VRAM/RAM/Disk). MoE harness
   materialises **all** experts in f32 in RAM — fine for tiny, infeasible at
   14B–47B.
5. **Loading.** Dense goes through adapter + `WeightMapper`. MoE bypasses both
   (reads `SafetensorsReader` directly) and the productive loader actively
   *refuses* MoE checkpoints.

---

## 3. Gap analysis

| Area | Dense status | MoE status | Gap to full path |
|---|---|---|---|
| tokenizer | ✅ productive | ✅ reusable as-is | none — tokenizer is family-agnostic |
| embeddings | ✅ graph node | ❌ not in MoE path | small — reuse dense `index_select(embed_tokens)` |
| config parsing | ✅ `LlamaConfig` | ❌ name-heuristic only | medium — add MoE fields (`num_experts`, `num_experts_per_tok`/`num_local_experts`, shared-expert, `norm_topk_prob`) |
| metadata | ✅ tensor_spec | ✅ `MoeWeightMap` | small — bridge `MoeWeightMap` ↔ adapter tensor map |
| adapter | ✅ 7 dense families | ❌ no MoE family | **large** — new MoE family adapter(s) + tensor spec for expert/router/shared/packed names |
| tensor binding | ✅ `WeightMapper` | ✅ `binding.rs` (classic+packed) | medium — make binding feed `WeightStore` instead of standalone `Vec<f32>` |
| attention | ✅ graph | ✅ reusable as-is | none — MoE families use standard GQA attention |
| norms | ✅ graph RMSNorm | ✅ reusable as-is | none |
| residuals | ✅ graph add | ✅ reusable as-is | none |
| MoE layers | ⚠️ dense MLP only | ✅ certified block (imperative) | **large** — express MoE block as graph nodes in the per-layer builder, OR call the imperative block from a graph custom-op |
| lm_head | ✅ tied/untied | ❌ not in MoE path | none — reuse dense |
| logits | ✅ graph output | ❌ not in MoE path | none — reuse dense |
| generation | ✅ prefill+decode+KV | ❌ not in MoE path | none — reuse dense once the block is graph-resident |
| GGUF | ✅ dense GGUF | ❌ no MoE GGUF | out of scope (→ MOE-GGUF-1) |
| safetensors | ✅ productive | ✅ via reader | small — route MoE experts through productive loader behind opt-in |

**The gaps are concentrated in 4 areas:** adapter (MoE family), config (MoE
fields), MoE-block-as-graph (substrate bridge), and loader (opt-in + fail-loud
lift). Everything around the FFN (attention, norms, residuals, embeddings,
lm_head, generation, tokenizer) is **already done and reusable**.

---

## 4. Reusability analysis

Estimated share of the dense full-path stack reusable for MoE, **unchanged**:

- **Reusable as-is (~70%):** tokenizer, embeddings, attention (GQA + RoPE +
  KV cache), RMSNorm, residual wiring, final norm, lm_head (tied/untied),
  logits, the generation loop (prefill + decode + KV harvest), graph executor,
  WeightStore tiering, config validation scaffold.
- **Needs adaptation (~20%):** config parsing (add MoE fields), the per-layer
  block builder (swap dense SwiGLU for a MoE block at the same point),
  adapter/tensor-spec (recognise expert/router/shared/packed names), loader
  (carry expert tensors instead of refusing them).
- **Inevitably new (~10%):** a MoE family adapter; the MoE-block-as-graph
  bridge (either new `GraphBuilder` helpers that emit router+experts+combine,
  or a graph custom-op wrapping the certified imperative `RealMoeLayer`);
  MoE-aware residency/memory strategy for large checkpoints.

The certified math (`src/moe/`) is the asset that does **not** need to be
re-derived — it is the oracle the graph version must match.

---

## 5. Target families

| Family | Experts | Shared | Convention | Difficulty | Notes |
|---|---|---|---|---|---|
| **Mixtral** | packed or classic `w1/w3/w2` | none | Atenia (renorm) | **Easiest** | no shared expert, no shared-gate, standard GQA attention; the MoE block is the *only* delta vs a dense Mistral, which Atenia already runs |
| Qwen-MoE (1.5/2/3) | classic + packed | yes (gated) on 1.5/2; none on 3 | HuggingFaceQwen (no renorm) + sigmoid shared gate | Medium | shared-expert + sigmoid gate add wiring; Qwen3 router name `mlp.router`; all already handled in `src/moe/` |
| DeepSeek-MoE | classic | yes (multiple shared) | not yet modelled | **Hardest** | multiple shared experts, fine-grained experts, possible aux-loss-free bias routing; convention not yet implemented or certified in `src/moe/` |

**Recommendation: Mixtral as the first full-path family.** Atenia already runs
dense Mistral end-to-end; a Mixtral decoder layer is *identical* to a Mistral
layer except the FFN is the (already-certified) MoE block. So the first full
path isolates exactly one new variable — the MoE block in the graph — against
an otherwise proven decoder.

---

## 6. Proposed roadmap (small, verifiable milestones)

No massive rewrites. Each step is independently testable and keeps fail-loud
active until the very end.

- **MOE-FULL-2 — MoE config fields (parse-only). ✅ DONE** (see
  `docs/HANDOFF_MOE_FULL_2.md`). Implemented as a decoupled sidecar
  `src/nn/llama/moe_config.rs::MoeConfig` (NOT folded into `LlamaConfig`, so
  dense parsing is untouched). Normalizes `num_experts`/`num_local_experts`/
  `n_routed_experts`, `num_experts_per_tok(en)`, shared-expert presence,
  `norm_topk_prob`, expert FFN sizes. 11 unit tests; Mixtral / Qwen2-MoE /
  Qwen3-MoE / DeepSeek configs detected, dense stays non-MoE. Inert: no
  productive path consumes it; fail-loud unchanged.

- **MOE-FULL-3 — Mixtral adapter + tensor spec (load-only). ✅ DONE** (see
  `docs/HANDOFF_MOE_FULL_3.md`). Implemented as a metadata-only adapter in the
  experimental sandbox `src/moe/mixtral_adapter.rs` (NOT the productive
  `src/model_adapters/` registry, so it cannot enable execution and fail-loud
  stays active). `MixtralAdapter::detect_family` + `recognize` build
  `MixtralMetadata` (layout packed/classic, tensor spec, expert count) from a
  `(name, shape)` listing, reusing `detect` / `data_plane` / `binding` /
  `MoeConfig`. 6 unit tests; both real Mixtral layouts recognized, dense
  unaffected, missing-router errors loud. No weight bytes loaded, no graph.
  (Note: deferred the `WeightStore`-load step to MOE-FULL-4/5; this milestone
  is recognition + tensor spec only, which is the load-only contract.)

- **MOE-FULL-4 — MoE block as a graph op. ✅ DONE** (see
  `docs/HANDOFF_MOE_FULL_4.md`). Added `NodeType::MoeRealLayerReference
  { layer_id }` — a single AMG node wrapping the certified imperative
  `RealMoeLayer` (forward_auto) via a process-global registry in
  `src/moe/graph_op.rs`. Validated `input → MoeRealLayerReference → output` in a
  real `Graph` against `RealMoeLayer::forward_auto` (<1e-5) and the MOE-16 HF
  reference (argmax-match, <0.5), using the committed real Mixtral layer-0
  fixture. 2 unit + 7 integration tests; lib suite 746 passed / 1 ignored.
  Substrate bridge done; no full model, no fail-loud lift.

- **MOE-FULL-5 — One decoder layer, MoE composition. ✅ DONE** (see
  `docs/HANDOFF_MOE_FULL_5.md`). `src/moe/decoder_layer.rs` composes norm +
  self-attention + residual + `MoeRealLayerReference` + residual in the AMG
  graph from existing primitives (no new op), with the real Mixtral layer-0
  fixture supplying the MoE sub-block. Validated vs an imperative reference
  within 1e-5 (4 unit + 6 integration tests; lib 750). **Single token,
  single-head** (softmax over one score = 1.0, so the score scale is moot) —
  it proves attention+residual+MoE *compose*; multi-token attention (RoPE/GQA/
  causal mask/KV cache) + HF single-layer logit comparison are MOE-FULL-6.

- **MOE-FULL-6 — Tiny full MoE transformer forward (no generation). ✅ DONE**
  (see `docs/HANDOFF_MOE_FULL_6.md`). `src/moe/full_forward.rs` composes
  embeddings + 2 decoder layers (real attention: RoPE + causal mask +
  multi-token; certified MoE block position-wise) + final norm + lm_head into
  logits, from existing AMG primitives (no new op). Validated vs an offline HF
  `MixtralForCausalLM` f64 reference: **max_abs_diff 7.451e-08**, per-position
  argmax matches (real tiny fixture ~251 KB). Honest simplification: **MHA, no
  GQA** (fixture `n_kv == n_heads`); `1/√d` absorbed by pre-scaling `w_q`.
  3 unit + 7 integration tests; lib 753.

- **MOE-FULL-7 — Generation: prefill + KV cache + incremental decode. ✅ DONE**
  (see `docs/HANDOFF_MOE_FULL_7.md`). `src/moe/generate.rs` adds a prefill graph
  that seeds a per-layer KV cache and an incremental decode graph (`seq=1`,
  `rope_with_offset(cached_len)`, `concat(cache, new)` over the seq axis, single-
  query attention with no mask), driven by a greedy loop that harvests
  K_full/V_full and re-injects them via `Graph::overwrite_parameter`. No new
  graph op. Locked two ways: **prefill+decode == full recompute** (the
  MOE-FULL-6 graph is the oracle) and an offline HF f64 **greedy** reference
  (`fixtures/moe/full_mixtral_gen.json`): generated ids match exactly,
  per-step logits **max_abs_diff 4.470e-08**. 3 unit + 5 integration tests.

- **MOE-FULL-8 — Large-MoE expert residency. ✅ DONE** (see
  `docs/HANDOFF_MOE_FULL_8.md`). `src/moe/residency.rs` places expert weights
  in Atenia's real residency tiers (`SharedParam` F32/RAM or `Disk`/NVMe via
  `disk_tier`) and resolves only the router-selected top-k experts per token,
  reusing the certified `route` + `top_k_routing_with` + SwiGLU + combine. No
  WeightStore/loader/runtime change (it *consumes* the infra). Output is bit-
  identical to `RealMoeLayer::forward_auto` (RAM and NVMe tiers). Evidence: a
  128-expert layer on NVMe holds ~router-only bytes in host RAM (**385× saving**
  vs full materialisation), only top-k materialised per forward. 6 unit + 4
  integration tests.

- **MOE-FULL-9 — GQA + productive integration (remaining).** GQA (load-time
  K/V tile or graph repeat-kv), a Mixtral family adapter on the productive load
  path + an explicit fail-loud lift, config-driven topology, VRAM expert tier.
  Only after correctness is proven.

DeepSeek-MoE and MoE-GGUF are explicitly **after** this line.

---

## 7. Risks

- **Substrate impedance (highest).** The dense path is graph-based; the
  certified MoE block is imperative. MOE-FULL-4 (graph custom-op vs native
  graph nodes) is the pivotal design choice — wrapping the imperative block is
  lower-risk and reuses the certified code; native nodes are more work but
  better for tiering/perf later. Recommend wrapping first.
- **Fail-loud regressions.** Every step until MOE-FULL-6 must keep the default
  loader refusing MoE; the opt-in flag is the only door. Risk: an adapter that
  silently accepts MoE on the default path → must be guarded + tested.
- **Memory blowup.** Materialising all experts in f32 (current harness) will
  OOM on real models; MOE-FULL-7 is mandatory before any non-tiny claim.
- **Config heterogeneity.** `num_experts` vs `num_local_experts`, router name
  `mlp.gate` vs `mlp.router`, shared-expert gating — already mapped in
  `src/moe/`, but the config/adapter layer must mirror it exactly or drift.
- **Certification at scale.** Full-model F64 reference is infeasible for large
  MoE (ADR-004); needs the MOE-1 partial/sub-reference methodology. Not a
  blocker for tiny Mixtral, but is for any production claim.

---

## 8. Effort estimate

Story points (relative, not calendar — consistent with `MILESTONES.md`):

| Milestone | Effort | Confidence |
|---|---|---|
| MOE-FULL-2 (config fields) | S | high |
| MOE-FULL-3 (adapter + tensor spec, gated load) | M | high |
| MOE-FULL-4 (MoE block as graph op) | M–L | medium (substrate choice) |
| MOE-FULL-5 (one full decoder layer) | M | medium |
| MOE-FULL-6 (full tiny Mixtral + generation) | L | medium |
| MOE-FULL-7 (large-MoE residency) | L | low (depends on tiering details) |

**Headline:** ~70% of the full-path stack is already reusable; the remaining
work is concentrated and incremental, not a rewrite. The single biggest design
decision is MOE-FULL-4 (how the certified MoE block enters the graph). Starting
with **Mixtral** (a dense Mistral + one MoE FFN) isolates that decision against
an otherwise-proven decoder, making it the lowest-risk first full path.

---

## Confirmation

- **Audit only.** No `src/` changes, no behaviour change, no fail-loud lift,
  no CI change, no CUDA/generation/Adapter-Toolkit modification. Docs-only.
- Fail-loud remains active; no MoE family is production-supported.
