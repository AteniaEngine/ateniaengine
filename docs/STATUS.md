# Status

An honest snapshot of what is **cabled to production signals** versus what is
still **scaffolding**. This is the document to read before depending on Atenia
for anything. For the history of how each piece landed, see
[MILESTONES.md](./MILESTONES.md); for direction, see
[ROADMAP.md](../ROADMAP.md).

**One-line summary.** Atenia Engine is **early research, single-author, active
development**. The execution-layer thesis is proven end-to-end on the dev
hardware (RTX 4070 Laptop, 8 GB VRAM, 32 GB RAM, NVMe); the public CLI surface
is stable; numeric certification is real and auditable. It is **not**
production-ready for unsupervised deployment, **not** multi-vendor, and several
execution paths are deliberately conservative or opt-in.

---

## Cabled to production signals

These are exercised end-to-end against real checkpoints (not synthetic) and
locked by regression tests.

- **Tier-aware placement.** A pure planner over
  `(metadata, free_ram, free_vram, kernel_dtype)` assigns every tensor to
  VRAM / RAM / Disk at load time. Llama 2 13B Chat runs end-to-end on
  8 GB VRAM + 32 GB RAM + NVMe with the transparency contract
  `argmax(clean) == argmax(forced 50 % LRU spill)` holding **bit-exactly** at
  13B scale.
- **Numeric certification (certified mode).** The default `certified` path
  (Path B + TF32) is gated on every code change by the four-model F64 fixture
  (TinyLlama, SmolLM2, Qwen 2.5 1.5B, Llama 3.2 1B), threshold
  `max_abs_diff < 0.5` per
  [ADR-004](./decisions/ADR-004-f64-reference-as-default.md). Per-checkpoint
  manifests live in [numcert/](./numcert/) and are reproducible offline.
- **Public CLI.** `atenia generate`, `atenia run`, `atenia probe` are stable
  across the M4.9 → M11 series and documented in [CLI.md](./CLI.md).
  `cargo install --path .` produces a working binary on Windows and Linux.
- **Real end-to-end (RUNTIME-REAL-1).** Re-confirmed on a real 2.2 GB
  TinyLlama-1.1B HF checkpoint, RTX 4070: loads (201 params, ~4.46 GiB
  resident, VRAM/RAM tiered) in ~13 s and greedily generates coherent,
  factually-correct text ("The capital of France is" → "Paris."), deterministic
  vs the committed fixture, EOS + bad-input handling intact. Certified vs f64
  (max_abs_diff 0.076, argmax 4/4). See
  [HANDOFF_RUNTIME_REAL_1.md](./HANDOFF_RUNTIME_REAL_1.md). Dense Llama-family
  text generation is **GREEN**.
- **Non-Llama end-to-end (RUNTIME-REAL-2).** Re-confirmed on a real 2.88 GB
  Qwen2.5-1.5B-Instruct HF checkpoint (`Qwen2ForCausalLM`,
  `llama-2-with-qkv-bias`: tied embeddings + QKV bias + 151936 vocab + θ=1e6),
  RTX 4070: loads 338 tensors (~2.87 GiB resident, VRAM/RAM tiered) in ~12 s,
  generates coherent + factually-correct text ("…is Paris."), deterministic,
  EOS + bad-input intact. Matches f64 to **0.000335** (argmax 4/4 — tighter than
  PyTorch BF16's own 1.53). See
  [HANDOFF_RUNTIME_REAL_2.md](./HANDOFF_RUNTIME_REAL_2.md). Qwen2.5 dense text
  generation is **GREEN**.
- **Gemma 2 end-to-end (RUNTIME-REAL-3).** Re-confirmed on a real 4.87 GiB
  sharded Gemma-2-2B-it HF checkpoint (`Gemma2ForCausalLM`: softcaps @50/30,
  scaled embeddings ×√2304, query_pre_attn_scalar, GeGLU, dual-norm, multi-EOS
  [1,107]), RTX 4070: loads 288 tensors / 2 shards (~6.75 GiB resident,
  VRAM/RAM tiered) in ~90 s, generates coherent + factually-correct text
  ("…is **Paris**. 🇫🇷"), deterministic, EOS (stop=107) + bad-input intact;
  SoftCap regression 8/8; certified↔fast bit-identical (numcert). **Caveat:** no
  f64 reference exists for Gemma 2 (numcert null — PyTorch f64 pipeline not
  wired for this family), so it is GREEN **behaviourally** but lacks the
  Llama/Qwen numerical-certification bar. See
  [HANDOFF_RUNTIME_REAL_3.md](./HANDOFF_RUNTIME_REAL_3.md).
- **Phi-3 end-to-end (RUNTIME-REAL-4).** Re-confirmed on a real ~7.6 GB sharded
  Phi-3.5-Mini-Instruct HF checkpoint (`Phi3ForCausalLM`: fused QKV split,
  fused gate_up split, LongRoPE; MHA), RTX 4070: loads 195 tensors / 2 shards
  (~8.99 GiB resident, VRAM 32 / RAM 163) in ~138 s, generates coherent +
  factually-correct text ("…is Paris."), deterministic, bad-input intact; phi3
  unit suite 23/23 (fused split + LongRoPE); certified↔fast bit-identical
  (numcert). Same caveat as Gemma 2: **no f64 reference** (numcert null), so
  GREEN **behaviourally** only. See
  [HANDOFF_RUNTIME_REAL_4.md](./HANDOFF_RUNTIME_REAL_4.md).
- **Dense breadth campaign — closed.** Four structurally-distinct families
  validated end-to-end on real checkpoints (RUNTIME-REAL-1..4): **Llama**
  (f64 0.076) · **Qwen2.5** (f64 0.000335) · **Gemma 2** (behavioural) ·
  **Phi-3** (behavioural). Remaining gaps (f64 pipeline for Gemma/Phi,
  throughput, larger variants) are tracked as separate milestones, not new
  breadth validation.
- **Real MoE generation (RUNTIME-MOE-1) — BLOCKED.** Attempted real Mixtral
  end-to-end; the real Mixtral-8x7B weights are **not present** locally
  (`models/Mixtral-8x7B-v0.1/` is a config+tokenizer stub; the only mixtral
  weights in-tree are the synthetic MOE-FULL fixtures). MoE runtime/routing
  stays **topology-certified** (MOE-FULL-15: `mixtral_scale` 1.639e-07,
  `mixtral_layer0` 1.164e-10, scope `certified_scaled`) but **no real-weight run
  exists**. Real Mixtral is **experimental / BLOCKED** pending: (1) provisioning
  ~94 GB real weights (disk-tierable on NVMe but slow), and (2) a sharded-loader
  task for the MoE path (real Mixtral is multi-shard; current MoE loader is
  single-file). See [HANDOFF_RUNTIME_MOE_1.md](./HANDOFF_RUNTIME_MOE_1.md).
- **Small real MoE (RUNTIME-MOE-2) — BLOCKED.** Audited the smallest real MoEs
  for a real-weight load: **Qwen1.5-MoE-A2.7B(-Chat)** (`Qwen2MoeForCausalLM`,
  supported) is **8-shard / 28.6 GB / ~57 GB as f32** and **Phi-mini-MoE**
  (`PhiMoEForCausalLM`) is an **unsupported architecture** + 4-shard. The MoE
  loader is **single-file + f32-into-RAM** (`MoeRuntime::load_from_dir` picks the
  first `.safetensors`; `ExpertTier::Ram` hardcoded, no disk spill), so no
  supported-family real MoE fits on the 32 GB host. Deferred without downloading
  (avoiding a guaranteed-fail 28.6 GB fetch). Unblock = a scoped **engine**
  milestone: (1) sharded safetensors loading in the MoE path, (2) disk-tier /
  bf16 residency — then validate Qwen1.5-MoE-A2.7B. See
  [HANDOFF_RUNTIME_MOE_2.md](./HANDOFF_RUNTIME_MOE_2.md).
- **Sharded MoE loading (MOE-PROD-1) — done.** The MoE runtime now loads
  **sharded** checkpoints (`model.safetensors.index.json` + multiple shards) via
  a new `MoeWeightSource` (single-file **or** sharded), proven **bit-for-bit
  identical** to single-file (`max_abs_diff == 0.0`) with clear errors for
  missing shard / missing tensor / corrupt index. This removes the **first** of
  the two RUNTIME-MOE-2 blockers. The **second** — the compute backend holding
  every weight as f32 in RAM (~57 GB for Qwen1.5-MoE > 32 GB) — is **still
  open** and needs a separate bf16/disk-backed-residency engine milestone. See
  [HANDOFF_MOE_PROD_1.md](./HANDOFF_MOE_PROD_1.md).
- **Disk-backed MoE expert residency (MOE-PROD-2) — done.** The **second**
  RUNTIME-MOE-2 blocker (experts held as f32 in RAM) is removed:
  `ATENIA_MOE_EXPERT_TIER=disk` streams each graph-MoE layer's experts onto NVMe
  during load (peak RAM ~one layer, not the whole model) and runs them through
  the certified `ResidentExpertLayer`. Disk-tier output is **bit-for-bit
  identical** to RAM (`max_abs_diff == 0.0`, `tests/moe_residency_tier_test.rs`);
  the RAM default is byte-identical to before. Estimate: Qwen1.5-MoE-A2.7B drops
  from ~57 GB f32 RAM to ~3 GB steady (experts ~28.6 GB on NVMe) → **fits** the
  32 GB host. Both engine blockers are now gone; reopening RUNTIME-MOE-2 is an
  environment step (download + run). Caveats: disk-tier generation is slow (no
  per-token expert cache yet); DeepSeek/MLA not tiered. See
  [HANDOFF_MOE_PROD_2.md](./HANDOFF_MOE_PROD_2.md).
- **First real MoE generation (RUNTIME-MOE-2 reopened).** Downloaded the real
  **Qwen1.5-MoE-A2.7B-Chat** (27 GB, 8 shards, `Qwen2MoeForCausalLM`, 60 experts
  top-4 + shared, 24 layers) and ran it end-to-end via `atenia moe-generate`
  with `ATENIA_MOE_EXPERT_TIER=disk`: **loads with experts streamed to NVMe
  (~50 GB), RAM bounded ~4 GB** (vs ~57 GB f32), and generates **coherent real
  text** — "What is the capital of France?" → " The capital of France"; "Rust is
  a programming language that" → " is designed to be fast,". Routing (top-4 of
  60 + shared), manifest gate, and fail-loud opt-in all verified; no dense
  fallback. Slow (~2–4 min/token, disk-tier, no expert cache). **Qwen-MoE is now
  real-GREEN behaviourally** (no full f64 for the 14.3 B model; certified at
  topology/block scale, MOE-FULL-15). See
  [HANDOFF_RUNTIME_MOE_2_REOPENED.md](./HANDOFF_RUNTIME_MOE_2_REOPENED.md).
- **Expert-cache integration (MOE-PROD-3).** The disk-tier MoE node now threads
  the existing per-layer `ExpertCache` (`forward_cached`) so repeated routed
  experts are served from RAM (`ATENIA_MOE_EXPERT_CACHE`, default `2*top_k`;
  `ATENIA_MOE_CACHE_STATS=1` reports the hit ratio). Bit-exact (disk+cache == RAM
  `max_abs_diff==0.0`; cached==uncached unit test). **Honest finding:** on real
  Qwen1.5-MoE it gets a **22.7 % routed hit ratio** but **no end-to-end speedup**
  (~3101 vs ~2905 s) — the bottleneck is **load-time NVMe tiering** (50 GB f32 as
  4392 files) + the **shared expert read every token** + per-file overhead, not
  routed re-reads. The real levers are bf16 tier storage + persisting the tier
  across runs. See [HANDOFF_MOE_PROD_3.md](./HANDOFF_MOE_PROD_3.md).
- **Persistent expert tier (MOE-PROD-4).** `ATENIA_MOE_TIER_PERSIST=1` writes the
  disk tier under deterministic names in `<base>/moe_tier/<model_id>/` with a
  `tier_manifest.json` and **reuses** it on a later load instead of rewriting the
  ~50 GB. Bit-exact (reuse mtimes unchanged + identical generation;
  `tests/moe_tier_persist_test.rs`); regenerates on file loss. **Real result** on
  Qwen1.5-MoE: cold load+gen **3757 s** → warm **2445 s** (**~35 % / ~22 min
  saved**; 4392 expert files reused, 0 rewritten, output identical) — the lever
  MOE-PROD-3 pointed at. New bottleneck: shard read + f32 expert assembly (scope
  A still assembles experts; scope B would skip that). Default off = unchanged
  ephemeral tier. See [HANDOFF_MOE_PROD_4.md](./HANDOFF_MOE_PROD_4.md).
- **Warm backend reconstruction (MOE-PROD-5, scope C).** A warm load now rebuilds
  the **whole** MoE backend (experts + attention + embed + lm_head + router +
  gate) **directly from the persistent tier — no shard read, no expert
  assembly** ("reconstructing N layers"), with a safe fallback to the certified
  shard path on any manifest/file doubt. Bit-exact (proved by deleting the shard
  files and still generating identically; `tests/moe_tier_reconstruct_test.rs`).
  **Real result** on Qwen1.5-MoE: warm load+gen **2445 s → 701 s (~71 % / ~29 min
  faster)**, output identical. New bottleneck is the per-token disk-tier
  generation, not the load. (Fixed a shared-expert FFN-width bug found by the
  real run.) Default off = unchanged. See
  [HANDOFF_MOE_PROD_5.md](./HANDOFF_MOE_PROD_5.md).
- **BF16 tier + shared-expert cache (MOE-PROD-6).** Cuts the per-token
  generation I/O that MOE-PROD-5 left as the bottleneck, without touching the
  MoE math/routing/outputs. (a) **BF16 expert tier**: routed + shared experts
  persist as bf16 (half the NVMe bytes), **auto-detected per tensor** — only
  bf16-representable values are truncated (lossless `>>16`/`<<16` round-trip);
  arbitrary f32 stays f32. The read path upcasts bf16→f32 via the existing
  `ensure_cpu` disk arm; warm reads detect the dtype by file size. (b) **Pinned
  shared-expert cache**: the shared expert (read *every* token, ~4× a routed
  expert on Qwen-MoE) is materialised once per layer and reused. Both are
  bit-exact and have env escape hatches (`ATENIA_MOE_TIER_BF16=0`,
  `ATENIA_MOE_SHARED_CACHE=0`) plus the certified shard fallback; manifest
  bumped to v3 (per-tensor dtype). **Real result** on Qwen1.5-MoE (rigorous
  same-prompt A/B): warm load+gen **350 s → 204 s (~42 %, −146 s)** — bf16 tier
  −120 s + shared cache −26 s — with the on-disk tier shrinking **53.3 → 28.6
  GiB**; output identical (token ids `16, 15`). New bottleneck: warm
  reconstruction + CPU expert matmul. See
  [HANDOFF_MOE_PROD_6.md](./HANDOFF_MOE_PROD_6.md).
- **Loaders.** Single-file and sharded HuggingFace safetensors; GGUF
  (F16 / Q8_0 / Q4_K_M / Q5_K / Q6_K). BF16 parameter storage (50 % RAM saving),
  BF16 KV cache (default on), RAM↔NVMe spill with chunked streaming.
- **Adapter layer.** Llama / Qwen 2 / Qwen 3 / Mistral / Phi-3 / Gemma 2 /
  Gemma 3 (text) family logic lives in `src/model_adapters/`; the
  execution core is family-agnostic
  (Phases 11–15). The **config** boundary is closed: all family-specific
  config semantics — defaults, validation, `rope_scaling` parsing (including
  the GGUF path) — are adapter-owned via `ConfigPolicy`
  (`default_*` / `validate_config` / `parse_rope_scaling` /
  `apply_config_defaults`). `LlamaConfig` and `gguf_config.rs` are now
  structural / format parsers only. Phase 16 closed the symmetric
  **weight-mapping** boundary: the GGUF→HF tensor-name mapping is
  adapter-owned via `GgufNameMapper` (default = arch-agnostic common
  rules; Phi-3 / Gemma 2 override), so `pipeline::build_gguf_name_map`
  no longer branches on `arch`. The execution core is fully
  family-agnostic for both config and weight mapping.
- **Determinism.** Greedy generation is reproducible bit-exact (D67 fixture);
  the lib test suite (726 tests passing, 1 ignored) is green.
- **CI.** A minimal GitHub Actions workflow runs on push / PR to `main`
  with **two blocking jobs**: a `cuda-toolkit` job (mirrors the
  locally-validated environment; no GPU, device tests auto-skip) running
  `cargo test --lib -- --test-threads=1` + `cargo test --test
  tinyllama_config_test`, and a `cpu-only` job that **enforces** the
  declared vendor-agnostic invariant (ROADMAP: "the engine's core never
  assumes NVIDIA-specific behaviour"; ADR-003 "GPU as inference
  baseline") — the CUDA-less build links and the non-GPU test subset
  passes (CPU-1 + CPU-2; promoted from non-blocking to blocking in
  CPU-5). Heavy on-disk GGUF / F64 drift tests stay operator-run
  (`#[ignore]`).
- **Operability hardening (M12, complete & CI-green).** The engine now
  explains failures instead of hiding them. The M12.1–M12.5 series closed
  the diagnostics/error-surface gaps: CUDA root-cause & VRAM-probe failures
  propagate instead of being swallowed (M12.1); `atenia run` load failures
  are clean errors with exit 2, not panics (M12.2); a consolidated
  env/hardware diagnostics block + shared tier-plan summary print on the
  run path (M12.3); silent CPU/RAM fallbacks are now visible — once-per-run
  `cuda_matmul` warn, M6 residency failed-layer aggregate, `try_all_paths`
  reasons, `impl Display for LoaderError` + actionable CLI hints (M12.4);
  `cuInit` CUresult is checked and a JSON-render serialise failure exits
  ≠ 0 (M12.5). No control-flow / numeric / tiering changes; success path
  unchanged.
- **Vendor-agnostic CPU-only build (closed).** `cargo build --lib` links
  with no CUDA toolkit installed and the non-GPU test subset passes.
  Previously a tracked link-time debt (the Rust FFI declared the CUDA
  kernel static libs unconditionally), now resolved: CPU-1 (build.rs
  emits an auto-detected `atenia_cuda` cfg) + CPU-2 (every CUDA `extern`
  / `#[link]` block gated `#[cfg(atenia_cuda)]` with
  identical-signature `#[cfg(not(atenia_cuda))]` stubs;
  `cuda_available()` is `false` without the backend). The CUDA build is
  byte-identical (full lib suite, currently 726 tests + 1 ignored). Enforced by the
  now-blocking `cpu-only` CI job (CPU-5). Not a multi-vendor compute
  backend — see *Single vendor* below.
- **Phi-3.5 GGUF end-to-end (closed).** `Phi-3.5-mini-instruct`
  Q4_K_M GGUF loads and generates coherent text end-to-end. Added on
  the GGUF path: Q5_K block decode, Phi-3 LongRope read from the
  GGUF `rope_factors_{short,long}` tensors (not just metadata), and
  the fused-tensor name mapping for both `attn_qkv` -> `qkv_proj` and
  `ffn_up` -> `gate_up_proj`. The post-consolidation validation
  warning that surfaced this is closed; no tracked debt remains.
- **Gemma 2 GGUF end-to-end (closed).** `gemma-2-2b-it` Q4_K_M
  GGUF loads and generates text identical to the HF safetensors
  reference. Six never-validated Gemma 2 GGUF-path defects were
  fixed in isolated, regression-tested layers: a mis-named
  post-FFN norm tensor (GAP-N1), wrong GGUF config keys for
  head_dim / softcaps (GAP-C1/C2), the 4-norm `ffn_norm` ->
  `pre_feedforward_layernorm` mapping with extra-first adapter
  composition (GAP-N2), and the root cause — llama.cpp pre-folds
  the RMSNorm `+1` into the Gemma 2 GGUF weights (measured exactly
  +1.0 element-wise vs HF), so the Gemma 2 GGUF transform table is
  the HF table minus the norm `+1` fold (GAP-T1). No RopeUnpermute
  needed. TinyLlama / Phi-3.5 GGUF regressions unaffected; no
  tracked debt remains.
- **Adapter Toolkit v1 (closed).** ADR-006's microplan is fully
  delivered: AT-0 (the ADR), AT-2 (conformance harness, the
  executable freeze oracle), AT-1a/b/c (the declarative
  `FamilyTensorSpec` data plus golden A/B oracle, then the GGUF
  name maps and load transforms rewired onto it, behaviour-
  preserving), AT-3a (the two hand-copied GGUF completeness gates
  in `pipeline.rs` collapsed into one helper), and AT-3b (each
  adapter declares its required GGUF dtype coverage and a
  conformance test asserts decoder support — preventing the
  Phi-3.5 Q5_K class of runtime `UnsupportedDType` bug). Validated
  end-to-end against two real families (Phi-3.5 + Gemma 2 GGUF
  both identical to HF) without using the imperative escape
  hatch. AT-4 (YAML / JSON templates, scaffolding generator,
  public Adapter SDK) remains explicitly deferred — no
  serialized contract is frozen and no automatic / magic model
  support is promised. A new family still requires a graph
  builder, numeric validation and explicit review.
- **Qwen3 family supported (HF safetensors + GGUF).** `Qwen3-0.6B`,
  `Qwen3-4B`, and `Qwen3-8B` (Q4_K_M GGUF) load and generate
  coherent text on the dev box. Topology delta vs Llama: per-head
  QK-Norm RMSNorm applied to Q and K after reshape-to-heads and
  before RoPE (γ shape `[head_dim]`), plus explicit `head_dim`
  (128, ≠ `hidden_size / num_attention_heads`) threaded through
  the projection shapes (q/k/v/o use `n_heads * head_dim` instead
  of `hidden`), and `attention_bias = false` (no QKV biases,
  opposite of Qwen2's family default). The 1 / √head_dim
  attention scale lives in the K-Norm γ (a pre-normalize scale
  would be stripped by RMSNorm; a post-normalize γ scale
  survives). No new AMG ops introduced. **LM head**: `build_qwen3`
  honours `tie_word_embeddings` — the small variants (0.6B / 1.7B)
  ship a redundant physical `lm_head.weight` while the larger
  variants (4B / 8B / 14B / 32B) are genuinely tied and ship
  none; the tied branch reuses `embed_tokens` transposed and is
  correct for every variant. **GGUF**: `general.architecture =
  "qwen3"` is accepted; the QK-Norm γ tensors map from
  `blk.N.attn_{q,k}_norm.weight`; llama.cpp does not row-permute
  Qwen3 q/k, so the GGUF path reuses the HF transform table
  (no `LlamaRopeUnpermuteRows`).
- **Model family validation.** The per-family mastery batteries
  below are consolidated, with their per-checkpoint tables, real
  fixes and out-of-scope notes, in `docs/MODEL_FAMILY_VALIDATION.md`
  — functional / family validation, distinct from ADR-004 numeric
  certification.
- **Llama-family mastery battery.** End-to-end load + 1-turn
  chat generation on the dev box across the Llama-family scope.
  Prompt `"What is the capital of France?"`, greedy decoding,
  certified mode, no code changes — every case routed through
  the existing `LlamaFamilyAdapter` unchanged. **9 / 9 PASS**:
  TinyLlama 1.1B safetensors, Llama 3.2 1B safetensors,
  Llama 3.2 1B Q4_K_M GGUF, Llama 3.2 3B safetensors,
  Llama 3.1 8B safetensors, Llama 3.1 8B Q4_K_M GGUF,
  Llama 3.1 8B Q5_K_M GGUF, Llama 3.1 8B Q6_K GGUF, and
  DeepSeek-R1-Distill-Llama-8B Q4_K_M GGUF (chain-of-thought
  reasoning trace coherent, max-tokens 120). All emitted
  *"The capital of France is Paris."* and halted on `<|eot_id|>`
  for the Llama-3.x instruct cases (single-EOS path correct;
  `tokenizer_config.json::eos_token` resolves to id 128009).
  The 8B-class loads through tier-aware spill (~3.5 GiB VRAM
  + RAM/Disk tiers) at 0.07–0.10 tok/s — bounded by spill, not
  a correctness regression. Llama 2 7B Chat is gated on HF and
  was not downloaded; Llama 2 13B Chat coverage stays via
  `m5_dc_llama2_13b_coherence_test` (integration test, locked
  regression). No adapter / config / tokenizer / transform
  change was required to land this battery — the audit at
  Phase 2 surfaced no real gap for the family beyond what
  Phases 11–16 already closed. Logs in
  `target/validation/llama_mastery/`.
- **Qwen-family mastery battery.** End-to-end load + 1-turn
  chat generation across the Qwen2 + Qwen3 scope, greedy,
  certified mode. **11 / 11 PASS**: Qwen2.5 1.5B / 3B / 7B
  safetensors, Qwen2.5 7B Q4_K_M / Q5_K_M / Q6_K GGUF, Qwen3
  0.6B / 4B safetensors, Qwen3 8B Q4_K_M GGUF, Qwen2.5-Coder
  1.5B safetensors, and DeepSeek-R1-Distill-Qwen-7B Q4_K_M GGUF
  (coherent chain-of-thought). Unlike the Llama battery this one
  surfaced real gaps and required adapter/config fixes (no core
  change):
  (1) **Qwen3 LM-head tie** — `build_qwen3` always registered a
  separate `lm_head.weight`, which the loader left zero-filled
  for the genuinely tied 4B/8B/14B/32B variants → uniform logits
  → constant degenerate token. Fixed to honour
  `tie_word_embeddings` like `build_llama`.
  (2) **Qwen2/Qwen3 GGUF unsupported** — `general.architecture =
  "qwen2"/"qwen3"` was rejected by the GGUF config arch
  whitelist. Added both, with `qwen2`/`qwen3` metadata-key
  prefixes and `model_type` mapping.
  (3) **Qwen2 GGUF QKV biases** — `COMMON_NAME_TABLE` gained the
  `attn_{q,k,v}.bias` suffixes, and the hard-coded
  `attention_bias = Some(false)` in the GGUF config parser became
  `None` so the adapter default resolves it (Qwen2 → biases on).
  (4) **Qwen3 GGUF QK-Norm** — `QWEN3_SPEC.name_extra` maps
  `blk.N.attn_{q,k}_norm.weight`; the Qwen3 adapter's GGUF weight
  mapper now uses the QK-Norm-aware HF table (llama.cpp does not
  row-permute Qwen2/Qwen3 q/k, so no `LlamaRopeUnpermuteRows`).
  Logs in `target/validation/qwen_mastery/`. The 7–8B-class loads
  run through tier-aware spill at 0.07–0.12 tok/s — bounded by
  spill, not a regression.
- **Gemma-family mastery battery.** End-to-end load + 1-turn chat
  generation across the Gemma 2 + Gemma 3 (text) scope, greedy,
  certified mode. **8 / 8 in-scope PASS**: Gemma 2 2B safetensors,
  Gemma 2 9B safetensors, Gemma 2 9B Q4_K_M / Q5_K_M / Q6_K GGUF,
  Gemma 3 1B safetensors, Gemma 3 1B Q4_K_M GGUF, Gemma 3 4B
  Q4_K_M GGUF (text path) — all emitted a coherent
  *"...the capital of France is **Paris**."* and halted on the
  chat turn-terminator. Gemma 3 (`Gemma3ForCausalLM` /
  `model_type = gemma3_text`) is a **new supported family**: it is
  Gemma 2's topology (dual-norm, GeGLU, embedding scale, soft-cap
  *removed*) plus per-head QK-Norm on q/k and a dual RoPE base
  frequency (local sliding-window layers use `rope_local_base_freq`,
  global layers use `rope_theta`, selected by
  `sliding_window_pattern`). New `Gemma3Adapter` +
  `build_gemma3` builder; no new AMG ops. Gaps found and fixed
  (adapter / config / tokenizer / decoder — no core change):
  (1) **Gemma 3 unsupported** — added the config fields
  (`rope_local_base_freq`, `sliding_window_pattern`), the adapter,
  the builder, the `GEMMA3_SPEC` transform tables, and the GGUF
  `general.architecture = "gemma3"` route.
  (2) **Multi-EOS** — generation collapsed a model's
  `eos_token_id` array to its first element. A Gemma instruct turn
  ends with `<end_of_turn>`, not `<eos>`, so Gemma 3 ran past its
  natural stop into off-distribution garbage. `LlamaConfig` now
  keeps the full `eos_token_ids` set and the generator halts on
  any of them; the pipeline additionally resolves the standard
  chat turn-terminators (`<end_of_turn>` / `<|eot_id|>` /
  `<|im_end|>`) by name from the vocabulary — needed for the GGUF
  path, whose metadata carries only a scalar `eos_token_id`.
  (3) **Q5_0 decoder** — the Gemma 3 1B Q4_K_M GGUF mixes Q5_0
  into its attention tensors (standard llama.cpp quant recipe for
  small models); added the `decode_q5_0` decoder (legacy ggml
  `block_q5_0`, 32-element 22-byte blocks).
  **Out of scope:** Gemma 3 4B *safetensors* is the multimodal
  `Gemma3ForConditionalGeneration` wrapper (vision tower, text
  config nested under `text_config`) — classified out of scope per
  the text-only focus; its text-only Q4_K_M GGUF is validated
  instead. Logs in `target/validation/gemma_mastery/`.
- **Phi-family mastery battery.** End-to-end load + 1-turn chat
  generation across the Phi-3 / Phi-3.5 / Phi-4 scope, greedy,
  certified mode. **8 / 8 PASS**: Phi-3.5-mini safetensors,
  Phi-3.5-mini Q4_K_M / Q5_K_M / Q6_K GGUF, Phi-3-mini-4k
  safetensors, Phi-3-mini-128k safetensors, Phi-4-mini
  safetensors, Phi-4-mini Q4_K_M GGUF — all generated a coherent
  *"The capital of France is Paris…"* continuation. Phi-3 / 3.5
  were already supported (fused QKV + fused gate_up + LongRope);
  this battery extended the existing `Phi3Adapter` / `build_phi3`
  to the rest of the family. Gaps found and fixed (builder /
  config — no core change):
  (1) **Plain-RoPE Phi-3** — `build_phi3` panicked unless the
  config carried a LongRope block, so Phi-3-mini-4k (4k context,
  no `rope_scaling`) crashed on load. Plain RoPE is now
  represented as a unit-factor `LongRope` (`original == max`
  position embeddings, all-1.0 factors) which degenerates exactly
  to standard RoPE — one builder code path, no separate branch.
  (2) **Phi-4 partial rotary** — Phi-4-mini sets
  `partial_rotary_factor = 0.75`: RoPE rotates only the first
  `round(0.75 · head_dim)` dims of q / k, the tail passes through
  un-rotated. Added `LlamaConfig::partial_rotary_factor` +
  `rotary_dim()`; the builder slices q / k into rotary / pass
  halves, rotates the first, and concatenates — built from
  existing `SliceLastDim` / `Concat` / `RoPE` nodes. A Phi-4 GGUF
  carries the rotated count as `phi3.rope.dimension_count`, which
  the GGUF config parser converts back to the fraction.
  (3) **Phi-4 grouped-query attention** — Phi-4-mini is GQA
  (24 q / 8 kv) where Phi-3 / 3.5 are MHA. Phi-3's fused
  `qkv_proj` is sliced into Q/K/V activations (not tiled at load
  like the standalone Llama K/V weight), so the builder now
  repeats each KV head `kv_groups` times in the graph
  (interleaved, matching HF `repeat_kv`) before the attention
  matmul — no-op for the MHA variants. Logs in
  `target/validation/phi_mastery/`. The 3.8B-class loads run
  through tier-aware spill at ~0.2–0.3 tok/s.
- **Mistral-dense mastery battery.** End-to-end load + 1-turn
  generation across the Mistral 7B dense scope, greedy, certified
  mode. **7 / 7 PASS**: Mistral-7B-v0.3 base safetensors,
  Mistral-7B-Instruct-v0.3 safetensors, Mistral-7B-Instruct-v0.3
  Q4_K_M / Q5_K_M / Q6_K GGUF, Mistral-7B-Instruct-v0.2
  safetensors, Mistral-7B-Instruct-v0.2 Q4_K_M GGUF — all
  generated a coherent *"…the capital of France is Paris…"*
  continuation (the base model via `--no-chat-template`).
  **No code change** — the audit confirmed Mistral dense is pure
  Llama topology (GQA 32/8, SwiGLU, RMSNorm, RoPE, no QKV bias,
  untied LM head); `MistralAdapter` already delegates graph build
  and weight mapping to the Llama path, and Mistral GGUF exports
  under `general.architecture = "llama"` so it loads through the
  Llama-family adapter unchanged. v0.3's dropped sliding-window
  and v0.2's `sliding_window = 4096` are both below the smoke
  context length, where sliding-window attention is equivalent to
  full causal — no divergence observed. Mixtral / Mistral-MoE
  remain explicitly out of scope (no MoE path was added or
  exercised). Logs in `target/validation/mistral_mastery/`.
- **SmolLM-family mastery battery.** End-to-end load + 1-turn
  generation across the SmolLM / SmolLM2 scope, greedy, certified
  mode. **9 / 9 PASS** (+ Falcon-3 7B regression): SmolLM2
  135M / 360M / 1.7B-Instruct safetensors, SmolLM2-1.7B-Instruct
  Q4_K_M / Q5_K_M / Q6_K GGUF, SmolLM 135M / 360M / 1.7B-Instruct
  safetensors, and Falcon-3-7B-Instruct safetensors — all loaded
  and generated coherent text (the 1.7B variants emitted
  *"The capital of France is Paris."* and halted on EOS; the
  135M models are verbose but coherent; SmolLM2-360M produced a
  well-formed refusal — a small-model behaviour, not an engine
  defect). **No code change** — SmolLM / SmolLM2 are built on the
  Llama architecture (`architectures = ["LlamaForCausalLM"]`,
  `model_type = "llama"`, MHA, RMSNorm, RoPE, SwiGLU, tied LM
  head), so they resolve to `LlamaFamilyAdapter` directly; there
  is no separate SmolLM adapter and none is needed. Falcon-3 is
  likewise `LlamaForCausalLM` (GQA, explicit `head_dim`) and rides
  the same path. SmolLM2-1.7B is one of the four ADR-004 F64
  fixture models, so its numeric certification was already
  locked. Logs in `target/validation/smollm_mastery/`.
- **Falcon-family mastery battery.** End-to-end load + 1-turn
  generation across the Falcon3 scope, greedy, certified mode.
  **6 / 6 PASS**: Falcon3-1B-Instruct, Falcon3-3B-Instruct,
  Falcon3-7B-Instruct safetensors, and Falcon3-7B-Instruct
  Q4_K_M / Q5_K_M / Q6_K GGUF — all emitted *"The capital of
  France is Paris."* and halted on EOS. Falcon3 is pure Llama
  topology (`architectures = ["LlamaForCausalLM"]`,
  `model_type = "llama"`, GQA, explicit `head_dim = 256`, RMSNorm,
  RoPE `theta = 1000042`, SwiGLU, untied LM head), so it resolves
  to `LlamaFamilyAdapter` directly and Falcon3 GGUF exports under
  `general.architecture = "llama"`. **One config-layer fix:**
  Falcon3-1B/3B-Instruct omit `bos_token_id` from `config.json`
  (it lives only in `generation_config.json`); the parser now
  falls back to `eos_token_id` when the field is absent — a
  generalizable tolerance fix, `LlamaConfig.bos_token_id` is not
  consumed by the generation path (the tokenizer owns BOS). No
  core / adapter / graph change. **Classic Falcon
  (`FalconForCausalLM` / `RWForCausalLM`) is out of scope:** it
  uses LayerNorm (not RMSNorm), parallel attention, and
  multi-query fused QKV — a distinct architecture that would
  require a new graph builder and new AMG nodes. Both classic
  paths fail loud cleanly (safetensors: missing
  `num_key_value_heads`; GGUF: unsupported
  `general.architecture = "falcon"`). Verdict: **Falcon3 dominada;
  Falcon clásico fuera de scope actual.** Logs in
  `target/validation/falcon_mastery/`.
- **Adapter Toolkit v2 (complete).** A declarative layer on top of
  the v1 model-adapter system, in `src/adapter_toolkit/`. A model
  is described by a YAML/JSON DSL instead of a hand-written Rust
  adapter; the toolkit parses it (`dsl`), resolves it to an IR
  (`spec` — `ResolvedAdapterSpec` + the normalised feature
  catalog), and generates a `GeneratedAdapter` that implements the
  v1 `AteniaModelAdapter` supertrait by **pure delegation** to the
  hand-written v1 adapter for the family. The `AdapterRegistry` is
  dynamic and v2-first / v1-fallback. **No core, graph-builder or
  v1-adapter code was modified; no Rust is generated dynamically.**
  Declarative validators (`validate`) fail loud on inconsistent
  specs (`gqa` without `kv_heads`, `fused_qkv` without
  `split_strategy`, contradictory `partial_rotary_factor`, unknown
  family/architecture). Three CLI subcommands — `atenia load`
  (parse + validate + build adapter; never runs generation),
  `atenia debug` (verbose + warnings), `atenia inspect` (auto-detect
  a `config.json` / `*.gguf` model dir and emit a loadable YAML).
  Five shipped examples under `config/adapters/`. **475 / 475**
  `cargo test --lib` (411 v1 baseline + 64 toolkit), zero
  regressions; `inspect → load` round-trips for TinyLlama, Llama
  3.2, Qwen 2.5, Gemma 2, Phi-3.5; v1 `atenia generate` unchanged.
  A post-completion technical-debt audit hardened three points:
  GGUF inspection cannot recover the RoPE variant (llama.cpp folds
  it into the `rope_factors` tensors) so it now emits an explicit
  note instead of guessing; the DSL `config`/`weights`/`attention`
  sections are documented and labelled as *declarative,
  validated-not-applied* constraints (`config.json` stays
  authoritative); an explicit unrecognised `model_type` now fails
  loud instead of being silently swapped. `serde_yaml 0.9` is
  deprecated upstream — accepted, contained (DSL front-end only,
  off the hot path), with a migration TODO.
- **Command-line interface (CLI-0 → CLI-7, complete).** A
  product-grade CLI built as a frontier layer in `src/cli/` — no
  runtime-core, loader or graph-builder change. **CLI-1** — a
  structured error system: stable `E-*` codes, a human
  *what-happened / how-to-fix* format, and a unified exit-code
  scheme (`0` success, `1` system/IO, `2` user input, `3` runtime,
  `101` internal panic), plus a panic boundary that renders a
  caught panic instead of dumping a backtrace. **CLI-2** — a
  logging layer with levels (`--quiet` / `--verbose` / `--debug` /
  `--trace` / `--log-level`), an optional `--log-file`, and a
  per-run `--trace-id`; stdout stays reserved for command results,
  stderr for logs. **CLI-3** — diagnostics: `atenia doctor` (host
  CPU/RAM/CUDA/build), `atenia diagnose --model` (pre-flight model
  check, no generation), `atenia capabilities` (supported families,
  formats, quants), all with `--json`. **CLI-4/5** — `atenia chat`,
  an interactive multi-turn REPL using the model chat template,
  with `/help` `/history` `/reset` `/clear` `/exit`, lazy pipeline
  load, and a streamed token-by-token response. **CLI-6** —
  `atenia download`, a curated downloader (three public, non-gated
  Hugging Face checkpoints: `smollm2-135m`, `tinyllama`,
  `qwen2.5-0.5b`) so first-time users go from `cargo install` to a
  running chat without learning `huggingface-cli`. Sequential
  fetch over `ureq` + rustls with `.partial` + atomic rename and a
  single retry; no resume, no checksum, no arbitrary repo support.
  **CLI-7** — `atenia quickstart`, a first-run UX that prints the
  recommended `doctor` → `download` → `diagnose` → `chat` flow with
  the exact commands; `--download` runs step 2 by reusing the CLI-6
  downloader (no duplication). Plan mode is fully non-interactive
  and scriptable. **517 / 517** `cargo test --lib` at CLI-7 close (the
  lib suite is now **628** with the experimental AQS subsystem); the CLI
  surface is covered by seven integration suites (`cli_errors`,
  `cli_logging`, `cli_diagnostics`, `cli_chat`, `cli_ux`,
  `cli_download`, `cli_quickstart`). Full reference: `docs/CLI.md`.
- **Local validation battery (post-Adapter-Toolkit-v1).** A
  load-and-generate sweep over the 18 checkpoints currently
  present under `models/` on the dev box (RTX 4070 Laptop, 8 GB
  VRAM, default certified mode, 12-token greedy continuation of
  `"The capital of France is "`): **17 PASS / 1 WARN / 0 FAIL**.
  The mandated regressions (TinyLlama Q4_K_M GGUF, Phi-3.5-mini
  Q4_K_M GGUF, gemma-2-2b-it Q4_K_M GGUF) all loaded and
  generated coherent text identical to their HF safetensors
  references. The only WARN was `gpt2-safetensors`, whose
  directory is missing `config.json` (incomplete checkpoint, not
  a load-path defect; GPT-2 is **not** a supported family — no
  GPT-2 adapter is registered). No new failure class surfaced;
  no regression in any previously-validated family.

## Opt-in / experimental (documented profile, not default)

These work and ship, but **off by default** with a known, documented numeric
profile. Operators opt in and accept the profile.

- **Fast mode** (`ATENIA_FAST_MODE=1`) — BF16-Tensor-Core native execution.
  Does **not** satisfy ADR-004 strict on every model by construction; the
  per-checkpoint envelope is documented in
  [ADR-005](./decisions/ADR-005-fast-mode-bf16-tc-envelope.md) (SmolLM2 1.7B
  is the worst-case sentinel of the M4.6 family at 2.33 drift).
- **INT8 W8A16** (`ATENIA_M9_INT8=1`) — measurable speedup
  (−9 % to −14 % on 13B) but misses ADR-004's `< 0.5` gate on all four
  fixture models. Ships **for the drift reason, not the performance reason**.
- **GGUF quantized models** — certified under the **functional** schema
  v2.0.0 (smoke-based, documented drift 0.0–10.19), *not* ADR-004 strict.
  Q4_K_M is aggressive 4-bit quantisation; the drift is intrinsic to the
  format, not an Atenia defect.
- **AQS — Atenia Quantization Search** (AQS-0 → AQS-10, complete). An
  isolated, **CPU-only, opt-in, experimental** research subsystem:
  `QuantizationPolicy` → drift evaluator → end-to-end harness →
  certification report → `3.0.0-draft` manifest → search engine → runner →
  `atenia search` CLI. It never runs in the default path, adds no
  dependency, and produces **draft** output only — it is **not** production
  certification (that remains ADR-004 / ADR-005). On TinyLlama only BF16 is
  ADR-004-certified; AWQ (α=0.25) is the best *useful-lossy* option; GPTQ
  (surrogate and real) did not beat the weight-only plateau. ~93 of the 628
  lib tests cover AQS. Full write-up: [AQS_OVERVIEW.md](./AQS_OVERVIEW.md).
- **MoE — Mixture-of-Experts experimental track** (MOE-0 → MOE-18, closed
  MOE-19). An isolated, **CPU-only, opt-in, experimental** compute + data plane
  in `src/moe/`: detect + fail-loud, classic and packed/fused expert binding,
  sparse execution, graph ops, real layer/stack assembly, validation + smoke
  harnesses, numerical metrics, and HF convention parity with automatic
  selection. Three tiny **real** checkpoints (Qwen1.5-MoE, Qwen2-MoE, Mixtral)
  run end-to-end; numerical parity with HuggingFace is ~1e-10 on the layer-0
  MoE block. The **productive loader still fails loud** on MoE checkpoints —
  MoE is **not** wired into the loader/runtime/Adapter Toolkit/CLI and **no MoE
  family is production-supported**. Full write-up:
  [MOE_OVERVIEW.md](./MOE_OVERVIEW.md).
- **MoE — full transformer + controlled runtime** (MOE-FULL-1 → MOE-FULL-15).
  Building on the experimental track: a full tiny MoE transformer (embeddings →
  attention/RoPE/causal mask → MoE block → lm_head), KV cache + greedy decode,
  expert residency (RAM/NVMe tiers) + LRU cache, GQA, and a **controlled, opt-in
  production path** (`moe::controlled_moe_generate` + `atenia moe-generate`,
  gated by a MoE certification manifest + `ATENIA_ENABLE_MOE=1`). **Three families
  certified vs HF f64 at three levels** — tiny end-to-end (Mixtral 7.451e-08,
  Qwen-MoE 5.960e-08, DeepSeek-MoE MLA attn 9.999e-06), **real-checkpoint** layer-0
  MoE block (~1e-10..1e-11), and **scale topology** (Mixtral-8x7B topology
  1.639e-07, Qwen 16-expert 1.490e-07, DeepSeek 16-routed 7.806e-03; argmax/greedy
  exact). The **dense loader still fails loud**; MoE runs **only** behind the
  explicit opt-in on certified families; unsupported variants (Qwen3 QK-norm,
  DeepSeek Q-LoRA) are refused clearly. **MoE remains experimental** — the
  multi-GB real weights are not certified, and there is no tokenizer text CLI nor
  a dense fail-loud lift. Matrix + verdict: [HANDOFF_MOE_FULL_15.md](./HANDOFF_MOE_FULL_15.md).

## Scaffolding / known limitations

Honest about the rough edges. None of these block the execution-layer thesis,
but they bound what you should rely on.

- **Conservative GPU eligibility.** The tier planner only marks a tensor
  GPU-eligible when `rank >= 2 && name.ends_with("_proj.weight")`. Embeddings,
  tied LM-head inputs, norms, biases, and masks stay in RAM even when VRAM has
  room — because many executor nodes are still CPU-only and call `ensure_cpu`
  before reading. This is deliberate (correctness over utilisation), so GPU SM
  utilisation is low on CLI generation today. Pass 1 of the
  [resolution plan](./ATENIA_RESOLUTION_PLAN.md) addresses the
  *observability* of this first.
- **CPU-only executor nodes.** `RmsNorm`, `Reshape`, `Permute`, `Transpose2D`,
  `IndexSelect`, `Softmax`, `SoftCap`, `BroadcastAdd/Mul`, `Concat`, and
  activations run on CPU. Projection matmuls can hit CUDA but outputs usually
  return to CPU between ops. The tied LM-head path transposes
  `embed_tokens.weight` every forward (CPU-heavy for large-vocab models) — a
  known follow-up.
- **Certified BF16→F32 VRAM slow path (no longer reproduces; not formally
  recertified).** Previously documented as a hard failure
  (`BF16->VRAM slow-path upload failed`) on Mistral 7B and Falcon 3 7B in
  certified/manifest mode. The post-Adapter-Toolkit-v1 local validation
  battery exercised the default certified path on the dev hardware (RTX 4070
  Laptop, 8 GB VRAM) and both `mistral-7b-v0.3` and `falcon3-7b-instruct`
  safetensors loaded and generated coherent text end-to-end. The original
  CUDA error path was not reproduced. This is a single-host observation —
  no F64 numcert was added for these two checkpoints, and "all Mistral /
  all Falcon variants" is not claimed; the underlying helper still returns
  `None` on the swallowed-error path, so a future regression is possible.
  The limitation is downgraded from "fails" to "previously observed,
  currently dormant" pending broader coverage.
- **Decode graph rebuilt per token.** Generation rebuilds a fresh decode graph
  every token (graph rebuild is <1 ms per D68, so this is not the bottleneck,
  but it shapes the GPU-utilisation profile).
- **No production training.** Autograd exists for *graph correctness*, not
  training. No training loops, no optimisers in the runtime. Training is v25+.
- **Single vendor.** NVIDIA CUDA only (sm_70+, Linux + Windows). Intel iGPU
  (v22), AMD ROCm (v23), Apple Metal (v24) are roadmap, not shipped. The core
  never assumes NVIDIA-specific behaviour, and the CUDA-less build is now
  enforced in CI — but multi-vendor execution itself is still not built (a
  CUDA-less binary links and runs the non-GPU surface; it does not provide
  an alternative compute backend).
- **Production hardening — CLI slice done (CLI-0 → CLI-7), engine-internal
  observability pending (v21).** The M12 series plus the CLI-0 → CLI-7 phases
  closed the user-facing error / logging / diagnostics slice: failures
  propagate with a stable `E-*` code and an exit code, logging has explicit
  levels and an optional log file, and `doctor` / `diagnose` report host and
  model state (see the *Command-line interface* entry above). Still pending
  for v21: replay harnesses, the installer/first-run UX, and gating the
  engine-internal `[APX]` / `[ATENIA]` log lines emitted by the runtime core
  on stderr (the CLI log level does not yet reach them — they are printed
  before / independently of the CLI logging layer). Known carried-over issue:
  the adaptive memory-pressure threshold (`0.85`) sits above the OS pagefile
  trigger on RAM-dominated boxes, so the OS pages before the reaction loop
  reacts.

---

## Readiness by audience

- **Systems / Rust / CUDA engineers** — solid ground. The architecture, the
  one-way layering, the kernels, and the test suite are real and inspectable.
- **AI infra engineers** — the tier-aware beyond-VRAM path is the proven
  differentiator on commodity hardware; expect to pin `ATENIA_DISK_TIER_DIR`
  to internal NVMe and to choose `certified` vs `fast` per workload.
- **Researchers / reviewers** — the per-checkpoint F64 certificate is the
  unique, auditable artefact; every number in [numcert/](./numcert/) is
  reproducible offline with a single `cargo test` invocation.
- **End users wanting a turnkey local-LLM tool** — not yet. M12 hardened the
  diagnostics and error surfaces (a prerequisite slice); the turnkey
  installer / first-run UX remains the v21 step.

*Hardware reference for every empirical number above: RTX 4070 Laptop, 8 GB
VRAM, 32 GB DDR5-5600, NVMe SN770, Windows 11.*
