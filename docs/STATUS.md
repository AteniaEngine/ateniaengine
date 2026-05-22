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
  the lib test suite (503 tests) is green.
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
  byte-identical (lib 369/369). Enforced by the now-blocking `cpu-only`
  CI job (CPU-5). Not a multi-vendor compute backend — see *Single
  vendor* below.
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
- **Command-line interface (CLI-0 → CLI-5, complete).** A
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
  load, and a streamed token-by-token response. **503 / 503**
  `cargo test --lib`; the CLI surface is covered by four
  integration suites (`cli_errors`, `cli_logging`, `cli_diagnostics`,
  `cli_chat`, `cli_ux`). Full reference: `docs/CLI.md`.
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
- **Production hardening — CLI slice done (CLI-0 → CLI-5), engine-internal
  observability pending (v21).** The M12 series plus the CLI-0 → CLI-5 phases
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
