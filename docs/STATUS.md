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
- **Adapter layer.** Llama / Qwen 2 / Mistral / Phi-3 / Gemma 2 family logic
  lives in `src/model_adapters/`; the execution core is family-agnostic
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
  the lib test suite (376 tests) is green.
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
- **Qwen3 family supported (HF safetensors).** `Qwen3-0.6B`
  loads and generates coherent text on the dev box. Topology
  delta vs Llama: per-head QK-Norm RMSNorm applied to Q and K
  after reshape-to-heads and before RoPE (γ shape `[head_dim]`),
  plus explicit `head_dim` (128 in 0.6B, ≠ `hidden_size /
  num_attention_heads = 64`) threaded through the projection
  shapes (q/k/v/o use `n_heads * head_dim` instead of `hidden`),
  and `attention_bias = false` (no QKV biases, opposite of
  Qwen2's family default). The 1 / √head_dim attention scale
  lives in the K-Norm γ (a pre-normalize scale would be
  stripped by RMSNorm; a post-normalize γ scale survives). No
  new AMG ops introduced. Qwen3 GGUF is out of scope this
  phase (`required_gguf_dtypes` empty). Larger Qwen3 variants
  (1.7B / 4B / 8B / 14B / 32B) share the same topology and
  should load through the same adapter without further code,
  modulo hardware fit — none validated end-to-end yet.
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
- **Production hardening — diagnostics slice done (M12), UX/logging pending
  (v21).** The M12.1–M12.5 series closed the observability/error-surface
  slice (see *Operability hardening* above): failures now propagate with a
  root cause and an exit code instead of being swallowed or panicking. Still
  pending for v21: structured logging, replay harnesses, and the
  installer/first-run UX. Known carried-over issue (untouched by M12): the
  adaptive memory-pressure threshold (`0.85`) sits above the OS pagefile
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
