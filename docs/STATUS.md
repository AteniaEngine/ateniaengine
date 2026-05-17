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
  (F16 / Q8_0 / Q4_K_M / Q6_K). BF16 parameter storage (50 % RAM saving),
  BF16 KV cache (default on), RAM↔NVMe spill with chunked streaming.
- **Adapter layer.** Llama / Qwen 2 / Mistral / Phi-3 / Gemma 2 family logic
  lives in `src/model_adapters/`; the execution core is family-agnostic
  (Phases 11–15). The **config** boundary is closed: all family-specific
  config semantics — defaults, validation, `rope_scaling` parsing (including
  the GGUF path) — are adapter-owned via `ConfigPolicy`
  (`default_*` / `validate_config` / `parse_rope_scaling` /
  `apply_config_defaults`). `LlamaConfig` and `gguf_config.rs` are now
  structural / format parsers only.
- **Determinism.** Greedy generation is reproducible bit-exact (D67 fixture);
  the lib test suite (369 tests) is green.
- **CI.** A minimal GitHub Actions workflow runs on push / PR to `main`:
  a **blocking** `cuda-toolkit` job (mirrors the locally-validated
  environment; no GPU, device tests auto-skip) running
  `cargo test --lib -- --test-threads=1` + `cargo test --test
  tinyllama_config_test`, and a **non-blocking, visible** `cpu-only` job
  that guards the declared vendor-agnostic invariant (ROADMAP: "the
  engine's core never assumes NVIDIA-specific behaviour"; ADR-003 "GPU as
  inference baseline"). Heavy on-disk GGUF / F64 drift tests stay
  operator-run (`#[ignore]`).
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
- **Certified BF16→F32 VRAM slow path.** Mistral 7B and Falcon 3 7B fail in
  certified/manifest mode with `BF16->VRAM slow-path upload failed`; the same
  checkpoints pass in fast mode. The helper currently returns `None`, so the
  underlying CUDA error is swallowed — diagnosis is a resolution-plan item.
- **Decode graph rebuilt per token.** Generation rebuilds a fresh decode graph
  every token (graph rebuild is <1 ms per D68, so this is not the bottleneck,
  but it shapes the GPU-utilisation profile).
- **No production training.** Autograd exists for *graph correctness*, not
  training. No training loops, no optimisers in the runtime. Training is v25+.
- **Single vendor.** NVIDIA CUDA only (sm_70+, Linux + Windows). Intel iGPU
  (v22), AMD ROCm (v23), Apple Metal (v24) are roadmap, not shipped. The core
  never assumes NVIDIA-specific behaviour, but multi-vendor is not built. The
  non-blocking `cpu-only` CI job exposed a real build-system drift against
  this invariant on its first run — see *Vendor-agnostic CPU-only build
  drift* below.
- **Vendor-agnostic CPU-only build drift (tracked, prioritised debt).** The
  non-blocking `cpu-only` CI job (no CUDA installed) fails at **link**, not
  compile: *"could not find native static library `batch_matmul`, perhaps an
  -L flag is missing?"* (`cargo build --lib`, exit 101). Cause: `build.rs`
  early-returns on a CUDA-less host (its "CPU-only build" path) and emits no
  `rustc-link-lib` / `-L` directives, yet the Rust FFI side still declares the
  CUDA kernel static libraries (`batch_matmul`, `atenia_kernels`,
  `matmul_kernel`, `linear_cuda`, `fused_linear_silu`, `bf16_to_f32`)
  unconditionally, so `rustc` cannot link the lib. This is a **build-system /
  FFI drift against the declared vendor-agnostic invariant** (ROADMAP: "the
  engine's core never assumes NVIDIA-specific behaviour"; ADR-003 "GPU as
  inference baseline") — **not** a conceptual failure of the engine: the CUDA
  path (the blocking `cuda-toolkit` job) is unaffected, and the `cpu-only`
  job is non-blocking by design precisely to surface this. Remediation is a
  dedicated future task (cfg-gate / stub / feature-split the CUDA FFI so a
  genuine CPU-only build links); it is **not** the Phase 16 weight-mapping
  item below, and it did not block M12.
- **Weight-mapping family boundary still open.** The *config* boundary is
  closed (Phases 13–15), but `src/v17/loader/gguf_to_hf_naming.rs` and
  `src/nn/llama/gguf_weight_loading.rs` still carry
  `if arch == "phi3" / "gemma2"` branches for GGUF→HF weight-name mapping.
  This is the symmetric counterpart not yet relocated behind the adapters'
  `GgufWeightMapper` / `HfWeightMapper` traits — the natural **Phase 16**
  candidate. It did not block M12.
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
