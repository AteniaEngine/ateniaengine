# HANDOFF — Post-M9 consolidation (M10 → Phase 16)

## 1. Scope

The per-milestone HANDOFF series stops at
[HANDOFF M9](./HANDOFF_APX_V20_M9.md). This document is the single
consolidation handoff for everything that closed after it: the rest of
APX v20 (M10, M11), then three post-v20 hardening series — **M12**
(operability / diagnostics), **CPU-1 → CPU-5** (vendor-agnostic build),
and **Phase 16** (weight-mapping boundary).

It is a *snapshot with pointers*, not a ledger.
[docs/STATUS.md](./STATUS.md) is the source of truth for current
readiness; [docs/MILESTONES.md](./MILESTONES.md) is the chronological
narrative with full per-sub-phase detail; [ROADMAP.md](../ROADMAP.md)
holds the forward plan. Where this file and STATUS.md ever disagree,
STATUS.md wins.

## 2. Current-state snapshot

Early research, single-author, active development. APX v20 is closed
end-to-end (M1 → M11): real Llama-family inference on commodity
hardware (RTX 4070 Laptop, 8 GB VRAM, 32 GB RAM, NVMe) with
per-checkpoint numeric certificates. The post-v20 hardening series are
closed; **no tracked debt remains**. `cargo test --lib` is 376/376;
`cargo test --test tinyllama_config_test` is 15 passed / 0 failed /
3 ignored. CI is dual-blocking and green.

## 3. Architecture — core vs adapters

Strict one-way layering, unchanged by the post-v20 work:

- **Execution core** (`amg/`, `gpu/`, `cuda/`, `v17/loader/`,
  tier planner): family-agnostic. Never branches on model identity.
- **Adapter layer** (`src/model_adapters/`): all per-family logic.
  Llama / Qwen 2 / Mistral / Phi-3 / Gemma 2. Selected from
  architecture / model_type metadata.
- **Tier planner**: a pure function of
  `(metadata, free_ram, free_vram, kernel_dtype)` → VRAM / RAM / Disk
  per tensor at load time.

Both family boundaries are now closed:

- **Config boundary** (Phases 13–15): defaults, validation,
  `rope_scaling` parsing — adapter-owned via `ConfigPolicy`.
  `LlamaConfig` / `gguf_config.rs` are structural / format parsers
  only.
- **Weight-mapping boundary** (Phase 16): GGUF→HF tensor-name mapping
  is adapter-owned via `GgufNameMapper`;
  `pipeline::build_gguf_name_map` no longer branches on `arch`.
  Layout transforms were already adapter-routed via
  `GgufWeightMapper` / `HfWeightMapper`.

The supertrait `AteniaModelAdapter` composes `ModelAdapter +
HfWeightMapper + GgufWeightMapper + GgufNameMapper +
StoreBackedGraphBuilder + ResidencyHints + ConfigPolicy`.
Deliberately left as-is (not core-depends-on-arch breaks):
`model_type_for_arch` (documented arch→model_type label bridge) and
`gguf_weight_loading.rs` (family transform fns invoked only via the
adapters).

## 4. Vendor-agnostic build (CPU-1 → CPU-5)

A CUDA-less `cargo build --lib` now links. Mechanism:

- `build.rs` auto-detects CUDA and emits
  `cargo::rustc-cfg=atenia_cuda` only when the kernels actually
  compiled and linked; `cargo::rustc-check-cfg=cfg(atenia_cuda)` is
  declared on every path.
- Every CUDA `extern` / `#[link]` block across `src/cuda/*` and
  `apx4_12/gpu_memory_pool.rs` is `#[cfg(atenia_cuda)]`-gated with an
  identical-signature `#[cfg(not(atenia_cuda))]` stub
  (`unreachable!` / `None` / `Err` / exact CPU result as appropriate).
- `cuda_available()` returns `false` unconditionally without the
  backend, so a CUDA-less binary never enters a CUDA path.

This is a **build** boundary, not a non-NVIDIA compute backend. NVIDIA
CUDA remains the only GPU backend; the CUDA success path is
byte-identical (lib 369/369 at the time of the series, now 376/376
after Phase 16's tests).

## 5. CI

Two **blocking** GitHub Actions jobs on push / PR to `main`:

- `cuda-toolkit (blocking)` — CUDA toolkit present, no GPU on the
  runner (device tests auto-skip): `cargo build --lib` +
  `cargo test --lib -- --test-threads=1` +
  `cargo test --test tinyllama_config_test`.
- `cpu-only / no-CUDA (blocking)` — no CUDA installed; proves the
  vendor-agnostic build links and the non-GPU test subset passes.
  Promoted from non-blocking in CPU-5.

`paths-ignore` skips CI for docs-only commits (`**.md`, `docs/**`,
`LICENSE`). Making `cpu-only` a *required* merge gate (vs
run-failing) is a GitHub branch-protection setting, separate from the
workflow file.

## 6. What is closed (commit anchors)

| Series | Commits |
|--------|---------|
| M12.1 — propagate CUDA root cause / VRAM-probe | `f2da023` |
| M12.2 — clean errors for `atenia run` load (exit 2) | `0f6eed0` |
| M12.3 — env/hardware diagnostics + tier-plan summary | `1733359` |
| M12.4 — visible fallbacks + actionable CLI errors | `f5d9380` |
| M12.5 — `cuInit` checked + JSON render exit ≠ 0 | `590e27a` |
| CPU-1 — `build.rs` emits `atenia_cuda` cfg | `74de40a` |
| CPU-2 C2a / C2b / C2c — FFI gating | `8fcaa6b` / `fd22571` / `8d08491` |
| CPU-5 — `cpu-only` promoted to blocking | `6898de7` |
| Phase 16.1 / 16.2 / 16.3 — weight-mapping boundary | `85010b6` / `3340cda` / `b19f726` |

Doc-refresh commits (docs-only, no CI): `6c837c1` (M12), `e55ee88`
(Phase 16), plus the ROADMAP consolidation. No numeric / control-flow
/ tiering / success-path change in any of the above.

## 7. What comes next

Forward work, organised on four axes (full text in
[ROADMAP.md](../ROADMAP.md) → *Forward roadmap*):

- **Expansion** — v22 Intel iGPU → v23 AMD ROCm → v24 Apple Metal,
  standing on the closed build boundary. Roadmap, not shipped.
- **Extensibility** — adapter boundary fully closed; future direction
  is a more declarative config/mapping description (adapter trait
  stays internal).
- **Runtime / performance** — track β (LLM.int8 outlier
  decomposition) to close the hardware-bound 7B / 13B velocity gap;
  track ε (experimental long-context governor, opt-in,
  non-certified).
- **Product / demo** — the remaining v21 work: installer / first-run
  UX, structured logging, replay harnesses.

**The one open known issue:** the adaptive memory-pressure threshold
(`0.85`) sits above the OS pagefile trigger on RAM-dominated boxes; it
should land below it (~0.78 on the dev box) with hysteresis.
Empirical baseline in [HANDOFF M4.7](./HANDOFF_APX_V20_M4.7.md).

## 8. How to resume / pointers

- [docs/STATUS.md](./STATUS.md) — current readiness (source of truth).
- [docs/MILESTONES.md](./MILESTONES.md) — full chronological history.
- [ROADMAP.md](../ROADMAP.md) — forward plan (four axes).
- [docs/decisions/](./decisions/) — ADR-003 (GPU as inference
  baseline), ADR-004 (F64 reference), ADR-005 (fast-mode envelope).
- Per-milestone deep dives: `docs/HANDOFF_APX_V20_M*.md` (M1 → M9);
  this file consolidates M10 → Phase 16.
