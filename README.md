# 🧠 Atenia Engine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17970198.svg)](https://doi.org/10.5281/zenodo.17970198)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/status-early%20research-orange)](./docs/STATUS.md)

> **The execution layer for LLM inference.**
>
> **Atenia is the layer that decides *where*, *when*, and *how* LLM inference runs.**

The way an OS scheduler decides which process runs on which core, Atenia
decides which tensor lives in VRAM, RAM or NVMe, in what numeric precision,
under what real memory pressure — and treats those decisions as a first-class,
auditable layer of the stack, not as plumbing hidden inside a wrapper.

Written from scratch in Rust + CUDA. No PyTorch, no ONNX, no llama.cpp
shell-out. It owns its graph IR, its CUDA dispatch, its memory tiering, and its
numeric policy. Every certified checkpoint ships a numeric certificate proving
its drift against an F64 reference — something **no other runtime in the
ecosystem publishes**.

> *The layer that decides how the model runs. The model itself runs untouched.*

NVIDIA CUDA today (sm_70+, Linux and Windows). AMD ROCm, Apple Metal, Intel
iGPU are v22+ work — see [docs/STATUS.md](./docs/STATUS.md) for the honest
readiness snapshot.

---

## In 30 seconds

**For systems / Rust / CUDA engineers**
- Native graph IR (AMG) with its own CPU + CUDA executors and hand-written
  kernels — the execution core, not a binding.
- Strict one-way layering: adapters never reach into CUDA, the core never
  reaches up into adapters, the tier planner is a pure function.
- Rust 2024, no GC, explicit memory ownership, zero PyTorch / ONNX /
  llama.cpp in the dependency tree.
- Vendor boundary is structural, not aspirational: every CUDA FFI symbol
  is `#[cfg]`-gated, so the core compiles and links with **no CUDA
  toolkit installed**. A blocking no-CUDA CI job enforces this alongside
  the CUDA-toolkit job — it is a *build* boundary, not a non-NVIDIA
  compute backend.

**For AI infra engineers**
- Placement is a decision, not a default: a pure planner maps every tensor to
  VRAM / RAM / NVMe from real free-memory + kernel dtype at load time.
- Beyond-VRAM is structural: Llama 2 13B runs on 8 GB VRAM + 32 GB RAM,
  bit-exact before and after a forced 50 % LRU spill to NVMe.
- Two execution modes (certified F32 / fast BF16-TC), selectable per workload
  and per tensor — the engine owns the speed/precision tradeoff explicitly.

**For researchers and reviewers**
- Numeric policy is auditable: every certified checkpoint ships a
  `numcert.json` with measured drift vs an F64 reference on a 4-position
  fixture.
- One `cargo test` command reproduces every number in every manifest — the
  certificate is verifiable offline, not promised.
- No other inference runtime in the ecosystem publishes per-checkpoint
  numeric certificates.

---

## The execution-layer model

Atenia is a layer of the stack, not a tool you reach for. Its product is the
set of execution decisions it makes — and the fact that those decisions are
isolated, one-way, and auditable.

```
   atenia generate / atenia run / atenia probe         (binary CLI)
   ────────────────────────────────────────────────────────────
   GenerationPipeline / demo::build_and_load_llama     (orchestration)
                              ▼
   Adapter layer  (model_adapters/)                    (per family)
   ┌─────────────────────────────────────────────────────────┐
   │  Llama │ Qwen 2 │ Mistral │ Phi-3 │ Gemma 2             │
   │  ─ graph builder  (scratch + store-backed)              │
   │  ─ weight mapper  (HF safetensors / GGUF)               │
   │  ─ config policy  (defaults per family)                 │
   │  ─ residency hints (VRAM/RAM placement preferences)     │
   └─────────────────────────────────────────────────────────┘
                              ▼
   AMG  (amg/)                                         (graph IR + executor)
   ┌─────────────────────────────────────────────────────────┐
   │  NodeType: MatMul / RoPE (plain, Llama3, LongRope) /    │
   │            SoftCap / SliceLastDim / RmsNorm / ...       │
   │  Forward + backward, deterministic execution            │
   └─────────────────────────────────────────────────────────┘
                              ▼
   Tier planner + dispatch  (gpu/, cuda/)              (placement + kernels)
   ┌─────────────────────────────────────────────────────────┐
   │  Pure planner: (metadata, free RAM, free VRAM, dtype)   │
   │                → Vram / Ram / Disk per tensor           │
   │  CUDA kernels: F32 (TF32), BF16-resident matmul,        │
   │                INT8 dequant, Q4_K_M decode              │
   │  Disk → GPU JIT pipeline (M8.7), CPU prefetch overlap   │
   └─────────────────────────────────────────────────────────┘
                              ▼
   CUDA / cuBLAS / NVMe / sysinfo                      (host platform)
```

The arrows are one-way. The execution core never reaches up into the adapter
layer; the adapter layer never reaches into CUDA; the tier planner is a pure
function. That isolation is what makes the execution decisions auditable in the
first place. See
[docs/ARCHITECTURE_REACTION_STRATEGIES.md](./docs/ARCHITECTURE_REACTION_STRATEGIES.md)
for the reaction-strategy design and [docs/MILESTONES.md](./docs/MILESTONES.md)
for how each layer landed.

---

## Numeric certification — the differentiator

PyTorch, vLLM, llama.cpp, and TGI run BF16 by default and ask you to *trust*
that the numbers are good enough. Atenia ships the **proof**: every certified
checkpoint carries a `numcert.json` manifest with measured drift versus an F64
reference on a 4-position fixture, the exact kernel path used, and the exact
command that reproduces the numbers. The fixture is not a runtime tax — it is
**certification data versioned with the model**, auditable by third parties,
reproducible offline.

**A concrete example.** SmolLM2 1.7B Instruct, certified mode:

| Metric | Value |
|--------|-------|
| Atenia F32 drift vs F64 reference | **0.217** (gate `< 0.5`, 2.3× margin) |
| Argmax match (4-position fixture) | **4 / 4** |
| Industry-default BF16 drift vs F64 | ~14.0 (≈9,700× worse) |
| Fast mode drift (BF16-TC, opt-in) | 2.33 — documented in [ADR-005](./docs/decisions/ADR-005-fast-mode-bf16-tc-envelope.md) |

Manifest: [`docs/numcert/smollm2-1.7b-instruct.numcert.json`](./docs/numcert/smollm2-1.7b-instruct.numcert.json).
Reproduce every number in it (with the four `*_SAFETENSORS_PATH` env vars
pointed at your local checkpoint copies):

```bash
ATENIA_M8_BF16_KERNEL=1 \
cargo test --release --test m8_5_full_family_validation_test \
  -- --ignored --nocapture
```

Numeric policy is documented in
[ADR-004](./docs/decisions/ADR-004-f64-reference-as-default.md) (F64 reference
as default) and [ADR-005](./docs/decisions/ADR-005-fast-mode-bf16-tc-envelope.md)
(fast-mode envelope). All per-checkpoint manifests live in
[`docs/numcert/`](./docs/numcert/); the certificate schema and operator
verification flow are in [docs/CERTIFICATION.md](./docs/CERTIFICATION.md).

---

## See it run

```bash
git clone https://github.com/AteniaEngine/ateniaengine.git
cd ateniaengine
cargo install --path .
```

The build script auto-detects CUDA Toolkit and (on Windows) MSVC BuildTools.
Override via `CUDA_PATH` / `MSVC_TOOLS_PATH` if detection fails.

```bash
# Llama 3.2 1B Instruct — the smallest fully-certified checkpoint.
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir ./models/llama-3.2-1b-instruct

atenia generate \
  --prompt "Explain memory tiering in two sentences." \
  --model ./models/llama-3.2-1b-instruct \
  --max-tokens 60
```

Reproduce the beyond-VRAM demo (Llama 2 13B on 8 GB VRAM) with one command —
`--mode c` forces a 50 % LRU spill to NVMe and verifies the argmax is bit-exact
before and after the spill (≈6.9 min on an RTX 4070 Laptop):

```bash
atenia run --mode c --model ./models/llama-2-13b-chat --cache-dir ./atenia-cache
```

Full CLI reference (GGUF, all flags, JSON output, exit codes):
[docs/CLI.md](./docs/CLI.md). Empirical baseline for the demo:
[docs/MILESTONES.md#m47--beyond-vram](./docs/MILESTONES.md#m47--beyond-vram).

---

## Adaptive memory tiering

Models that don't fit in VRAM still run — without GGUF lock-in. Atenia plans
placement **per-tensor** across VRAM / RAM / NVMe at load time via a pure
planner over `(metadata, free_ram, free_vram, kernel_dtype)`. The trajectory:

- **M6** — tier-aware loader, 1.46× on Llama 2 7B Chat, bit-identical output.
- **M7** — 13B runs on a 32 GB box without BSOD via automatic Disk overflow.
- **M8** — BF16-resident VRAM kernels, 1.31×–1.36×.
- **M8.7** — Disk → GPU JIT pipeline, 154 streamed weights/forward at 98.7 %
  prefetch hit rate, 1.30× more.

The 13B-class beyond-VRAM demo is the proof of life; the same machinery applies
to every certified model. No competitor does this end-to-end on commodity
hardware — vLLM / TGI assume datacenter VRAM, llama.cpp does GGUF-format
offload but not arbitrary safetensors with per-tensor planning.

---

## Multi-family adapter layer

Phi-3's LongRope, Gemma 2's SoftCap and dual-norm, Qwen 2's QKV biases, and
GGUF's fused-weight conventions all live in their own adapter modules
(`src/model_adapters/`). The execution core does not know which family it is
running. Adding a new family is a **contained change**, not a core
modification — proven by the Phase 11–12 migration that moved config policy,
tie-embeddings, attention bias, and LongRope parsing out of the core and behind
the adapter registry. Details in
[docs/MILESTONES.md](./docs/MILESTONES.md#adapter-phases-1112--multi-family-adapter-layer).

---

## Technical enablers

The execution decisions above are made real by:

- **AMG (Adaptive Model Graph)** — graph IR with native CPU + CUDA executors,
  forward + backward autograd, deterministic execution, fused kernels.
- **Two-mode numeric dispatch** — `certified` (cuBLAS F32 + TF32 Tensor
  Cores, default) and `fast` (native BF16 Tensor Cores, opt-in via
  `ATENIA_FAST_MODE=1`), routable per-tensor via the certificate manifest.
- **CUDA kernels** — BF16-resident matmul, INT8 per-group dequant, Q4_K_M
  decode, disk → GPU JIT streaming with CPU prefetch overlap.
- **Loaders** — single-file + sharded HF safetensors; GGUF F16 / Q8_0 /
  Q4_K_M / Q6_K. Same load pipeline, same adapter layer, same numeric
  contract across both formats.

### Certified models

| Model                        | Family      | Params | Format        | Notes                          |
|------------------------------|-------------|--------|---------------|--------------------------------|
| TinyLlama 1.1B Chat          | Llama       | 1.1B   | safetensors   | F64 reference fixture          |
| SmolLM2 1.7B Instruct        | Llama       | 1.7B   | safetensors   | F64 reference fixture          |
| Llama 3.2 1B Instruct        | Llama       | 1B     | safetensors   | F64 reference fixture          |
| Llama 3.2 3B Instruct        | Llama       | 3B     | safetensors   | certified                      |
| Llama 3.1 8B Instruct        | Llama       | 8B     | safetensors   | certified                      |
| Llama 2 13B Chat             | Llama       | 13B    | safetensors   | beyond-VRAM demo target        |
| Qwen 2.5 1.5B Instruct       | Qwen 2      | 1.5B   | safetensors   | F64 reference fixture          |
| Mistral 7B v0.3              | Mistral     | 7B     | safetensors   | certified                      |
| Falcon 3 7B Instruct         | Llama-compat| 7B     | safetensors   | certified                      |
| Phi-3.5 Mini Instruct        | Phi-3       | 3.8B   | safetensors   | LongRope + fused QKV/gate_up   |
| Gemma 2 2B Instruct          | Gemma 2     | 2B     | safetensors   | dual-norm + SoftCap@50/30      |
| TinyLlama Q8_0 / Q4_K_M      | Llama       | 1.1B   | GGUF          | functional certification       |
| Llama 3.2 1B Q4_K_M          | Llama       | 1B     | GGUF          | functional certification       |
| Phi-3.5 Mini Q4_K_M          | Phi-3       | 3.8B   | GGUF          | functional certification       |
| SmolLM2 1.7B Instruct GGUF   | Llama       | 1.7B   | GGUF          | functional certification       |

GGUF models are certified under the **functional** schema v2.0.0 (smoke-based,
documented intrinsic quantisation drift), not ADR-004 strict — see
[docs/STATUS.md](./docs/STATUS.md) for what that means in practice.

---

## What Atenia is not

- **Not a machine learning framework.** No training loops, no optimisers.
  Autograd exists for graph correctness; production training is v25+ work.
- **Not a compiler.** No generic IR lowering, no codegen for arbitrary
  backends. Kernels are hand-written, dispatched per tensor type and
  residency.
- **Not a model zoo with weights bundled.** Bring your own HuggingFace or GGUF
  checkpoint; the certified-models table lists the architectures that work.
- **Not multi-vendor today.** NVIDIA CUDA is the only GPU backend (sm_70+,
  Linux and Windows). The CUDA FFI is fully `#[cfg]`-gated, so the core
  compiles and links with no CUDA toolkit present — a structural vendor
  boundary enforced by a **blocking** no-CUDA CI job, not just an aspiration.
  That is a *build* boundary: it does **not** add a non-NVIDIA compute
  backend. AMD ROCm (v23), Apple Metal (v24), Intel iGPU (v22) remain
  roadmap; the codebase is structured to make them possible, not to claim
  them as shipped.
- **Not an SDK.** The adapter trait is internal; extending Atenia to a new
  family means working inside the repository.
- **Not production-ready for unsupervised deployment.** Early research, active
  development. See [docs/STATUS.md](./docs/STATUS.md).

---

## Documentation

### Start here
- [docs/STATUS.md](./docs/STATUS.md) — honest snapshot of what is cabled to
  production signals vs. what is still scaffolding.
- [docs/CLI.md](./docs/CLI.md) — full CLI reference.
- [docs/CERTIFICATION.md](./docs/CERTIFICATION.md) — numeric-certificate
  schema, generation procedure, operator verification flow.

### Direction and history
- [ROADMAP.md](./ROADMAP.md) — product vision, numeric-contract strategy,
  APX versioning roadmap.
- [docs/MILESTONES.md](./docs/MILESTONES.md) — chronological narrative of every
  shipped milestone (M1 → M11.D.5, Adapter Phases 11–12).
- [docs/ATENIA_RESOLUTION_PLAN.md](./docs/ATENIA_RESOLUTION_PLAN.md) — current
  passes against the RTX 3090 battery findings.

### Architectural decisions
- [docs/decisions/](./docs/decisions/) — ADR-001 through ADR-005.
- [docs/ARCHITECTURE_REACTION_STRATEGIES.md](./docs/ARCHITECTURE_REACTION_STRATEGIES.md)
  — reaction-strategy design grounding adaptive execution.

### Per-checkpoint certificates
- [docs/numcert/](./docs/numcert/) — one `<model>.numcert.json` per certified
  checkpoint.

### Deep dives (HANDOFFs)
Every milestone closes with a `docs/HANDOFF_APX_V20_M*.md` covering the
architectural choices, empirical numbers, and the test fixtures that lock the
contract. Indexed from [docs/MILESTONES.md](./docs/MILESTONES.md).

---

## Reproducing the numbers

The repository ships ~369 `cargo test` functions covering tensor operations,
graph construction, adapter dispatch, weight loading, numeric drift, and
generation contracts:

```bash
cargo test --lib -- --test-threads=1
```

The M4.6 four-model F64 fixture (TinyLlama, SmolLM2, Qwen 2.5, Llama 3.2 1B)
regenerates via the command in
[Numeric certification](#numeric-certification--the-differentiator) above —
every value in [`docs/numcert/`](./docs/numcert/) is verifiable against that
output.

---

## Status, license, citation

**Status.** Early research. Single-author development, active. The public CLI
surface (`atenia generate`, `atenia run`, `atenia probe`) is stable across the
M4.9 → M11 series. Full readiness breakdown: [docs/STATUS.md](./docs/STATUS.md).

**License.** Apache 2.0 — see [LICENSE](./LICENSE).

**Citation.**

```bibtex
@software{atenia_engine,
  author       = {Alonso Albella, Guillermo},
  title        = {Atenia Engine: The Execution Layer for LLM Inference},
  year         = {2025},
  organization = {GAAIA LABS},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17970198},
  url          = {https://doi.org/10.5281/zenodo.17970198}
}
```

**Links.**

- Repository: [github.com/AteniaEngine/ateniaengine](https://github.com/AteniaEngine/ateniaengine)
- Homepage: [ateniaengine.com](https://ateniaengine.com)
- DOI: [10.5281/zenodo.17970198](https://doi.org/10.5281/zenodo.17970198)
- Research paper: [docs/paper/](./docs/paper/) (in preparation)

**Author.** Guillermo Alonso Albella — GAAIA LABS.

---

> Atenia is the layer that decides how the model runs.
> The model itself runs untouched.
