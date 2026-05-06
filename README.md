# 🧠 Atenia Engine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17970198.svg)](https://doi.org/10.5281/zenodo.17970198)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**An execution-centric AI runtime system written from scratch in Rust.**

> [!NOTE]
> **Status: Early research in progress.**  
> This project is a working prototype with architectural scaffolding for 
> hardware-adaptive execution intelligence. Several capabilities are now 
> wired to real hardware signals; others remain scaffolding being connected 
> milestone by milestone.  
> See [Current State](#-current-state) for an honest breakdown, and 
> [ROADMAP.md](./ROADMAP.md) for the delivery plan.

---

## 🚀 The killer demo — reproducible in one command

Llama 2 13B Chat — a 26 GB BF16 model — runs end-to-end on a developer laptop with **8 GB of VRAM and 32 GB of RAM**. It does not fit in VRAM, does not fit in RAM alone, and is mediated by Atenia's M3-e reaction loop with VRAM ↔ RAM ↔ disk offload.

**The transparency contract is bit-exact**: the model's output before and after a forced 50 % LRU spill to disk is **identical** down to the last bit (`argmax = 1, logit = 4.7747` on both sides of the spill). The engine adapts execution to hardware reality; the model output does not change.

**6.9 minutes wall-clock on the dev box** (RTX 4070 Laptop, 32 GB RAM, NVMe spill cache):

```
Phases:
  Build graph    ....   ~1s
  Load weights   .... ~165s   (~156 MB/s)
  Warmup forward .... 200s
  Force LRU spill ... 19s     (866 tensors migrated)
  Post-spill fwd .... 23s
Transparency contract:
  argmax(pre)  = 1, logit 4.7747
  argmax(post) = 1, logit 4.7747
  [PASS] ✓ argmax(pre-spill) == argmax(post-spill) bit-exactly —
          the LRU spill + lazy-restore cycle is mathematically
          transparent at this parameter scale.
Total wall-clock: 6.9 minutes.
```

Reproduce with one command after `git clone` (requires HuggingFace auth for the model download):

```bash
git clone https://github.com/AteniaEngine/ateniaengine.git
cd ateniaengine

# Download Llama 2 13B Chat (~26 GB).
huggingface-cli download meta-llama/Llama-2-13b-chat-hf \
    --local-dir ./models/llama-2-13b-chat \
    --include '*.safetensors' '*.json' 'tokenizer*'

cargo install --path .

atenia run --mode c \
           --model ./models/llama-2-13b-chat \
           --cache-dir ./atenia-cache
```

Three modes are available; full CLI reference in [docs/CLI.md](./docs/CLI.md):

- `--mode a` — clean RAM, no spill (baseline; ~5.4 min on the dev box).
- `--mode b` — autonomous LRU spill triggered by simulated memory pressure (~8 min; trigger plumbing validation).
- `--mode c` — forced 50 % LRU spill, the canonical transparency-contract path (~6.9 min on the dev box).

**Hardware prerequisites for reproduction:**

- **CPU**: x86-64 with AVX2 + FMA (Intel since Haswell 2013, AMD since Excavator 2015). ARM/Apple Silicon support is on the v24 roadmap.
- **RAM**: 32 GB recommended, 28 GB minimum. The CLI warns below 28 GB but does not abort.
- **Disk for spill cache** (`--cache-dir`): NVMe at ≥ 200 MB/s sustained write. The CLI runs a 100 MB benchmark at startup and warns if the chosen path is below the floor.
- **GPU**: optional for the killer demo (`atenia run --mode c` runs end-to-end on CPU with the same transparency contract). **Recommended for the M6 / M7 / M8 generation paths** which use NVIDIA CUDA: M6 routes attention/FFN projections to VRAM (1.46× speedup on Llama 2 7B Chat), M7 spills overflow to NVMe, M8 keeps weights as BF16 in VRAM (1.31× on 7B, 1.36× on 13B over their respective baselines). Tested on RTX 4070 Laptop (8 GB VRAM, sm_89) with CUDA 11.8+. AMD ROCm and Apple Metal are on the v23/v24 roadmap.
- **Disk space**: ~30 GB for the model checkpoint, ~14 GB for the spill cache.

The full empirical baseline lives in [HANDOFF M4.7](./docs/HANDOFF_APX_V20_M4.7.md) (the original beyond-VRAM execution work) and [HANDOFF M4.8](./docs/HANDOFF_APX_V20_M4.8.md) (the 3.5× performance pass that brought the wall-clock from 18.75 min to demoable). The CLI surface is documented in [HANDOFF M4.9](./docs/HANDOFF_APX_V20_M4.9.md).

### Or: Atenia chats (M5 ✅)

Same engine, different subcommand. Greedy generation, token streaming, EOS halt:

```bash
atenia generate \
    --prompt "Hello, how are you?" \
    --model ./models/llama-2-13b-chat \
    --max-tokens 20
```

```text
> Hello, how are you?

Hello! I'm just an AI, I don't have feelings or emotions
```

The architectural headline: **24.24 GiB resident** for the BF16 13B model with prefill + decode graphs sharing weights via `Arc<TensorStorage>` — naïve cloning would have hit ~52 GiB → OOM on a 32 GiB box. See [HANDOFF M5](./docs/HANDOFF_APX_V20_M5.md) for the M5.a–M5.f.a sub-phase chain (eleven commits, twelve architectural decisions D58–D69).

### Or: Atenia uses your GPU (M6 ✅)

The tier-aware loader is **the default** since commit `afaa975`. Attention/FFN projections go to VRAM at load time; the rest stays in RAM. The M6 step 4d mixed-residency dispatch consumes the resident weight without re-uploading per call. Set `ATENIA_LEGACY_LOADER=1` only if you specifically need the pre-M6 CPU-resident path.

```bash
atenia generate \
    --prompt "Hello, how are you?" \
    --model ./models/llama-2-7b-chat \
    --max-tokens 5
```

On the dev box (RTX 4070 Laptop, 8 GB VRAM, 32 GB RAM): **8.22 s/tok vs 12.02 s/tok** on the legacy CPU path — **1.46× faster** with bit-identical output. Atenia's first measured GPU acceleration of an LLM forward.

The plan: 60 attention/FFN projection weights go to VRAM (6.7 GiB), the rest to RAM (9.2 GiB), zero Disk overflow. See [HANDOFF M6](./docs/HANDOFF_APX_V20_M6.md) for the architecture (load-time tier planner + direct-VRAM upload + mixed-residency dispatch) and [INVESTIGATION_M6_REPLAN.md](./INVESTIGATION_M6_REPLAN.md) for the design-iteration history (a May 2 BSOD on the original post-load approach forced a structural redesign).

### Or: Atenia runs Llama 2 13B Chat on a 32 GiB box (M7 ✅)

Same default loader, 13B model. M7 adds a Disk fast-path (raw BF16 bytes mmap → NVMe with no F32 transient) and an adaptive RAM headroom that inflates when the model dominates RAM. Set `ATENIA_DISK_TIER_DIR` to a fast SSD.

```powershell
$env:ATENIA_DISK_TIER_DIR = "D:\atenia-m7-cache"

atenia generate `
    --prompt "Hello, how are you?" `
    --model ./models/llama-2-13b-chat `
    --max-tokens 5
```

The planner places **239 tensors directly on NVMe** (20.14 GiB), 126 in RAM, 38 in VRAM. Peak free RAM **7.36 GiB throughout** the run (rollback floor 2 GiB), `disk_busy_pct` saturated for ~1 s total, no BSOD. 13B runs end-to-end on a 32 GiB box with a single 8 GiB GPU at **36.6 s/tok**. The May 2 BSOD scenario that forced the M6 replan is now closed by construction. See [HANDOFF M7](./docs/HANDOFF_APX_V20_M7.md).

### Or: Atenia keeps weights as BF16 in VRAM (M8 ✅)

`ATENIA_M8_BF16_KERNEL=1` flips on the M8 path. Weights live as BF16 in VRAM at half the F32 byte cost (`numel × 2`), doubling the planner's effective capacity. The dispatcher upcasts each weight to a fresh F32 transient on-device per-matmul (Path B) and runs `cublasGemmEx(F32, F32, F32)` — F32 numerics preserved end-to-end, drift matches the M4.7.2.e CPU path. The flag is gated by an adaptive heuristic: it only activates if `model_total > 0.7 × free_ram`, so 7B models that fit in RAM with headroom keep the M6 path automatically.

```powershell
$env:ATENIA_M8_BF16_KERNEL = "1"
$env:ATENIA_DISK_TIER_DIR  = "D:\atenia-m8-cache"

atenia generate `
    --prompt "Hello, how are you?" `
    --model ./models/llama-2-13b-chat `
    --max-tokens 5
```

```text
[ATENIA] M8 BF16 kernel active: VRAM budget doubles ...
[ATENIA] Adaptive headroom: model 24.24 GiB, free RAM 19.28 GiB
         → RAM headroom 18.75 GiB (8.00 base + 10.75 overflow)
[ATENIA] Tier-aware loader plan:
  VRAM: 82 tensors (6.74 GiB)
  RAM:  124 tensors (0.49 GiB)
  Disk: 197 tensors (17.01 GiB)

> Hello, how are you?

 Hello! I'

---
Generated: 5 tokens in 135.0s (0.04 tok/s)
```

| Path | Llama 2 7B Chat | Llama 2 13B Chat |
|---|---|---|
| M6 / M7.3 baseline | 8.22 s/tok | 36.6 s/tok |
| **M8 (BF16 in VRAM)** | **6.26 s/tok (1.31×)** | **27.0 s/tok (1.36×)** |

The 4-model F64 validation under M8 (TinyLlama 1.1B, SmolLM2 1.7B, Qwen 2.5 1.5B, Llama 3.2 1B) passes ADR-004 with **drift 21–12,500× under threshold** — same numerical envelope as the M4.7.2.e CPU BF16 storage path. Path B was the architectural correction after the M8.4-original "BF16 input × BF16 weight × F32 accumulate" path cascaded BF16 activation truncation through 16-28 layers and broke ADR-004 by 2-5×; Path B keeps the activation as F32 throughout, upcasts the BF16 weight at matmul time, and runs an F32 GEMM that matches M4.7.2.e bit-for-bit modulo cuBLAS internal rounding (1.64e-7 single-op drift). See [HANDOFF M8](./docs/HANDOFF_APX_V20_M8.md) for the full sub-phase ledger and the M8.5 4-model F64 numbers.

### Or: Atenia streams Disk weights to GPU (M8.7 ✅)

`ATENIA_M8_7_ENABLED=1` flips on the Disk → GPU JIT pipeline. For every Disk-tier weight that the M7 planner pushed to NVMe, the dispatch streams it through a host pinned buffer to a VRAM staging slot, runs the M8.4c BF16 kernel, and frees the slot. A single-slot CPU prefetch overlaps the next NVMe read with the current matmul's GPU compute. The Path B numerical contract is preserved end-to-end; argmax matches M8 bit-exact.

```powershell
$env:ATENIA_M8_BF16_KERNEL = "1"
$env:ATENIA_M8_7_ENABLED   = "1"
$env:ATENIA_DISK_TIER_DIR  = "D:\atenia-m8-cache"

atenia generate `
    --prompt "Hello, how are you?" `
    --model ./models/llama-2-13b-chat `
    --max-tokens 5
```

```text
Disk-streamed matmuls (M8.7.0): 154
Disk prefetch hits (M8.7.1.a):  152      (98.7 % hit rate)
BF16-resident matmuls (M8.4c):  232
Generated: 5 tokens in 103.5s (0.05 tok/s)
```

On the dev box (RTX 4070 Laptop, 8 GB VRAM, 32 GB RAM): **20.7 s/tok with M8.7 active vs 27.0 s/tok M8 baseline = 1.30× faster.** The killer demo at seq=4 reports **10.22 s forward (vs 13.50 s M8 close = 1.32×)**. See [HANDOFF M8.7](./docs/HANDOFF_APX_V20_M8.7.md) for the sub-phase ledger and the M8.7.1.b/c VRAM-budget analysis that deferred the async-pipeline follow-up to a future operator with more VRAM headroom.

### Or: Atenia halves the KV cache with no token drift (M8.6 ✅)

The KV cache between decode steps now lives at BF16 by default. The graph stays F32; the cast is applied only in the runtime ledger between `harvest_cache_*` and `overwrite_parameter`. Bytes per token in the Llama 2 13B Chat KV cache **drop from 1.5625 MiB to 0.78 MiB** — at seq=2048, that's **3.2 GiB → 1.6 GiB** of RAM freed, which directly extends the working-set budget M9's tier planner can use.

The decision is gated by the TinyLlama 1.1B-Chat 8-token determinism fixture: under the BF16 default it returned **bit-identical** token IDs (`[29907, 13946, 368, 29991, 2266, 526, 777, 6455]`) and the same decoded text (`"Certainly! Here are some examples"`) as the F32 baseline. No fixture regen, no token drift.

Operators who need the legacy F32 path back can opt out with `$env:ATENIA_LEGACY_F32_KV_CACHE = "1"`. See [HANDOFF M8.6](./docs/HANDOFF_APX_V20_M8.6.md).

### Or: Atenia speeds up Llama 2 13B at INT8 (M9 ⚠️ opt-in / experimental)

`ATENIA_M9_INT8=1` activates the INT8 W8A16 weight-quantisation path. The full INT8 infrastructure shipped (per-group quantizer, CUDA dequant kernel, `TensorStorage::CpuInt8`, loader integration, tier-planner cost model). The M9.0 microbench measured **~2× speedup** over the M8.4c BF16 dispatch on the four dominant Llama 13B shapes. End-to-end on the dev box (RTX 4070 Laptop, 8 GB VRAM, 32 GB RAM):

| Configuration | s/tok | Δ vs M8.7 |
|---|---:|---|
| M8.7 baseline (no M9) | 20.7 | — |
| **M9 default** (`ATENIA_M9_INT8=1`) | **18.8** | **−9%** |
| **M9 + `ATENIA_RAM_HEADROOM_OVERRIDE_GIB=8`** | **17.7** | **−14%** |

The default mode is structurally correct out of the box (post-close fixes `997fd8c` cost-honest + `0d0b4a5` two-pass headroom). The override flag pushes ~50 disk-streamed tensors back to RAM-resident at the cost of a thinner safety margin against OS pagefile thrash — operator opt-in.

**Honest disclaimer**: simple absmax INT8 (per-channel and per-group {32, 128}) does **not** satisfy ADR-004 strict (`max_abs_diff < 0.5` vs F64) on the four-model fixture. Best drift result was Llama 3.2 1B at 0.516 under group size 128 — only 3.2% over the gate; worst was SmolLM2 1.7B at 11.25. Argmax preservation is good (typically 3/4 to 4/4 positions match the F64 reference); absolute-magnitude drift cascades over 16-32 layers in a way absmax alone cannot fully control.

**Use M9 INT8 if** you accept the drift trade-off in exchange for ~9–14% faster generation — text remains coherent. **Do not use M9 INT8 if** you need ADR-004 numerics; stay on the M8.6 default. ADR-004-strict INT8 requires either FFN-only mixed precision (M9.5) or LLM.int8 / GPTQ outlier decomposition (M10+).

```powershell
$env:ATENIA_M8_BF16_KERNEL = "1"
$env:ATENIA_M9_INT8 = "1"          # opt-in, NOT ADR-004 strict
cargo run --release --example llama2_13b_demo
```

See [HANDOFF M9](./docs/HANDOFF_APX_V20_M9.md) for the full 12-run drift matrix and the four follow-up tracks.

---

## 🎯 Vision

Modern AI runtimes assume stable hardware.

**Reality does not.**

GPUs are shared. Memory pressure fluctuates. Schedulers jitter. Execution policies thrash. Most production failures in AI systems are not numerical bugs — they are **decision failures** in the execution layer.

Atenia Engine aims to treat execution as a first-class adaptive system: one that observes runtime signals, reasons about stability and risk, and adapts execution policies without modifying computational semantics.

This repository contains the architectural foundation and the reference implementation under active development.

---

## 📐 Numeric certification

Atenia certifies every supported model with **auditable drift data versus an F64 ground-truth reference**. Each checkpoint ships with a `<model>.numcert.json` manifest (in [`docs/numcert/`](./docs/numcert/)) that documents, with measured numbers, exactly how much each execution mode drifts from the F64 reference on a 4-position fixture forward — **and exactly which `cargo test` invocation reproduces those numbers.**

This is the core differentiator: **no other inference engine in the ecosystem ships per-checkpoint numerical certificates.** PyTorch / vLLM / llama.cpp / TGI run with industry-default BF16 and ask you to trust that the numbers are good enough; Atenia runs at the same speed where it matters and ships the proof.

**Two execution modes**, both auditable:

- **`certified`** *(default)* — `cublasGemmEx(F32, F32, F32)` with `CUBLAS_COMPUTE_32F_FAST_TF32` (TF32 Tensor Cores). Every model in the M4.6 fixture passes [ADR-004](./docs/decisions/ADR-004-f64-reference-as-default.md) strict (`max_abs_diff < 0.5` vs F64) with margin 2.3× to 51× depending on the model. Use for research, audited deployments, anything whose numerical equivalence to F64 must be defensible.
- **`fast`** *(opt-in via `ATENIA_FAST_MODE=1`)* — native BF16 Tensor Cores via `cublasGemmEx(BF16, BF16, F32)`. Industry-standard drift profile documented per-checkpoint in [ADR-005](./docs/decisions/ADR-005-fast-mode-bf16-tc-envelope.md). 1.14× to 1.19× faster on 13B-class workloads (M8.7 path) over certified; up to ~2× on a fully-VRAM-resident model. Use for chat / generation workloads where industry-standard drift is acceptable.

**Empirical drift envelope** (4-model F64 fixture, dev hardware):

| Model               | Certified drift | Fast drift  | Recommended mode |
|---------------------|----------------:|------------:|------------------|
| TinyLlama 1.1B Chat | 0.076039        | 0.901545    | `certified`       |
| SmolLM2 1.7B        | 0.217142        | 2.331949    | `certified`       |
| Qwen 2.5 1.5B       | 0.022496        | 0.184907    | **`fast`**        |
| Llama 3.2 1B Inst.  | 0.009715        | 0.268043    | `certified` (argmax flip at pos 2 under fast) |

Every number above is reproducible. Run

```sh
cargo test --release --test m8_5_full_family_validation_test \
  -- --ignored --nocapture
```

with `ATENIA_M8_BF16_KERNEL=1` (and optionally `ATENIA_FAST_MODE=1` for fast-mode numbers) and the four `*_SAFETENSORS_PATH` env vars pointed at your local checkpoint copies. The fixture prints the drift, argmax match, and dispatcher counter delta — every value in the manifest is verifiable against this output.

See [docs/CERTIFICATION.md](./docs/CERTIFICATION.md) for the full schema, generation procedure, and operator verification flow.

---

## ⚙️ Execution Is Not Plumbing

In most AI systems, execution is treated as plumbing:

```
launch kernels → move data → hope the hardware behaves
```

Atenia Engine starts from a different premise:

> **Execution makes decisions. Decisions must adapt to reality.**

Execution determines *where*, *when*, and *how* computation runs. Under dynamic conditions, these decisions must be observed, reasoned about, stabilized, and refined over time.

---

## 🧭 Design Principles

These are the principles guiding every design decision in Atenia. Some are fully realized today, others are under active development:

- **🧱 Stability before performance** — Short-term gains mean nothing if execution collapses under noise.
- **🔒 Adaptation without semantic drift** — The engine may change *how* things run, never *what* is computed.
- **🧠 Learning by experience, without ML** — Execution outcomes are distilled into persistent memory — no opaque training loops in the runtime.
- **🔬 Observable and reproducible** — Every behavior claimed by the engine must be verifiable through executable tests.

See [docs/ARCHITECTURE_REACTION_STRATEGIES.md](./docs/ARCHITECTURE_REACTION_STRATEGIES.md) for the reaction-strategy design that grounds future APX milestones.

---

## 📊 Current State

Atenia Engine is implemented in Rust. The project follows an APX (Adaptive Execution) versioning scheme from v1 to v25, describing the progression from primitives to a fully adaptive runtime.

### ✅ What works today

Real, executable, deterministic:

**Real LLM inference (APX v20 M4.5–M8)**

Headline progression on the dev box (RTX 4070 Laptop, 32 GB RAM, NVMe):

| Milestone | Llama 2 7B Chat | Llama 2 13B Chat | Numerics |
|---|---|---|---|
| **M4.7** baseline (CPU) | — | runs end-to-end via VRAM ↔ RAM ↔ NVMe LRU spill (~6.9 min wall-clock) | bit-exact transparency contract |
| **M4.8** perf pass | — | **3.5×** faster (5.38 min) | F64 4-model drift improved on every model |
| **M5** | 12.02 s/tok (CPU) | 36.6 s/tok | conversational text, 24.24 GiB resident (Arc-shared) |
| **M6** (tier-aware loader) | **8.22 s/tok (1.46×)** | — | bit-identical to CPU baseline |
| **M7** (Disk overflow) | — | **36.6 s/tok**, 239 tensors on NVMe, no BSOD | bit-identical |
| **M8** (BF16 in VRAM) | **6.26 s/tok (1.31×)** | **27.0 s/tok (1.36×)** | ADR-004 4-model passes with margin 21–12,500× |
| **M8.7** (Disk → GPU JIT) | — | **20.7 s/tok (1.30× over M8)** | argmax bit-exact with M8, prefetch hit rate 98.7 % |
| **M8.6** (BF16 KV cache) | — | KV cache **3.2 GiB → 1.6 GiB** at seq=2048 | TinyLlama determinism fixture bit-identical |

Each milestone's deep dive lives in its HANDOFF; the bullets below summarise the architecture each one shipped.

- **🌊 Disk → GPU JIT pipeline (M8.7 ✅).** With `ATENIA_M8_7_ENABLED=1`, every Disk-tier weight in the 13B forward streams from NVMe through a host pinned-buffer cache into a transient VRAM staging slot, dispatches via the M8.4c BF16 kernel, and frees the slot — replacing the M8 baseline's CPU-AVX2 decode + matmul on disk-tier layers. A single-slot CPU prefetch overlaps the next NVMe read with the current matmul's GPU compute (98.7 % hit rate empirically). Path B is now stream-aware; future operators with more VRAM headroom can flip M8.7.1.b/c on for full async upload + dedicated copy/compute streams without re-touching the kernel. 154 disk-streamed matmuls per forward on the 13B; argmax bit-exact with M8. See [HANDOFF M8.7](./docs/HANDOFF_APX_V20_M8.7.md).

- **🧮 BF16-resident VRAM kernels (M8 ✅).** Weights live as BF16 in VRAM at half the F32 byte cost; the dispatcher upcasts each weight to a fresh F32 transient on-device per-matmul (Path B) and runs `cublasGemmEx(F32, F32, F32)`. F64 4-model validation passes ADR-004 with margin 21–12,500×. The flag is gated by an adaptive heuristic (`model_total > 0.7 × free_ram`), so 7B-class models that fit in RAM keep the M6 path automatically. See [HANDOFF M8](./docs/HANDOFF_APX_V20_M8.md).

- **🧱 13B-friendly tiers (M7 ✅).** Disk fast-path (raw BF16 bytes mmap → NVMe with no F32 transient) + adaptive RAM headroom that inflates when the model dominates RAM. Llama 2 13B Chat ran end-to-end with 239 tensors on NVMe, peak free RAM ≥ 7.36 GiB, no BSOD. The May 2 BSOD scenario that forced the M6 replan is closed by construction. See [HANDOFF M7](./docs/HANDOFF_APX_V20_M7.md).

- **🚀 Tier-aware GPU loader (M6 ✅).** Per-tensor residency plan (`(metadata, free_ram, free_vram) → Tier`) decided at load time. Attention/FFN projection weights go directly to VRAM; the M6 step 4d mixed-residency dispatch consumes resident weights without re-uploading per call. Atenia's first measured GPU acceleration of an LLM forward, bit-identical to the CPU baseline. See [HANDOFF M6](./docs/HANDOFF_APX_V20_M6.md) and [INVESTIGATION_M6_REPLAN.md](./INVESTIGATION_M6_REPLAN.md) for the May 2 BSOD post-mortem that forced the architectural redesign.

- **💬 Atenia chats (M5 ✅).** `atenia generate` ships: HF byte-exact chat-template rendering (Jinja2), KV-cache-aware attention via `NodeType::Concat axis=2` + RoPE position offset, prefill + decode graphs sharing weights via `Arc<TensorStorage>` so the BF16 13B model stays at **24.24 GiB resident** (vs ~52 GiB naïve). Twelve architectural decisions (D58–D69) locked across M5.a–M5.f.a. R2 graph-level falsifier 3/3, R6 generation contract 4/4, D67 determinism fixture bit-exact across runs. See [HANDOFF M5](./docs/HANDOFF_APX_V20_M5.md).

- **🚀 Llama 2 13B Chat runs end-to-end on dev-class hardware in 6.9 minutes, reproducible with one command.** RTX 4070 Laptop with 8 GB VRAM and 32 GB RAM — a workload that **does not fit in VRAM**, **does not fit in RAM alone** (BF16 weights ≈ 26 GB), and is mediated by Atenia's M3-e reaction loop with VRAM ↔ RAM ↔ disk offload. The transparency contract is exact: `argmax(clean RAM) == argmax(after forced 50 % LRU spill) == 1, logit 4.7747` **bit-exactly**, on the same input. The selective LRU spill + lazy-restore cycle (M4.7.5.d + M4.7.4.d) writes 13 GB across **866 tensors** to NVMe in 19 s, then restores them through the `ensure_cpu` Disk-arm during a 23 s post-spill forward — without changing a single bit of the output. End-to-end wall-clock on the dev box: warmup forward 200 s, spill 19 s, post-spill forward 23 s — **6.9 minutes total** from `atenia run --mode c` to `[PASS] ✓`. The v20 thesis "adapt execution to hardware reality, not the other way around" is demonstrated against a real workload, not synthetic memory-pressure injection. See [HANDOFF M4.7](./docs/HANDOFF_APX_V20_M4.7.md) for the full empirical baseline and [HANDOFF M4.9](./docs/HANDOFF_APX_V20_M4.9.md) for the public reproduction surface.
- **⚡ Performance optimization (M4.8) — 49.5× speedup on the production matmul shape.** The 18.75-minute baseline 13B forward dropped to **5.38 minutes** (3.49× cumulative) via a six-step rewrite of the CPU matmul path: numeric `apx_mode_at_least` comparison closing a latent lex-compare bug, runtime `is_x86_feature_detected!("avx2")` registration replacing a compile-time `#[cfg]` gate, 8-lane AVX2 BF16 → F32 SIMD decode (5.71 → 15.77 GB/s), rayon `par_chunks_mut` for BatchMatMul (7.1× over serial) and per-row MatMul partitioning, and `matrixmultiply::sgemm` cache-blocked panels for shapes ≥ 1 MFLOP. Per-shape gains on the bench harness: `4×5120×13824` **49.5× (1954 → 39 ms)**, `1×5120×5120` 13.4×, `1×4096×32000` 9.2×, BatchMatMul `40×4×128×128` 4.25×. Vendor-agnostic by construction: AVX2 + FMA baseline (Intel **and** AMD x86-64), NEON-ready for Apple Silicon (v24), **no MKL** anywhere in the dep graph. F64 four-model drift improved on every M4.6 family model under the new path. See [HANDOFF M4.8](./docs/HANDOFF_APX_V20_M4.8.md) for the full sub-step breakdown.
- **🖥 Public CLI (M4.9) — `atenia run --mode c` reproduces the killer demo in one command.** Single binary with three subcommands: `probe` (cross-vendor hardware enumeration, gated behind `hw-probe` feature), `run --mode {a|b|c}` (the tri-mode killer-demo runner), and `explain` (legacy v13 narrative explainer, preserved). Mode A is the clean-RAM baseline (5.4 min on the dev box), Mode B validates the autonomous LRU spill trigger (~8 min, panic absorbed via `catch_unwind`), Mode C is the canonical transparency-contract path (6.9 min on the dev box, exit code 3 on contract violation). Output in human-readable text or stable-schema JSON for scripted reproduction. Heartbeat dots stream to stderr during long phases; final report goes to stdout so `--output json | jq …` works cleanly. Hardware soft-warning when total RAM < 28 GB; cache-dir disk-throughput probe with 200 MB/s warning floor. **No new env vars introduced** — overrides go via CLI flags. See [docs/CLI.md](./docs/CLI.md) for the full reference and [HANDOFF M4.9](./docs/HANDOFF_APX_V20_M4.9.md) for the closing notes.
- **🤖 Four production LLMs run end-to-end on CPU.** TinyLlama 1.1B Chat, SmolLM2 1.7B Instruct, Qwen 2.5 1.5B Instruct, and Llama 3.2 1B Instruct load from HuggingFace `.safetensors` and produce logits validated against PyTorch F64 ground truth per [ADR-004](./docs/decisions/ADR-004-f64-reference-as-default.md). Atenia F32 drift sits between **1.32×10⁻⁴ and 1.45×10⁻³** — three-to-four orders of magnitude closer to mathematical truth than industry-default BF16 inference on the same checkpoints. Argmax matches F64 on every position of every model. Each of these four is also re-validated bit-exact under M4.7's BF16 storage + GPU dispatch + disk spill + LRU policy and again under M4.8's parallel matmul stack — the entire reaction-loop + perf cycle is mathematically transparent at 1B-class scale.
- **📦 Sharded safetensors loader.** Multi-file HuggingFace checkpoints (`model-NNNNN-of-NNNNN.safetensors` + `index.json`) load via `ShardedSafetensorsReader` with drop-after-decode. Verified on Mistral 7B v0.3 (14.5 GB across 3 shards) and Llama 2 13B Chat (26.0 GB across 3 shards); peak RAM stays bounded by the largest single shard, not the sum.
- **💾 Native BF16 parameter storage.** `TensorStorage::CpuBf16(Vec<u16>)` halves the persistent RAM footprint of model parameters (verified at exactly 50.0 % on TinyLlama). All four 1B-class production checkpoints re-validated under BF16 storage active — drift bit-exact identical to the precision-floor spike, all under the ADR-004 threshold. The same storage variant carries the 26 GB Llama 2 13B parameter set on a 32 GB box.
- **🌊 RAM ↔ SSD streaming with LRU eviction.** `migrate_selected_cpu_to_disk` spills the bottom 50 % of the M4.7.5.b touch order on `DeepDegrade` verdicts; `ensure_cpu` Disk-arm dispatches on the on-disk dtype (`DiskDtype::F32` or `BF16`) and lazy-restores BF16 bytes as F32 on-CPU at the next consumer. Files keep the 50 % footprint contract on disk too (BF16 spilled at native 2-byte width, not upcast to F32). Validated bit-exact on every 1B-class model and on the 13 B demo target.

**Engine foundations**

- **🦀 Tensor engine** — Forward + backward with autograd, CPU + CUDA paths.
- **🧩 AMG (Adaptive Model Graph)** — Graph representation with its own executor, independent of PyTorch/TF.
- **⚡ Fused kernels** — Attention and QKV on CPU and GPU.
- **🖼 Convolutional ops** — `Conv2D` and `MaxPool2D` natively in AMG (forward, backward, tape) (APX v20 M1).
- **🔒 Deterministic execution** — Same input, same output, every time.
- **🏗 Portable build** — `build.rs` auto-detects CUDA Toolkit and MSVC BuildTools; overridable via `CUDA_PATH` and `MSVC_TOOLS_PATH`.

**Adaptive execution scaffolding**

- **📡 Real memory telemetry** — VRAM (via `nvidia-smi`) and RAM (via `sysinfo`), validated against ground-truth readings.
- **🧠 Signal producers** — `FailureCounter` (time-windowed) and `LatencyMonitor` (P50-baseline spike detection) with deterministic purge semantics.
- **🚦 SignalBus** — Assembles real telemetry into `GuardConditions` (v16) and `PolicyEvidenceSnapshot` (v15).
- **⚡ Reactive execution hook** — `ReactiveExecutionContext` wires the SignalBus to the AMG executor; `execute_checked` returns a typed `ExecutionAbortReason` on guard trip (APX v20 M2).
- **📋 Policy registry** — Pure, deterministic, explainable policy layer with evidence-aware evaluation.
- **📜 Execution contracts** — Data structures, validators, replay scaffolding.

### 🟡 Scaffolding still being wired to runtime signals

Architecture in place, full end-to-end wiring in progress:

- **Production guards** — The guard framework (v16) and SignalBus (v19) are real, but built-in `ExecutionGuard` implementations live in tests only. Guards in production code are a v21 deliverable.
- **Execution Policies (v15)** — `DecisionBias` reacts to real SignalBus evidence, but no execution path consumes the resulting bias yet.
- **AMM Forecaster** — Exposes real VRAM and RAM. Integration with memory-manager allocation decisions is pending.
- **Fusion Selector (v6.10/6.11)** — `fused_qkv_us = 0` is a placeholder awaiting paired measurement.
- **`FragmentationWarning`** — Intentionally not emitted (external proxies would be misleading). Deferred until Atenia exposes its own GPU allocator.
- **MNIST pipeline** — Conv2D / MaxPool / Dense run end-to-end on synthetic data; real MNIST dataset + trained weights pending.
- **GPU residency-aware MatMul / BatchMatMul** — shipped as **M4.7.3 ✅** (residency triples) → **M6 ✅** (mixed-residency dispatch + load-time tier planner) → **M8 ✅** (BF16-resident kernel via Path B: BF16 weight in VRAM + F32 transient upcast + cublasGemmEx F32 GEMM). End-to-end at production: `try_gpu_matmul` recognises `(Cpu, Cuda(BF16), Cpu)` triples under `ATENIA_M8_BF16_KERNEL=1` and routes to `cuda_matmul_bf16_inplace`. Linear residency stays gated behind the `try_gpu_linear` MiniFlux constraint and is a separate milestone.

### Roadmap (APX v18 → v25)

**Completed:**

- **v12** — Initial learning engine scaffolding (withdrawn paper)
- **v13** — Hybrid Execution Engine (H.E.E.) — adaptive placement scaffolding
- **v14** — Execution timeline + profile infrastructure
- **v15** — Policy layer (5 built-in policies, evidence-driven evaluation)
- **v16** — Execution contracts and guard framework
- **v17** — Kernel normalization and symbolic GPU chain
- **v18** — Memory telemetry foundation:
  - Real VRAM probe via `nvidia-smi`
  - Memory pressure detection (replaces the withdrawn predictive fallback test)
  - System RAM telemetry via `sysinfo`
- **v19** — SignalBus: integrated sensor-to-decision pipeline
  - All 4 `GuardConditions` fields sourced from real telemetry
  - 4 of 5 `PolicySignalKind` variants produced; `FragmentationWarning` deferred
  - `FailureCounter` and `LatencyMonitor` as internal state producers

**v20 — Real model runtime integration (in progress, M1–M9 closed):**

| Milestone | What it closed | HANDOFF |
|---|---|---|
| **M1–M3** | Conv2D / MaxPool2D in AMG; reactive executor; real GPU storage + M3-e VRAM→RAM migration | [M3](./docs/HANDOFF_APX_V20_M3.md) |
| **M4 → M4.6** | Safetensors loader; TinyLlama / SmolLM2 / Qwen 2.5 / Llama 3.2 end-to-end; F64 4-model validation per [ADR-004](./docs/decisions/ADR-004-f64-reference-as-default.md) | [M4](./docs/HANDOFF_APX_V20_M4.md) → [M4.6](./docs/HANDOFF_APX_V20_M4.6.md) |
| **M4.7** | Beyond-VRAM killer demo. Llama 2 13B Chat on 8 GB VRAM + 32 GB RAM via VRAM ↔ RAM ↔ NVMe LRU spill, transparency contract closed bit-exactly | [M4.7](./docs/HANDOFF_APX_V20_M4.7.md) |
| **M4.8** | Performance pass. **49.5×** on the production matmul shape, **3.5×** on 13B end-to-end. AVX2 + FMA + NEON, no MKL | [M4.8](./docs/HANDOFF_APX_V20_M4.8.md) |
| **M4.9** | Public CLI. `atenia run --mode c` reproduces the killer demo in 6.9 min via one command. Stable JSON schema | [M4.9](./docs/HANDOFF_APX_V20_M4.9.md) + [CLI](./docs/CLI.md) |
| **M5** | Inference UX. `atenia generate` ships; Llama 2 13B Chat answers conversationally; **24.24 GiB resident** via Arc-shared weights | [M5](./docs/HANDOFF_APX_V20_M5.md) |
| **M6** | Tier-aware GPU loader. **1.46×** on Llama 2 7B Chat with bit-identical output | [M6](./docs/HANDOFF_APX_V20_M6.md) |
| **M7** | 13B-friendly tiers. Disk fast-path + adaptive RAM headroom; 13B on 32 GiB box without BSOD | [M7](./docs/HANDOFF_APX_V20_M7.md) |
| **M8** | BF16-resident VRAM kernels (Path B). **1.31× on 7B / 1.36× on 13B**; F64 4-model passes ADR-004 with margin 21–12,500× | [M8](./docs/HANDOFF_APX_V20_M8.md) |
| **M8.7** | Disk → GPU JIT pipeline. **20.7 s/tok on 13B (1.30× over M8 baseline)**, 154 disk-streamed matmuls per forward, 98.7 % CPU-prefetch hit rate, argmax bit-exact with M8 | [M8.7](./docs/HANDOFF_APX_V20_M8.7.md) |
| **M8.6** | BF16 KV cache (D62 resolved). Default flipped on after TinyLlama 1.1B determinism fixture came back bit-identical; **3.2 GiB → 1.6 GiB at seq=2048** on Llama 2 13B; `ATENIA_LEGACY_F32_KV_CACHE=1` opt-out | [M8.6](./docs/HANDOFF_APX_V20_M8.6.md) |
| **M9** ⚠️ *opt-in / experimental* | INT8 W8A16 weight quantisation. Post-close fixes (cost honesty + RAM headroom against pressure) deliver a measurable end-to-end win: 13B / dev-box **18.8 s/tok default, 17.7 s/tok with `ATENIA_RAM_HEADROOM_OVERRIDE_GIB=8`** vs M8.7 baseline 20.7 s/tok (−9% / −14%). **ADR-004 not satisfied** with absmax across 12 fixture runs (best 0.516, worst 11.25); ships opt-in for the drift reason, not for the perf reason — see HANDOFF for the four follow-up tracks (FFN-only, LLM.int8, GPTQ, or pivot) | [M9](./docs/HANDOFF_APX_V20_M9.md) |
| *next active* | **TBD** — α / β / γ / δ tracks documented in [HANDOFF M9 §10](./docs/HANDOFF_APX_V20_M9.md#10-open-issues--how-to-resume) | — |

**Later:**

- **v21** — Emergent policy decisions: production guards and policies consume SignalBus output to shape real execution paths
- **v22** — Multi-backend foundation: vendor-neutral abstraction for hardware probes and kernel compilation (NVIDIA + Intel iGPU as first coexistence target, via DXGI on Windows)
- **v23** — ROCm backend (AMD)
- **v24** — Metal backend (Apple Silicon)
- **v25** — Distributed execution, autonomous runtime

Full, current roadmap: [ROADMAP.md](./ROADMAP.md).

---

## 🔬 Running the Code

Atenia Engine compiles with Rust stable (2024 edition or later) and requires no external ML frameworks.

```bash
cargo build --release
cargo test
```

The build script auto-detects CUDA Toolkit and MSVC BuildTools installation paths. If detection fails (CUDA installed in a non-standard location, multiple MSVC versions), override via the `CUDA_PATH` and `MSVC_TOOLS_PATH` environment variables.

### Test coverage

The repository ships **~1200 `#[test]` functions across ~370 test files**, covering:

- Tensor operations and autograd correctness
- Graph construction and execution (CPU + CUDA numerical equivalence where applicable)
- Gradient checking via central finite differences
- F64 numerical validation per [ADR-004](./docs/decisions/ADR-004-f64-reference-as-default.md) on the four production LLMs
- BF16 storage round-trip equivalence with the precision-floor spike (regression gate)
- Sharded safetensors loading on real multi-file checkpoints
- SignalBus producing `GuardConditions` / `PolicyEvidenceSnapshot` from real probes
- AMG executor aborting cleanly on guard verdicts and resuming when conditions are clean

See [docs/TESTS.md](./docs/TESTS.md) and [tests/README.md](./tests/README.md) for methodology and categorisation.

> [!WARNING]
> **Note on test methodology.**  
> Some tests from earlier APX versions use controlled harnesses that 
> inject runtime conditions (memory pressure, policy competition) to 
> exercise the scaffolding. These are being rewritten to derive signals 
> from the engine itself as part of the v18+ roadmap; several were 
> completed in v18 (memory pressure detection) and v19 (SignalBus-driven 
> integration tests).

---

## ❌ What Atenia Engine Is Not

- ❌ Not a machine learning framework
- ❌ Not a compiler or graph optimizer
- ❌ Does not modify model semantics
- ❌ Does not require retraining
- ❌ Does not assume ideal hardware

Atenia is designed to sit **below** ML frameworks and **above** raw hardware execution — addressing a layer they largely ignore: **execution stability**.

---

## 🛠 Implementation

- 🦀 Implemented in **Rust** (2024 edition)
- 🔒 Deterministic execution behavior
- 🧵 Explicit memory and concurrency control
- 🚫 No garbage collection
- 🧩 No opaque runtime adaptation

---

## 📚 Documentation

**Roadmap and architecture**

- [ROADMAP.md](./ROADMAP.md) — Versioned roadmap with milestones and design constraints
- [docs/APX.md](./docs/APX.md) — Per-version APX notes
- [docs/ARCHITECTURE_REACTION_STRATEGIES.md](./docs/ARCHITECTURE_REACTION_STRATEGIES.md) — Reaction strategies (APX v21+)
- [docs/RESEARCH_INTEL_APIS.md](./docs/RESEARCH_INTEL_APIS.md) — Multi-vendor GPU API research (APX v22+)

**Milestone closing notes (HANDOFFs)**

- [HANDOFF M3](./docs/HANDOFF_APX_V20_M3.md) — Reactive executor + GPU storage + M3-e migration
- [HANDOFF M4](./docs/HANDOFF_APX_V20_M4.md) — Safetensors loader
- [HANDOFF M4.5](./docs/HANDOFF_APX_V20_M4.5.md) — TinyLlama end-to-end
- [HANDOFF M4.6](./docs/HANDOFF_APX_V20_M4.6.md) — Llama-family expansion + F64 validation methodology
- [HANDOFF M4.7](./docs/HANDOFF_APX_V20_M4.7.md) — Beyond-VRAM killer demo (Llama 2 13B Chat on 8 GB VRAM + 32 GB RAM, transparency contract closed)
- [HANDOFF M4.8](./docs/HANDOFF_APX_V20_M4.8.md) — Performance optimisation (3.5× on 13B Mode A; 49.5× on the production matmul shape; vendor-agnostic AVX2/FMA + matrixmultiply)
- [HANDOFF M4.9](./docs/HANDOFF_APX_V20_M4.9.md) — Public CLI demo (`atenia run --mode c` reproduces the killer demo in 6.9 min via one command)
- [HANDOFF M5](./docs/HANDOFF_APX_V20_M5.md) — Tokenizer + KV cache + token-by-token generation (`atenia generate` ships; Llama 2 13B Chat answers conversationally; Arc-shared weights at 24.24 GiB)
- [HANDOFF M6](./docs/HANDOFF_APX_V20_M6.md) — Tier-aware GPU loader (VRAM → RAM → NVMe planner; 1.46× speedup on Llama 2 7B Chat; bit-identical output)
- [HANDOFF M7](./docs/HANDOFF_APX_V20_M7.md) — 13B-friendly tiers (Disk fast-path + adaptive RAM headroom; Llama 2 13B Chat end-to-end with 239 tensors on NVMe, 7.36 GiB RAM headroom, no BSOD)
- [HANDOFF M8](./docs/HANDOFF_APX_V20_M8.md) — BF16-resident VRAM kernels (Path B: BF16 storage + F32 upcast per-matmul; 1.31× on 7B, 1.36× on 13B; ADR-004 4-model F64 validation passes with margin 21–12,500×)

**Models layout**

- [docs/MODELS_LAYOUT.md](./docs/MODELS_LAYOUT.md) — canonical model checkpoint paths (operator-local, git-ignored) + env-var → path mapping for the F64 4-model validation tests

**Architectural Decision Records (ADRs)**

- [docs/decisions/](./docs/decisions/) — ADR-001 through ADR-004
- [ADR-004](./docs/decisions/ADR-004-f64-reference-as-default.md) — F64 reference as primary numerical validation methodology

**Testing**

- [docs/TESTS.md](./docs/TESTS.md) — Test methodology and categorisation
- Subversion READMEs: [`src/v13`](./src/v13/README.md), [`src/v14`](./src/v14/README.md), [`src/v15`](./src/v15/README.md), [`src/v16`](./src/v16/README.md), [`src/v17`](./src/v17/README.md)

---

## 🧾 Intellectual Property

- **Patent:** USPTO Provisional Application **63/941,875** (filed December 16, 2025)
- **License:** Apache License 2.0 (with explicit patent grant)
- **Author:** Guillermo Alonso Albella — GAAIA Labs (Independent Research Initiative)

Apache 2.0 allows broad adoption, modification, and commercial use while providing explicit patent protection.

---

## 📄 Research Paper

The initial research preprint has been withdrawn while the implementation matures to fully back its empirical claims.

See [`paper/README.md`](./paper/README.md) for details. A revised version with end-to-end empirical validation will be published once runtime signal integration and real model loading (APX v20+) are complete.

---

## 🤝 Contributing

This is a research-in-progress. Contributions, issues, and technical discussions are welcome — especially from people with experience in:

- GPU runtime systems and CUDA / ROCm low-level APIs
- Memory management and OOM prevention
- Adaptive scheduling and execution policies
- Systems research and MLSys

Open an issue or reach out if you want to collaborate on any specific layer.

---

## 🌐 Links

- 🌍 **Website:** [ateniaengine.com](https://ateniaengine.com)
- 💾 **Repository:** [github.com/AteniaEngine/ateniaengine](https://github.com/AteniaEngine/ateniaengine)
- 🧾 **Zenodo archive:** [10.5281/zenodo.17970198](https://doi.org/10.5281/zenodo.17970198)

---

## 👤 Author

**Guillermo Alonso Albella**  
GAAIA Labs — Independent Research Initiative

---

## About this project's development

Atenia Engine is developed by Guillermo Alonso Albella with significant use of AI code generation tools — primarily Claude and Claude Code from Anthropic. This section exists because I think transparency about how a project is built matters, and because "made with AI assistance" tends to be either hidden or oversold in most projects.

Here's how it actually works:

**What I decide:**
- Architecture and design choices
- Trade-offs between approaches
- What gets implemented and in what order
- Which ideas are worth pursuing and which aren't
- Code review of every change before merge

**What the AI tools execute:**
- Implementation of approved designs
- Routine refactors and test writing
- Research into specific technical options
- Pattern-matching across existing codebases

**What we do together:**
- Reasoning about complex trade-offs
- Debugging specific issues
- Evaluating alternatives when the path isn't clear

Every commit on this repo passes through my review. The AI tools don't merge autonomously. If you see a questionable decision, it's mine, not the AI's.

This workflow lets one person work at the scope this project requires. The architectural thinking, user-facing decisions, and quality standards are human. The volume of boilerplate and research that would otherwise block progress is handled by tools.

If you have thoughts about this approach, open an issue. I'm more interested in honest feedback than in pretending this was written by hand.

---

## 🧠 Final Note

This README does not try to sell.

It states a position — honestly, including what is built and what is still being built.

**And that's what makes it real.**
