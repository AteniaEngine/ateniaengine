# ADR-005 — Fast mode (BF16 Tensor Core native) drift envelope and per-checkpoint certification

## Status

Accepted (M10.2.1).

## Context

ADR-004 established F64 as Atenia's reference truth and F32 as the production execution path, with a strict gate `max_abs_diff < 0.5` against F64 on a 4-model fixture (TinyLlama 1.1B, SmolLM2 1.7B, Qwen 2.5 1.5B, Llama 3.2 1B). Every M4.7 → M9 milestone was validated against this gate; the engine ships today with Path B M8.4c (F32 GEMM with BF16 weight upcast) holding drift between 4.0 × 10⁻⁵ and 2.4 × 10⁻² across the fixture.

ADR-004 §"Trigger to Revisit" anticipated the introduction of new precision modes:

> When a new precision mode is introduced, add a new ADR with the drift envelope evidence.

M10.2.1 introduces **fast mode** (`ATENIA_FAST_MODE=1`), a BF16-Tensor-Core native execution path that:

- Stores weights as BF16 in VRAM (M4.7.2 / M8.3 — unchanged from certified mode).
- Casts the F32 activation to BF16 host-side before the H→D upload.
- Runs `cublasGemmEx` with `CUDA_R_16BF` for both inputs, `CUDA_R_32F` accumulate, `CUBLAS_COMPUTE_32F` — engaging the Ada/Hopper Tensor Cores natively.

This path is numerically equivalent to the M8.4-original path that M8.4c rejected for ADR-004 strict reasons. The upcast that Path B M8.4c performs is the only thing standing between Atenia's certified numerics and the industry-standard BF16-TC profile that PyTorch / vLLM / llama.cpp / TGI run by default.

The strategic reframe (see ROADMAP §"Numeric contract strategy"): the F64 fixture is more valuable as **per-checkpoint certification data** versioned with the model than as a runtime gate. Fast mode is the first execution path Atenia ships outside ADR-004 strict; this ADR documents its envelope and the certification policy that replaces blanket-runtime-guarantee correctness.

## Decision

1. **Fast mode is opt-in, never default**, gated by `ATENIA_FAST_MODE=1`.
2. **Fast mode does not satisfy ADR-004 strict** (`max_abs_diff < 0.5` vs F64) on every model — by construction. The envelope below documents the per-model drift on the M4.6 fixture; operators who enable fast mode are accepting the documented profile.
3. **Certified mode (default) remains the ADR-004 strict path**. Path B M8.4c with `CUBLAS_COMPUTE_32F_FAST_TF32` (M10.2.0) is the production default; the 4-model F64 fixture continues to gate every code change against ADR-004 ≤ 0.5.
4. **The M4.6 fixture is dual-mode aware** (M10.2.1): under `ATENIA_FAST_MODE=1` the fixture documents drift and argmax in stderr but does not panic on the strict gates. Under default (certified) it asserts ADR-004 strict as before.
5. **Per-checkpoint certification supersedes runtime guarantee**. M10.3 (next) extends this ADR with a `<model>.numcert.json` manifest versioned with each checkpoint, prescribing per-tensor precision policy and the resulting end-to-end drift bound.

## Empirical envelope (M10.2.1)

Measured by `tests/m8_5_full_family_validation_test.rs` under `ATENIA_M8_BF16_KERNEL=1 ATENIA_FAST_MODE=1` on the dev box (RTX 4070 Laptop, 8 GB VRAM, 32 GB RAM, Windows 11). Counter contract: `vram_bf16_native_matmul_count` advanced by 7 × num_layers per model (112 / 154 / 168 / 196), `vram_bf16_matmul_count` (certified counter) stayed at 0 — confirming the dispatcher routed every BF16 matmul through the native fast kernel.

| Model               | Drift fast mode | Drift certified (Path B + TF32) | Argmax fast | Argmax certified | ADR-004 fast | ADR-004 certified |
|---------------------|----------------:|---------------------------------:|-------------|------------------|--------------|--------------------|
| TinyLlama 1.1B Chat | **0.901545**    | 0.076039                         | 4/4         | 4/4              | ❌ 1.8× over | ✓ 6.6× margin     |
| SmolLM2 1.7B        | **2.331949**    | 0.217142                         | 4/4         | 4/4              | ❌ 4.7× over | ✓ 2.3× margin     |
| Qwen 2.5 1.5B       | **0.184907**    | 0.022496                         | 4/4         | 4/4              | ✓ 2.7× margin | ✓ 22× margin     |
| Llama 3.2 1B Inst.  | **0.268043**    | 0.009715                         | **3/4**     | 4/4              | ✓ 1.9× margin | ✓ 51× margin     |

The drift values reproduce the M8.4-original measurements (HANDOFF M8 §Headline) bit-for-bit — confirming that the fast kernel produces the same numerics as the path M8.4c rejected. The Llama 3.2 position-2 argmax flip (id 171 in fast vs id 863 in F64) was also a known M8.4-original signature; under fast mode it is back, and operators must accept this.

End-to-end coherence on Llama 2 13B Chat (no F64 reference fixture due to 70-billion-element weight footprint): the same 20-token greedy decode as the M10.0 baseline ("Tell me about the history of Rome" → "Rome, the Eternal City, has a rich and complex history that spans over two and") produced **bit-identical token IDs** under fast mode vs certified. The cumulative drift across 13B parameters and 32 layers stayed below the argmax-flip threshold for this prompt at greedy temperature. This is a single sample point, not a statistical claim; M10.3's per-checkpoint certification will provide the systematic envelope for production scale.

## Performance envelope

Llama 2 13B Chat smoke (`ATENIA_M8_7_ENABLED=1 ATENIA_DISK_TIER_DIR=D:/atenia-cache ATENIA_RAM_HEADROOM_OVERRIDE_GIB=8`):

| Mode        | s/tok  | vs M10.0 baseline (17.93 s/tok) | vs M10.2.0 certified (17.30 s/tok) |
|-------------|-------:|--------------------------------:|------------------------------------:|
| Certified (M10.2.0 — Path B + TF32) | 17.30 | 1.04× | — |
| **Fast (M10.2.1)**                  | **15.12** | **1.19×** | **1.14×** |

Lift on the 13B M8.7 path is modest because the wall-clock is dominated by the 50–110 disk-streamed matmuls (NVMe → PCIe → GPU staging), not by the per-matmul VRAM compute mode where fast mode acts. On a model that fits entirely in VRAM, the lift would track the M8.0 microbench measurement of ~2× over Path A on the 4 dominant decode shapes.

## Selection guidance

Operators choose between modes per their workload:

- **Certified (default)**: F64-validated end-to-end drift bounded by ADR-004 (`< 0.5`). Use for any output whose numerical equivalence to F64 ground truth must be defensible — research, scientific computation on LLM outputs, audited deployments.
- **Fast (`ATENIA_FAST_MODE=1`)**: industry-standard BF16-TC drift profile. Use for chat / generation workloads where argmax-equivalence at near-tie positions is acceptable and the per-matmul speedup matters.

The fixture data above should be consulted before enabling fast mode on a given checkpoint. SmolLM2 1.7B is the worst-case sentinel of the M4.6 family at 2.33 drift (4.7× over the strict gate); models with similar training profiles should be expected to land in that range under fast mode.

## Trigger to revisit

- A future precision mode is added (INT8 with outlier decomposition / GPTQ / AWQ / FP8 hardware Tensor Cores at sm_100+) → new ADR documenting that mode's envelope, same template.
- M10.3 lands `<model>.numcert.json` per-checkpoint certification → this ADR is amended with a section pointing to the manifest format and the per-tensor policy semantics.
- The default flips from `certified` to `fast` (M10.2.flip, gated on broader empirical coverage) → this ADR is amended with the empirical evidence supporting the flip and the fixture's gate is revisited.

## Consequences

- The F64 fixture is no longer a single-mode gate. The strict ADR-004 contract holds under certified mode; under fast mode the fixture is a documenting probe.
- Atenia ships two execution modes, both auditable. Per-checkpoint claims must specify which mode they were measured under.
- The product framing ("correctness as a per-checkpoint certificate, not a runtime tax") is now structurally supported. The README and any external claim about Atenia's drift profile must say which mode it refers to.
- M10.3 is the natural extension: from "two modes per binary" to "per-tensor mode prescribed by the certification manifest, dispatched per-matmul in production".
