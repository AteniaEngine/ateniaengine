# HANDOFF — FP8-SAFETENSORS-1: FP8 safetensors read support

**Goal:** read FP8 safetensors checkpoints (the modern low-precision dtype gap),
without new families, without Numeric Policy / CUDA / GPTQ / AWQ, without
touching the Adapter Toolkit, reusing the existing loader and fail-loud.

## FASE 1 — FP8 audit

Two 8-bit float formats appear in the safetensors ecosystem (OCP FP8), both in
the `safetensors` crate's `Dtype`:
- **`F8_E4M3`** (1 sign · 4 exp, bias 7 · 3 mantissa) — the `e4m3fn` variant:
  **no infinities**, single NaN (`S.1111.111`), max finite **448**. The dominant
  FP8 *weight* format (PyTorch `float8_e4m3fn`, NVIDIA Transformer Engine,
  DeepSeek-V3 weights).
- **`F8_E5M2`** (1 sign · 5 exp, bias 15 · 2 mantissa) — IEEE-like, has ±inf /
  NaN, max finite **57344**. More common for gradients/activations.

## FASE 2 — Compatibility analysis

- The engine `DType` enum already carried a placeholder `FP8` variant (1 byte)
  and `map_dtype` already mapped `F8_E4M3 | F8_E5M2 → DType::FP8`, but
  `TensorEntry::to_vec_f32` rejected it (`"FP8 decode not planned for M4 scope"`).
- Nothing else in the stack understands FP8 (no kernels, no graph dtype) — and
  per the rules it must not (no Numeric Policy, no CUDA). So FP8 must be
  **decoded to F32 at read time**, exactly like BF16/F16 are upcast.

## FASE 3 — Design (decode-at-read, no DType churn)

**`SafetensorsReader` decodes FP8 → F32 at construction** into a side `decoded`
buffer and presents the tensor as a plain **F32** entry. The rest of the stack
(weight mapper, graph, kernels, tier planner, adapters) sees only F32 — **zero**
changes outside the reader, no new compute dtype, F32/F16/BF16 paths untouched.
Because both the single-file and sharded loaders go through `SafetensorsReader`,
both get FP8 support for free (and so does the `.bin` transcode path, which ends
in a `SafetensorsReader`).

## FASE 4 — Implementation

- `src/v17/loader/safetensors_reader.rs`:
  - `fp8_e4m3_to_f32` / `fp8_e5m2_to_f32` — exact OCP decoders (subnormals,
    NaN/inf handled; FP8→F32 widening is lossless).
  - Construction loop special-cases `F8_E4M3 | F8_E5M2`: validates the body is
    1 byte/element, decodes into the side `decoded: Vec<u8>` buffer, and records
    an entry with `dtype = F32`, `from_decoded = true`. `iter()` / `get()` slice
    from `decoded` vs `bytes` per entry. Empty `decoded` (zero overhead) when a
    file has no FP8 tensors.
  - **Fail-loud:** FP8 body length ≠ elems → error; genuinely unsupported
    safetensors dtypes (I64/BOOL/F64/…) still error via `map_dtype`.

## FASE 5–7 — Validation & real evidence

- **Unit (CI):** FP8 decoders verified against hand-computed known values
  (0/±1/max/min/subnormal/NaN/inf) for both formats; all finite E4M3 values
  proven exactly F32-representable.
- **Fixture (CI):** real FP8 safetensors fixtures (`tests/fixtures/fp8/`,
  produced by `torch.float8_e4m3fn` / `e5m2` + `safetensors.torch`) decode
  **bit-identical** to PyTorch's own `fp8.to(float32)` upcast (committed F32
  reference). The reader presents the FP8 tensors as F32 (12 elems × 4 bytes).
- **Real model (e2e, `#[ignore]`):** SmolLM2-135M with **every weight stored as
  FP8 E4M3** loads through the full pipeline and generates **coherent text**
  ("…the city of Paris, a vibrant metropolis known for its historical") — FP8 is
  lossy, so the bar is "loads and runs coherently", not text parity.

## Deliverable answers

1. **FP8 variants supported:** `F8_E4M3` (e4m3fn) and `F8_E5M2`, read from
   safetensors, decoded to F32.
2. **How implemented:** decode-at-read in `SafetensorsReader` into a side buffer;
   FP8 surfaces as F32 downstream; no DType-compute / kernel / adapter changes.
3. **Supported cases:** single-file and sharded safetensors (and `.bin` →
   safetensors) containing E4M3/E5M2 tensors, mixed freely with F32/F16/BF16;
   any supported family.
4. **Rejected cases:** FP8 body-length mismatch → error; other safetensors
   dtypes (int/bool/F64) → error; no silent fallback. FP8 is decoded losslessly
   to F32 — Atenia does not *compute* in FP8 (no Numeric Policy / CUDA touched).
5. **Tests added:** 3 decoder units + 3 fixture reads (bit-exact vs torch) +
   1 e2e `#[ignore]`.
6. **Real validation:** bit-exact decode vs PyTorch; a real all-FP8 SmolLM2-135M
   loads and generates coherently end to end.
7. **Coverage gained:** FP8 safetensors checkpoints (the modern low-precision
   distribution dtype) now load directly — combined with the FORMAT-INTAKE work,
   the container/dtype intake surface is broad (safetensors F32/F16/BF16/FP8,
   sharded, GGUF, single + sharded `.bin`).
8-9. Commit / CI: single commit; CI green (decoder units + fixture reads run in
   CI; the model e2e is `#[ignore]`).
10. **Recommended next:** `.pth` (legacy single-pickle torch), OR a streaming
   `.bin` path for very large `.bin`-only models. Defer GPTQ/AWQ (quantized
   *weight decoding* — a numeric/AQS surface, not a container/dtype read).

## Limits / notes

FP8 tensors are decoded to F32 in RAM (a transient 4× side buffer alongside the
1-byte on-disk bytes for the reader's lifetime, then hoisted into the store as
F32 / BF16 like any other weight). Atenia reads FP8; it does not compute in FP8.
No real public FP8 *full model* of a supported family was available locally
(DeepSeek-V3 FP8 is huge + out of scope), so the end-to-end evidence is a
locally-converted SmolLM2-135M — documented as such, not presented as a shipped
checkpoint.

## Files

- `src/v17/loader/safetensors_reader.rs` — FP8 decoders + decode-at-read side
  buffer + units.
- `src/cli/diagnostics.rs` — `atenia capabilities` lists FP8.
- `tests/fp8_safetensors_test.rs` (new) + `tests/fixtures/fp8/` (new).
- `docs/HANDOFF_FP8_SAFETENSORS_1.md` (this) + `docs/STATUS.md` +
  `docs/MODEL_COVERAGE_EXECUTIVE_AUDIT.md`.
