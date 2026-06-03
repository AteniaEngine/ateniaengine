# HANDOFF — FORMAT-INTAKE-1: PyTorch `.bin` intake

**Goal:** widen *format* coverage (the audit's #2 gap) without new families,
without touching Numeric Policy / CUDA / MoE / adapters, reusing the existing
load path. Many otherwise-supported checkpoints were unloadable purely because
they ship as `pytorch_model.bin` rather than safetensors. This milestone adds a
**single-file PyTorch `.bin` loader**.

## FASE 1–3 — Audit, ecosystem, gap

Format support before this milestone (from `MODEL_COVERAGE_EXECUTIVE_AUDIT.md`
+ code): **safetensors** (single + sharded) and **GGUF** (read) supported;
**PyTorch `.bin`/`.pth`** absent; **GPTQ/AWQ/bitsandbytes** absent (GPTQ/AWQ
exist only as experimental *post-load* quant policies in `src/quant/`, not
checkpoint loaders); **FP8** absent. Highest-coverage / lowest-risk addition:
**`.bin`** — it is the legacy HF distribution format, reads to the same f32/f16/
bf16 weights safetensors already handles (no new numeric/quant surface, unlike
GPTQ/AWQ), and needs no family changes.

## FASE 4 — Design (reuse-first)

A `torch.save` `.bin` is a **ZIP archive** (stored/uncompressed) of a Python
**pickle** state dict + raw little-endian tensor storages. Instead of teaching
the whole pipeline a new on-disk format, `.bin` is **transcoded to an in-memory
safetensors buffer**, consumed by the existing
`SafetensorsReader::from_bytes` — so the weight mapper, **adapter layer**
(family + adapter resolution + validation, all reused per rule 6), load
transforms, and tier planning are **unchanged**. Detection slots into
`pipeline.rs` *after* GGUF/safetensors (safetensors is always preferred).

## FASE 5 — Implementation

- **`src/v17/loader/pytorch_bin.rs`** (new) — `transcode_bin_to_safetensors(&[u8])
  -> Result<Vec<u8>>`:
  - **Hand-rolled minimal ZIP reader** (STORED entries only — `torch.save` never
    compresses; no new crate dependency, preserving the dependency-light /
    vendor-agnostic posture). ZIP64 + compressed entries → clear error.
  - **Restricted pickle VM** implementing only the opcode subset `torch.save`
    emits for a state dict, accepting a **whitelist** of globals
    (`collections.OrderedDict`, `torch._utils._rebuild_tensor_v2` /
    `_rebuild_parameter`, `torch.{Float,Half,BFloat16}Storage`). Any other
    global/opcode is a hard error — **never executed** (no RCE surface), never
    silently skipped.
  - Reads each tensor's contiguous storage, validates `offset==0` + contiguous
    stride + `numel`, and serialises to safetensors via the `safetensors` crate.
- **`src/nn/llama/pipeline.rs`** — additive `.bin` branch: when no GGUF/
  safetensors is present, a single `pytorch_model.bin` is transcoded and routed
  through the existing single-file reader path (both the tier-aware and legacy
  loaders); logs `[ATENIA] format: PyTorch .bin intake — transcoded … → in-memory
  safetensors`.
- **`src/cli/diagnostics.rs`** — `atenia capabilities` now lists PyTorch `.bin`.

## FASE 6–7 — Validation & failure modes

- **Round-trip (CI):** `tests/format_intake_bin_test.rs` transcodes a committed
  real `torch.save` fixture (`tests/fixtures/pytorch_bin/tiny.bin`) and asserts
  it is **byte-identical** (name set, shape, dtype, raw bytes) to a reference
  safetensors saved from the same state dict — across F32 (2D), F16 (2D), BF16
  (1D), F32 (2D). Proves the ZIP reader + unpickler are correct on real torch
  output, with no torch dependency at test time.
- **End-to-end (real model):** SmolLM2-135M-Instruct converted to `.bin` loads
  through the full pipeline and generates **identical greedy text** to the
  safetensors original ("…the city of Paris, a city of 2.5 million
  inhabitants. Paris"). (`#[ignore]`, needs the local model dirs.)
- **Failure modes (no silent fallback):** truncated/non-zip → error; compressed
  zip entry → error; ZIP64 → error; big-endian → error; non-contiguous /
  storage-sharing view → error; Double/Long/Int/Bool storage → error naming the
  tensor; **sharded `pytorch_model.bin.index.json` → refused** with a "convert
  to safetensors" message; unknown pickle global/opcode → refused. safetensors
  is always preferred when present; `.bin` never shadows it.

## FASE 8 — Tests

- `pytorch_bin` units (4, CI): non-zip reject, empty reject, contiguous-stride,
  storage-dtype whitelist.
- `format_intake_bin_test` (3 CI + 1 `#[ignore]`): byte-exact round-trip;
  truncated `.bin` fail-loud; non-`.bin` fail-loud; e2e generation parity.

## Deliverable answers

1. **Format added:** single-file PyTorch `.bin` (`pytorch_model.bin`, the
   `torch.save` ZIP+pickle format).
2. **Why:** highest coverage-per-risk — the legacy HF format for otherwise-
   supported families; reads to the same f32/f16/bf16 weights safetensors
   handles (no new numeric/quant surface, unlike GPTQ/AWQ); no family changes.
3. **Coverage gained:** every supported-family checkpoint distributed as a
   single-file `.bin` now loads directly (no external conversion), proven
   bit-identical to safetensors end to end.
4. **Limitations:** single-file only (sharded `.bin` refused with guidance);
   contiguous F32/F16/BF16 tensors only (Double/Int/Bool, non-contiguous /
   storage-sharing views, compressed-zip, ZIP64, big-endian, legacy non-zip
   torch<1.6 → fail-loud); transcode holds the model bytes in RAM once.
5. **Still pending:** sharded `.bin`; GPTQ / AWQ / bitsandbytes (quantized —
   need weight *decoding*, a numeric concern); FP8 safetensors; `.pth`.
6. **Tests added:** see FASE 8 (4 units + 3 CI integration + 1 e2e ignored).
7. **Results:** all green; e2e `.bin`↔safetensors generation identical.
8-9. Commit / CI: single commit; CI green (units + round-trip run in CI; e2e is
   `#[ignore]`).
10. **Recommended next:** **sharded `.bin`** (concatenate transcoded shards —
   small increment, closes the 7B+ `.bin` hole) OR **FP8 safetensors** (modern
   checkpoints increasingly ship FP8). Defer GPTQ/AWQ (they are a *quantized-
   weight decoding* problem — numeric surface — not a container-format problem).

## Files

- `src/v17/loader/pytorch_bin.rs` (new) + `src/v17/loader/mod.rs` (`pub mod`).
- `src/nn/llama/pipeline.rs` — additive `.bin` detection + single-file routing.
- `src/cli/diagnostics.rs` — capabilities lists `.bin`.
- `tests/format_intake_bin_test.rs` (new) +
  `tests/fixtures/pytorch_bin/{tiny.bin,tiny_reference.safetensors}` (new).
- `docs/HANDOFF_FORMAT_INTAKE_1.md` (this) + `docs/STATUS.md` +
  `docs/MODEL_COVERAGE_EXECUTIVE_AUDIT.md`.
