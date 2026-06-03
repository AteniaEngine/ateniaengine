# HANDOFF — FORMAT-INTAKE-2: sharded PyTorch `.bin`

**Goal:** load multi-file `pytorch_model-0000k-of-000NN.bin` checkpoints, closing
the single-file-only limitation of FORMAT-INTAKE-1 — without new family loaders,
reusing the FI-1 transcode and the existing safetensors shard-index parser.

## FASE 1 — Sharded `.bin` audit

A sharded HF `.bin` checkpoint ships N shard files plus
`pytorch_model.bin.index.json`, whose schema is **identical** to the safetensors
index already parsed by `src/v17/loader/shard_index.rs`:
```json
{ "metadata": { "total_size": … },
  "weight_map": { "model.embed_tokens.weight": "pytorch_model-00001-of-00002.bin", … } }
```
`weight_map` is tensor-name → shard-filename, injective by construction.

## FASE 2 — Design (reuse-first)

`sharded .bin → transcode each shard (FI-1) → assemble all tensors → one
in-memory safetensors buffer → SafetensorsReader::from_bytes → existing
pipeline`. The adapter layer, weight mapper, transforms and tier planning are
**unchanged**; only the weight *source* is assembled from shards.

## FASE 3–4 — Shard resolution & assembly

- **`ShardIndex::from_file`** (reused) parses `pytorch_model.bin.index.json`
  (rejects malformed / empty `weight_map` and duplicate index keys).
- **`pytorch_bin::bin_to_tensors`** — FI-1's transcode refactored to expose the
  per-shard materialised tensors (`BinTensor { name, dtype, shape, bytes }`);
  `transcode_bin_to_safetensors` is now `serialize_tensors(bin_to_tensors(…))`.
- **`pytorch_bin::transcode_sharded_bin_to_safetensors(index_path)`** — reads
  each distinct shard, transcodes it, assembles into one `BTreeMap<name,
  BinTensor>`, cross-checks against `weight_map`, and serialises to safetensors.
- **`pipeline.rs`** — the `bin_index` branch (previously a refusal) now calls the
  sharded transcode and routes the resulting in-memory `SafetensorsReader`
  through the existing single-file path; logs `[ATENIA] format: PyTorch sharded
  .bin intake — assembled … → in-memory safetensors`.

## FASE 5 — Failure modes (fail-loud, no silent fallback)

- **missing / unreadable shard** → error naming the shard path;
- **duplicate tensor across shards** (same name in two shard files) → error;
- **tensor declared in `weight_map` but absent from the shards** → error;
- **tensor present in shards but not declared in `weight_map`** → error;
- **malformed / empty `weight_map`, duplicate index keys** → error (via `ShardIndex`);
- plus every per-shard FI-1 guard (compressed-zip, ZIP64, big-endian,
  non-contiguous / storage-sharing, Double/Int/Bool dtype, legacy non-zip).

## FASE 6–7 — Tests & real validation

- **Round-trip (CI):** committed 2-shard fixture
  (`tests/fixtures/pytorch_bin_sharded/`) transcodes to a safetensors buffer
  **byte-identical** to a reference safetensors assembled from the same tensors
  (F32 2D, F16 2D, BF16 1D, F32 2D split across the two shards).
- **Negative (CI):** missing shard; weight_map ghost tensor; undeclared shard
  tensor; duplicate tensor across shards — all rejected with specific messages.
- **End-to-end (real model):** SmolLM2-135M-Instruct split into a **2-shard
  `.bin`** loads through the full pipeline and generates **identical greedy
  text** to the safetensors original ("…the city of Paris, a city of 2.5
  million inhabitants. Paris"). (`#[ignore]`; same test as FI-1, the pipeline
  auto-detects single vs sharded.)

## Deliverable answers

1. **How sharded support works:** parse `pytorch_model.bin.index.json` (reused
   `ShardIndex`), transcode each shard via the FI-1 path, assemble all tensors
   into one in-memory safetensors buffer with fail-loud consistency checks, hand
   it to the unchanged `SafetensorsReader`/adapter/pipeline.
2. **Exact formats supported:** `pytorch_model.bin.index.json` + N shard files
   `pytorch_model-0000k-of-000NN.bin` (any shard names the index lists), torch
   `save` ZIP+pickle, contiguous F32/F16/BF16 tensors, little-endian.
3. **Supported cases:** 2..N shards; any tensor→shard partition; tensors of
   F32/F16/BF16; single-file `.bin` still works (FI-1); safetensors/GGUF still
   preferred when present.
4. **Rejected cases:** missing shard, duplicate tensor across shards, weight_map
   ghost tensor, undeclared shard tensor, malformed/empty weight_map, plus all
   FI-1 per-shard rejections — every one a clear error, no silent fallback.
5. **Tests added:** 1 round-trip + 4 negative (CI) + the shared e2e parity test
   now exercised on a sharded dir.
6. **Real validation:** SmolLM2-135M 2-shard `.bin` ≡ safetensors greedy text.
7. **Coverage gained:** multi-shard `.bin` checkpoints (the 7B+ legacy format)
   now load directly; combined with FI-1, **all single- and multi-file `.bin`
   checkpoints of supported families** load with no external conversion.
8-9. Commit / CI: single commit; CI green (round-trip + negatives run in CI; e2e
   `#[ignore]`).
10. **Recommended next:** **FP8 safetensors** (modern checkpoints increasingly
   ship FP8 — a real dtype-decode gap) OR `.pth`. Defer GPTQ/AWQ (quantized-
   weight *decoding* — a numeric surface, not a container format).

## Limits / notes

Assembling all shards holds the model bytes in RAM once (plus a per-shard
transient) — fine for the model sizes that ship as `.bin`; very large checkpoints
ship safetensors (streamed shard-by-shard). A future streaming `.bin` path could
avoid the single-buffer peak if a large `.bin`-only model appears.

## Files

- `src/v17/loader/pytorch_bin.rs` — `bin_to_tensors` + `serialize_tensors` +
  `transcode_sharded_bin_to_safetensors`.
- `src/nn/llama/pipeline.rs` — `bin_index` branch now assembles shards.
- `tests/format_intake_bin_test.rs` — sharded round-trip + 4 negatives.
- `tests/fixtures/pytorch_bin_sharded/` (new) — 2-shard fixture + index +
  assembled reference.
- `docs/HANDOFF_FORMAT_INTAKE_2.md` (this) + `docs/STATUS.md` +
  `docs/MODEL_COVERAGE_EXECUTIVE_AUDIT.md`.
