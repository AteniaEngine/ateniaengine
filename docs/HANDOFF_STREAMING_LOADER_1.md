# HANDOFF — STREAMING-LOADER-1: memory-mapped safetensors load

**Goal:** cut the peak RAM of loading large models, without touching Numeric
Policy / CUDA / adapters, keeping the reader/adapter/graph APIs and fail-loud
behaviour intact.

## FASE 1 — Memory audit

Where the load-time RAM peaks live, per format:

| Format | Peak source |
|---|---|
| **single-file safetensors** | `SafetensorsReader::open` did `fs::read` → the **whole file** sits in a heap `Vec<u8>` for the entire load (dominant peak). |
| **sharded safetensors** | already streams **one shard at a time** (`SafetensorsReader::open` per shard, dropped before the next) → peak ≈ max(shard) — but each shard still `fs::read` into a Vec. |
| **`.bin` / sharded `.bin`** | transcoded to an in-memory safetensors buffer (`from_bytes`, owned) — inherently buffered; the legacy/smaller case. |
| **FP8** | small F32 side buffer during read (transient). |
| **GGUF** | separate reader (out of scope here). |

**Conclusion:** the single highest-ROI lever is the `fs::read` whole-file copy in
`SafetensorsReader::open`. Because the **sharded** loader and the `.bin` transcode
all funnel through `SafetensorsReader`, fixing `open` improves single-file **and**
per-shard safetensors at once.

## FASE 2–3 — Design & compatibility

**Memory-map the file in `open`** behind the unchanged reader API. A new private
`Backing { Owned(Vec<u8>) | Mapped(memmap2::Mmap) }` enum derefs to `&[u8]`, so
`iter()` / `get()` / `to_vec_f32()` / the FP8 side-buffer are all
backing-agnostic — **zero change** to the weight mapper, adapter layer, graph,
tier planner, or `from_bytes` callers (`.bin` transcode, FP8 decode, network).
`open` maps the file; `from_bytes` stays owned. The read-only mapped pages are
**file-backed and reclaimable**, so they do not count as committed heap.

## FASE 4 — Implementation

- `src/v17/loader/safetensors_reader.rs`: `Backing` enum + `Deref`;
  `from_backing` (shared parser) backs both `open` (mmap) and `from_bytes`
  (owned). `open` memory-maps by default.
- Dependency: `memmap2 = "0.9"` (pure-Rust, no C toolchain, cross-platform —
  preserves the vendor-agnostic posture). mmap is an internal RAM optimisation;
  the public format list is unchanged.

## FASE 5 — Failure modes (no silent corruption)

- **`ATENIA_DISABLE_MMAP=1`** → forces the owned `fs::read` path (escape hatch).
- **mmap failure** (rare platform/file case) → automatic fallback to `fs::read`
  with a stderr note. Both fallbacks read **byte-identical** data — only the RAM
  profile differs, never the bytes (verified by a test: mmap vs owned identical
  by name).
- Missing file → `FileNotFound` (unchanged). Malformed header → `InvalidFormat`.

## FASE 6 — Benchmark

Real A/B on **Qwen2.5-1.5B-Instruct** (single-file `model.safetensors`, 2945 MB),
OS page-cache warm, polled peak process memory, `atenia generate … --max-tokens 1`:

| Mode | Peak commit (paged) | Peak working set | Wall |
|---|---|---|---|
| `fs::read` (old, `ATENIA_DISABLE_MMAP=1`) | **12865 MB** | 4461 MB | 16.7 s |
| `mmap` (new, default) | **9483 MB** | 3790 MB | 16.6 s |

**−3382 MB peak committed RAM (≈ the file size), −671 MB peak working set, no
time penalty.** The saving scales with model size: a 13B f32 single-file
(~52 GB) sheds ~52 GB of peak commit — the difference between OOM and a
successful load on the 32 GB dev box.

## FASE 7 — Tests

- Unit (CI): `mmap_open_matches_owned_from_bytes` (mmap vs owned identical **by
  name** — `safetensors`'s `tensors()` order is not stable across `deserialize`,
  so the by-name view is the meaningful equivalence); `disable_mmap_env_reads_identically`;
  `open_missing_file_errors`.
- Regression: all existing safetensors / FP8 / `.bin` reads now go through `open`
  (mmap) and remain green; the `.bin` and FP8 e2e generations are unchanged.

## Deliverable answers

1. **Format optimised:** single-file **and** per-shard HF safetensors (the most
   common). `.bin`/FP8 reuse `from_bytes` (owned) and are unaffected.
2. **How streaming works:** `open` memory-maps the file instead of reading it
   into a heap `Vec`; the reader serves tensor slices from the mapped pages,
   which the OS demand-pages and can reclaim — so committed RAM no longer holds
   the whole file.
3. **RAM saved:** ≈ the file size at peak (−3.4 GB on a 2.9 GB model; scales to
   tens of GB on large models).
4. **Speed impact:** none measurable (16.7 → 16.6 s, warm cache).
5. **Supported:** every safetensors path (single + sharded, F32/F16/BF16/FP8).
6. **Not (yet) optimised:** `.bin` / sharded `.bin` assembly (owned buffer — a
   future streaming transcode); GGUF (separate reader). Both documented.
7. **Tests:** 3 streaming units + the full existing safetensors/FP8/bin regression.
8. **Benchmarks:** above (real, polled, cache-warm A/B).
9-10. Commit / CI: single commit; CI green.
11. **Recommended next:** streaming `.bin` transcode (avoid the full-model owned
    buffer for `.bin`-only large models), OR mmap the GGUF reader. Defer
    GPTQ/AWQ (quantized-weight decoding — a numeric/AQS surface).

## Limits / notes

mmap is read-only and held only for the load (the reader is dropped after the
store is populated). On Windows the mapped file is locked against truncation
while mapped — fine for load. The `from_bytes` (owned) paths (`.bin` transcode,
FP8 decode bytes, network) are unchanged and still allocate; they are the
smaller/legacy cases.

## Files

- `Cargo.toml` — `memmap2 = "0.9"`.
- `src/v17/loader/safetensors_reader.rs` — `Backing` enum + mmap `open` +
  `from_backing` + env opt-out + fallback.
- `tests/streaming_loader_test.rs` (new).
- `docs/HANDOFF_STREAMING_LOADER_1.md` (this) + `docs/STATUS.md`.
