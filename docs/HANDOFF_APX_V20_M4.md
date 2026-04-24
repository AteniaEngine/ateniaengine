# Handoff — APX v20 M4 (ModelLoader mechanics, at M4 close)

**Status at handoff**: M4 conservative plan closed. All four planned
sub-phases (M4-a through M4-d) landed in the same sprint. Loader
mechanics are functional end-to-end: Atenia can read a safetensors
file, validate its structure against a target graph, populate
parameters in-place with actionable errors on mismatch, and decode
F32 / BF16 / F16 tensors into the host-side F32 storage the engine
uses today.

M4.5 (targets against real LLM checkpoints — TinyLlama, Llama 3.2
1B) is deliberately carved off as a separate milestone. The
boundary between "the loader works" (M4) and "a specific external
model produces bit-close logits against a PyTorch reference"
(M4.5) is drawn on purpose: the M4 sub-phases can all be validated
with synthetic data and Atenia's own MiniFlux model, while M4.5
introduces architecture-specific gaps (RoPE, GQA) plus external
dependency and numerical-debugging surface.

**Last M4 commit**: `9f1ee82` (M4-d, BF16/F16 decode).

---

## What is ready

| Sub-phase | Commit | Summary |
|-----------|--------|---------|
| M4-a | `229fb25` | Safetensors reader standalone: `SafetensorsReader::from_bytes` / `open`, iterator + by-name access, F32 decode via `to_vec_f32`, metadata passthrough. Uses the official HuggingFace `safetensors` crate. 9 tests. |
| M4-b | `4d3cf5f` | MiniFlux safetensors roundtrip validation. `MiniFluxHandles` gains a parallel `param_names: Vec<String>` that exposes the logical names `register_weight` was already using internally for deterministic init seeds. End-to-end roundtrip (serialize → deserialize → bit-exact) proven on a live MiniFlux graph. 3 tests + 5 existing callers updated to match the new tuple return shapes. |
| M4-c | `e0df986` | `WeightMapper` formalizes the "safetensors tensor name → graph parameter node_id" mapping, with shape validation (new `LoaderError::ShapeMismatch`), dtype propagation, and `LoadReport { loaded, skipped, missing }`. Loose-mode by default. Constructor rejects length mismatch and duplicate names. 5 tests. |
| M4-d | `9f1ee82` | BF16 and F16 decode in `TensorEntry::to_vec_f32`. BF16 is manual (shift + `from_bits`, lossless). F16 uses the `half` crate (handles subnormals / NaN / infinities correctly). The `WeightMapper` needs no changes — once the reader stops returning `UnsupportedDType`, BF16/F16 checkpoints flow through identically to F32. 4 new tests + 1 M4-c test updated from "expect error" to "expect success". |

---

## Architectural decisions locked

Treat as invariants. Future work extends rather than re-litigates.

1. **Loader lives under `src/v17/loader/`**. The pre-existing
   skeleton (`ModelLoader`, `LoaderPolicy`, `LoaderError`,
   `MemoryMap`) is extended, not replaced. Adding `safetensors_reader.rs`
   and `weight_mapper.rs` as siblings keeps related primitives
   co-located without fragmenting across crates.

2. **API shape: struct `WeightMapper`, not a free function**. Gives
   the loader a clear place to grow (dry-run validation, strict-mode,
   alias tables, sharded-file support) without changing call-site
   signatures.

3. **`WeightMapper::from_param_names_and_ids(&[String], &[usize])`**.
   Constructor takes two index-aligned slices, not a specific
   architecture handle type. TinyLlama (M4.5) and every future model
   builder reuse the same mapper. No `from_xyz_handles`
   per-architecture constructors.

4. **Loose-mode default on `load_into`**. Tensors present in the
   reader but absent from the mapper land in `LoadReport.skipped`;
   tensors in the mapper but absent from the reader land in
   `LoadReport.missing`. Neither aborts the load. Shape mismatch
   and dtype-unsupported DO abort — those are correctness failures,
   not routine variance.

5. **`half` crate is an unconditional dependency**. Not feature-
   gated. BF16/F16 are the common case of real LLM checkpoints
   (Llama, Mistral, Qwen, TinyLlama all distribute BF16), so every
   future use of the loader needs this path available. ~30 KB
   transitive footprint.

6. **BF16 decode is hand-rolled; F16 decode delegates to `half`**.
   BF16 → F32 is a single bit-shift with no rounding, trivial and
   clear in-source. F16 → F32 involves different exponent bias and
   mantissa width plus subnormal / NaN / denormal edge cases;
   deferring to `half` avoids hand-crafting those correctly.

7. **Downcast-on-load, not native F16/BF16 storage**. A loaded
   checkpoint's bytes are converted to `Vec<f32>` at read time and
   stored that way. The host-side memory footprint of a BF16
   checkpoint therefore doubles on load (2 bytes → 4 bytes per
   element). Accepted for M4: the 32 GB RAM dev machine
   comfortably holds a 1-2 GB native checkpoint at 2-4 GB F32;
   downcast keeps the engine's kernels F32-only, avoiding a
   parallel dtype-specific code path. Native F16/BF16 storage is
   a separate post-M4 milestone (M5 or M6 candidate).

8. **`register_weight` in `src/nn/mini_flux.rs` extended with
   `&mut Vec<String>`; `register_weight` in `src/nn/transformer.rs`
   untouched**. The two copies are known-duplicated (different
   internal signatures) — unifying them is a cleanup separate from
   M4's scope. Changing both would have doubled the blast radius
   without adding validation the M4 tests exercise.

9. **`MiniFluxHandles.param_names` is index-aligned with
   `param_ids` (Variant A, parallel Vecs)**. Rejected the
   alternative of a `Vec<NamedParam { name, node_id }>` struct to
   minimize blast radius across 6 existing test call sites.
   Promoting to the structured shape is a valid future refactor;
   M4-b does not pay its cost.

10. **New `LoaderError` variants added in M4**: `InvalidFormat`,
    `UnsupportedDType`, `ShapeMismatch`. All three carry actionable
    context (tensor name, expected vs actual shape). The pre-M4
    match in `src/v17/inference/infer.rs::start_inference` was
    updated to stay exhaustive; it routes the new variants through
    the existing `InferenceError::LoadFailed` umbrella. The
    inference entrypoint is NOT yet wired to the safetensors loader
    itself — that integration is M4.5 or later.

---

## Gaps explicitly closed in M4

- Safetensors parsing (header, metadata, body offsets).
- Bit-exact Atenia-side roundtrip
  (`MiniFlux → serialize → deserialize → bit-exact`).
- Name-based mapping from safetensors names to graph parameter
  node_ids.
- Shape validation with actionable error messages.
- BF16 and F16 → F32 conversion on load.
- `LoadReport` diagnostics for partial and extra tensors.
- Duplicate / misaligned parameter name detection at mapper
  construction.

---

## Gaps explicitly NOT closed — scope deferred

Honest list of everything M4 touches the boundary of but does not
deliver. Each item has a clear next-step context rather than an
open-ended "someday".

- **Real LLM loading end-to-end**. Requires M4.5. Every M4 test
  uses synthetic data or Atenia's own MiniFlux — no external
  checkpoint has been loaded yet.

- **RoPE (Rotary Positional Embedding)**. Does not exist as a
  `NodeType`. Estimated 2-4 days (forward is a 2D rotation of
  consecutive pairs; backward is the derivative of that rotation;
  tests against the analytic formula). Required by every Llama-
  family model from Llama 1 onward.

- **LayerNorm**. Does not exist (only `RmsNorm`). Required by
  GPT-2, BERT, ViT. Not required by Llama / Mistral / Qwen. M4.5
  targets Llama-family, so LayerNorm stays deferred.

- **GQA (Grouped Query Attention)**. Llama 3+ uses 8 KV heads vs
  32 Q heads (and similar ratios). Buildable from existing
  primitives but requires a specific reshape + broadcast pattern
  in the graph builder. Deferred to post-M4.5 or whenever Llama 3
  becomes a target.

- **Tokenizer**. Out of scope. M4 / M4.5 work with pre-generated
  integer token tensors as input. HuggingFace `tokenizers` crate
  integration is post-M5.

- **KV cache for autoregressive generation**. Not needed for
  single-forward validation, which is what M4.5 targets. Needed
  when Atenia generates text token-by-token. Post-M5.

- **Native F16 / BF16 storage**. M4 downcasts on load, doubling
  memory. Native storage requires dtype-specific kernels or
  intrinsics paths — a separate milestone (M5 or M6 candidate).

- **Multi-file (sharded) safetensors**. HuggingFace distributes
  Llama-7B and larger as
  `model-00001-of-00002.safetensors` + `model.safetensors.index.json`.
  M4-a handles single-file only. TinyLlama ships single-file,
  so M4.5 can proceed without this. Easy extension when needed.

- **Causal mask as a primitive**. No helper exists; buildable
  via `Parameter` (upper-triangular -inf) + `BroadcastAdd` +
  `Softmax`. Mini_flux does not use masking. Needed for
  autoregressive LLMs in M4.5.

- **Memory-mapped reader**. `SafetensorsReader::open` reads the
  whole file into a `Vec<u8>`. `memmap2` integration is deferred
  until a real load shows the copy-into-RAM cost in the profile.
  For TinyLlama at 2 GB BF16 it may or may not matter — empirical
  question for M4.5.

- **Native safetensors writer**. M4-b uses the `safetensors` crate's
  own `serialize` writer for roundtrip tests. Atenia does not yet
  expose a helper to produce safetensors files from its own
  graphs — useful if we ever need to save trained weights. Post-M4.

- **README.md milestone sync**. Inherited as admin debt from the
  M3 handoff. The README still describes M3 as "🟡 in progress"
  and does not mention M4 at all. Stays deferred here; a single
  README-sync commit covering M3 and M4 is the appropriate follow-
  up once M4.5 starts or closes.

---

## M4.5 scope proposed (tentative)

The investigation in `HANDOFF_APX_V20_M3.md` §10 mapped four sub-
phases for M4.5, still valid:

| Sub-phase | Content |
|-----------|---------|
| M4.5-a | RoPE implementation (new `NodeType::RoPE { base_freq }`, forward + backward). Tests against analytic formula and against a reference Python implementation. |
| M4.5-b | TinyLlama graph builder: adapt `build_transformer_block` to the exact Llama topology (RoPE where appropriate, HF naming convention in `register_weight`). |
| M4.5-c | End-to-end load: download `TinyLlama-1.1B` safetensors from HuggingFace, feed through M4 loader, execute forward on a fixed token sequence. |
| M4.5-d | Numerical validation against reference (PyTorch or candle fixtures committed as npz / JSON). |

Honest estimate: **2-3 weeks of focused work**, with non-trivial
risk of numerical-mismatch debugging extending the timeline. The
classic failure modes are epsilon drift in RMSNorm vs PyTorch,
softmax stability differences, attention scaling order-of-ops
variance, and BF16 precision artifacts accumulating through
many layers. Expect to debug at least one of these.

**Alternative conservative framing**: cut M4.5 at M4.5-b (builder
done, nothing loaded yet) and promote M4.5-c + M4.5-d to a further
M4.5-extended sprint. Decision to make at M4.5 kickoff, not now.

---

## Dependencies added in M4

- `safetensors = "0.4"` (M4-a) — HuggingFace official Rust crate
  for reading and writing safetensors files. Small footprint,
  depends mainly on `serde_json`.
- `half = "2"` (M4-d) — de-facto standard Rust crate for IEEE 754
  binary16 operations. Used for F16 → F32 conversion. Small
  footprint, zero heavy deps.

Both unconditional (not feature-gated). Both chosen in preference
to hand-rolled equivalents to minimize surface for subtle bugs
(safetensors format edge cases; F16 subnormal / NaN handling).

---

## Test coverage summary

Tests added or modified under M4:

| File | Tests | Added in |
|------|-------|----------|
| `tests/safetensors_reader_test.rs` | 12 (9 for F32/parse/metadata + 4 for BF16/F16 decode and roundtrip) | M4-a, M4-d |
| `tests/miniflux_safetensors_roundtrip_test.rs` | 3 (index alignment, naming, bit-exact roundtrip) | M4-b |
| `tests/weight_mapper_test.rs` | 5 (load bit-exact, shape mismatch, missing, skipped, BF16 via mapper) | M4-c, M4-d (BF16 test updated) |
| 3 unit tests inside `src/v17/loader/weight_mapper.rs` | 3 (constructor: length mismatch, duplicates, empty legal) | M4-c |

**Total M4-specific: 23 tests.**

Plus 6 pre-existing tests updated to match new return-tuple shapes
after M4-b extended `build_mini_flux_language_model` and
`build_language_training_graph`:
  - `tests/apx_end_to_end_adaptive_execution_test.rs`
  - `tests/apx_learning_effect_test.rs`
  - `tests/apx_runtime_stability_test.rs`
  - `tests/apx_safe_autotuning_test.rs`
  - `tests/apx_semantic_equivalence_test.rs`
  - `tests/mini_flux_common.rs`

All pass; no correctness-path regression in the core M3 test suite.

---

## How to resume on M4.5

1. Read this file and the **Architectural decisions locked**
   section above.
2. Skim `HANDOFF_APX_V20_M3.md` §10 ("Deferred performance
   optimizations") for the M4.5 investigation notes that still
   apply.
3. Decide the target model up front: **TinyLlama-1.1B** is the
   recommended starting point (single-file safetensors, RMSNorm +
   RoPE, no GQA, Llama-family topology). Llama 3.2 1B is a
   close second but adds GQA.
4. Run `cargo build --lib` + `cargo test --test safetensors_reader_test`
   + `cargo test --test weight_mapper_test` + `cargo test --test
   miniflux_safetensors_roundtrip_test` to confirm M4 still green
   after any intervening work.
5. Start M4.5 with an investigation-previa (same pattern that
   worked for every M4 sub-phase) before writing code. Specifically
   investigate RoPE: what exact base_freq values do Llama and
   TinyLlama use? Where in the transformer block does RoPE apply
   (before or after Q/K projection)? Do we need to cache cos/sin
   tables or compute on the fly?
6. First real external validation of the M4 loader: have the
   M4.5-a or M4.5-b investigation download a small safetensors
   file (e.g. TinyLlama's `model.safetensors`) and run
   `SafetensorsReader::open` on it. Print the tensor list. This
   is a 10-minute sanity check that the M4 reader handles the
   format in the wild — if anything breaks (unusual dtype, giant
   shape, encoding quirk) we find out before building on top.

---

## Observations from the M4 sprint

Recorded briefly so the decisions taken in-flight are not lost.

- **The investigation-previa pattern paid off twice**. Checking
  the existing `v17/loader` skeleton before starting M4-a revealed
  `ModelLoader` + `LoaderError` + `ModelFormat::SafeTensors` were
  already in place; M4-a extended rather than replaced. Checking
  `register_weight` before M4-b revealed the logical names
  already existed internally (used only to seed deterministic
  init) — exposing them was a ~40 LOC change rather than a
  reinvented naming system.

- **Boundaries between sub-phases stayed clean by design**.
  M4-c identified that `TensorEntry::to_vec_f32` was the single
  point that needed to change for M4-d. The M4-d commit touched
  that function plus tests, with no mapper changes.

- **API refinement during implementation avoided later debt**.
  The M4-c design discussion proposed `from_handles(&MiniFluxHandles)`
  as a convenience constructor; the refined choice of
  `from_param_names_and_ids(&[String], &[usize])` keeps the
  mapper decoupled from any specific architecture handle. When
  TinyLlama's handle type appears in M4.5, the mapper takes it
  unchanged.

- **Blast-radius audits before signature changes were the
  cheapest disaster-prevention tool**. Before M4-b extended
  `build_mini_flux_language_model`'s return tuple, grepping for
  callers surfaced exactly 6 test files. All six were mechanical
  one-line destructuring changes. No surprise in the commit.

- **Claude Code consistently caught API ergonomics the spec
  didn't**. Examples: detecting that `SafetensorsReader` needed
  `Debug` for `expect_err` to compile (M4-a), noticing that
  `safetensors::SafeTensors` 0.4 does not expose `metadata()` on
  the view type and suggesting `read_metadata` as the correct
  entry (M4-a), pointing out that `cuda_malloc`-backed pool
  cannot cleanly trigger `PoolExhausted` deterministically (Debt
  #3 Fase 3.3, deferred). Treat these as signal to pause and
  confirm, not to paper over.

- **M4 closed in a single sprint**. Original estimate was "up to
  M4-d in ~1 week". Actual span was one active session. The
  multiplier came from: pre-existing skeleton, small and
  well-bounded scope per sub-phase, and investigation-previa
  surfacing gaps before they became mid-implementation pivots.
  Do not generalize — M4.5 has more unknowns (external model,
  numerical debugging) and should be sized honestly at its own
  kickoff, not extrapolated from M4.
