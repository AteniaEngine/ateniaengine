# Handoff — APX v20 M4.5 (Real Model Execution, at M4.5 close)

**Status at handoff**: M4.5 closed. Atenia executes a HuggingFace
TinyLlama-1.1B-Chat-v1.0 checkpoint end-to-end on CPU with logits
that match a PyTorch reference within F32-vs-BF16 precision drift
over 22 transformer blocks. All six sub-phases of the original
plan landed (M4.5-a, M4.5-b0, sub-step 3.0, M4.5-b1, M4.5-c,
M4.5-d.1).

The boundary between "the loader works" (M4) and "a real model
produces logits that agree with PyTorch on the same input"
(M4.5) is now crossed: the engine now hosts a complete Llama-2
class architecture with Rotary embeddings, Grouped Query
Attention (resolved at load time), causal masking, SwiGLU FFN,
RMSNorm with learnable γ, residual connections, and an LM head,
all composable from primitives that exist in `NodeType`.

**Last M4.5 commit**: `40fc4fa` (M4.5-d.1, PyTorch numerical
validation fixture and comparison test).

**Empirical baseline**: with `tokens = [1, 100, 200, 300]` and
the real `models/tinyllama-1.1b/model.safetensors`, Atenia's
logits sweep reports `max_abs_diff = 0.732`, `mean_abs_diff =
0.059`, `0%` of values with `|diff| > 1.0`. Top-1 prediction
disagrees with PyTorch (Atenia picks `id=29871`, PyTorch picks
`id=595`), but the two are in a top-2 near-tie: PyTorch's pick
is Atenia's #2 with logit difference `0.034`, and Atenia's pick
is PyTorch's #2 with logit difference `≈ 0`. Behavior consistent
with BF16-vs-F32 accumulation, not a correctness bug.

---

## What is ready

| Sub-phase | Commit | Summary |
|-----------|--------|---------|
| M4.5-a | `ece2750` | `NodeType::RoPE { head_dim: usize, base_freq: u32 }` with half-split (HuggingFace) layout. Forward + backward in `src/amg/graph.rs`, helper module `src/nn/rope.rs`. PyTorch reference fixture under `tests/fixtures/rope_reference/`. 8 tests (canary bit-exact vs PyTorch, position-zero identity, L2-norm preservation, multi-head independence, backward via finite-diff and via graph tape, integration with `Mul`). |
| M4.5-b0 | `b7c2a11` | Two new primitives: `NodeType::Permute { perm: Vec<usize> }` for general transposes (data copy, contiguous output) and `BatchMatMul` extended to rank 3 \|\| rank 4. Required for the multi-head attention layout swap `[b, s, h, d] → [b, h, s, d]` and for 4D `Q @ K.T` scores. 14 tests (9 Permute, 5 BMM4D). |
| Sub-step 3.0 | `f598868` | `NodeType::BroadcastMul`, the multiplicative analogue of `BroadcastAdd`. Found necessary mid-investigation: Atenia's `RmsNorm` does not multiply by γ, so RMSNorm-with-γ is composed as `rms_norm(x) → broadcast_mul(., γ_param)` with γ reshaped to `[1, 1, hidden]`. Backward derivation is the multiplication chain rule with reduction over broadcast dims. 6 tests. |
| M4.5-b1 | `2105ebd` | Three pieces in one commit: `TinyLlamaConfig` (parses HF `config.json` with HF-faithful field names and validates structural invariants); `WeightMapper` extended with a per-tensor `LoadTransform` pipeline (`Transpose2D`, `TileGroupedDim`, `Scale`, `Reshape`); and `build_tinyllama` (full 22-layer Llama-2 graph builder). M4-c API stays bit-exact backward-compatible (empty transform list ≡ direct copy). 24 tests across three files. |
| M4.5-c | `dcc61e7` | End-to-end smoke test in `tests/tinyllama_end_to_end_test.rs`. `#[ignore]`-gated, consumes `TINYLLAMA_SAFETENSORS_PATH`. Builds graph, loads 201 real BF16 tensors, drops the reader to free 2 GB raw bytes, runs forward on synthetic tokens, asserts shape `[1, 4, 32000]` + every logit finite + `max_abs < 1000`. No production code changed: the test is the integration that proves the builder, transforms, and execute path compose. |
| M4.5-d.1 | `40fc4fa` | PyTorch numerical validation. `tests/fixtures/tinyllama_reference/generate.py` runs HF `transformers.AutoModelForCausalLM` in BF16 on the same tokens and dumps logits; `tests/tinyllama_numerical_validation_test.rs` runs the Atenia pipeline and compares element-wise. The first run quantified drift as `max_abs = 0.73`, `mean_abs = 0.06`, `0%` of logits with `|diff| > 1.0` — within the F32-vs-BF16 envelope, not a bug. |

Every sub-phase commit is on `main` and pushed to `origin/main`.

---

## Architectural decisions locked

Treat as invariants. Future work extends rather than re-litigates.

1. **RoPE half-split layout, not interleaved**. The HuggingFace
   convention (`y[i] = x[i]·cos − x[i+half]·sin`,
   `y[i+half] = x[i+half]·cos + x[i]·sin`) is required for
   numerical equivalence with the PyTorch reference. The original
   RoPE paper uses a different (interleaved) layout and would
   produce different numbers under the same weights. The
   `NodeType::RoPE` doc comment names the convention explicitly.

2. **`NodeType::RoPE { base_freq: u32 }`, not `f32`**. Preserves
   the `Eq` derive on `NodeType`. No model in scope uses a
   sub-integer base frequency (Llama 1/2/TinyLlama: 10000;
   Llama 3+: 500000). When a model with non-integer θ appears,
   widen the field then.

3. **Positions implicit `[0..seq)` in RoPE**. KV-cached inference
   needs an explicit position offset; that argument-extension
   work is M5+. M4.5 inference recomputes the full sequence
   every call, which only works because seq is small in tests.

4. **Permute via data copy**. Atenia assumes `Layout::Contiguous`
   in 25+ sites including `MatMul`'s explicit assertion. Stride
   manipulation would silently break downstream ops; full data
   copy with `Layout::Contiguous` output is the only consistent
   strategy. `O(n)` cost is irrelevant against the matmul-
   dominated compute it serves.

5. **`BatchMatMul` extension to rank 3 \|\| rank 4 in the same
   `NodeType`**, not a new `BatchMatMul4D` variant. Detection by
   rank in the match arm; rank-4 inputs flatten dims 0+1 into a
   single outer batch before reaching the dispatcher. CUDA / SIMD
   dispatchers see flat memory and an opaque outer count — they
   are unaware of the rank-4 path. Backward extends identically.

6. **`NodeType::BroadcastMul` is its own primitive, not an
   extension of `RmsNorm`**. Modular and reusable: LayerNorm γ
   (when LayerNorm lands), per-channel bias, future LoRA scales
   all benefit. The alternative — extending `RmsNorm` to take
   `[x, γ]` as inputs — would break the existing single-input
   API used by MiniFlux without adding flexibility.

7. **GQA via tile-on-load, factor `kv_groups`**. K_proj and
   V_proj weights are replicated `kv_groups` times along the
   head dim during weight loading (TinyLlama: factor 8). The
   graph sees pure 32-head MHA. Trade-off: K/V weights occupy
   `kv_groups×` more RAM (≈ 700 MB extra for TinyLlama —
   acceptable for a 1 GB-class model on 16 GB hardware). When
   memory becomes the constraint (Llama 2 70B-class with M4.7
   beyond-VRAM execution), introduce a `repeat_kv` primitive
   and remove the load-time tile. Until then, tile-on-load is
   strictly simpler.

8. **Linear weight transpose on load (`[out, in]` HF →
   `[in, out]` Atenia) via `LoadTransform::Transpose2D`**. Atenia
   `nn::linear::linear` is documented as `weight: [in_features,
   out_features]`; HuggingFace stores `[out, in]`. Transposing
   on load keeps every `gb.matmul(x_flat, w)` call site
   convention-aligned without a runtime transpose node.

9. **Attention scaling pre-folded into K_proj weights**.
   `LoadTransform::Scale { factor: 1/sqrt(head_dim) }` runs as
   the last transform on every K_proj. Math: `scores = (Q @ (K
   ·1/√d).T) = (Q @ K.T) / √d`. The graph contains no scaling
   node. Bit-exact equivalent at FP32 precision; a few-ULP
   difference at FP16/BF16 — acceptable.

10. **Causal mask as runtime-shaped `Parameter [1, 1, seq, seq]`**,
    not `[1, 1, max_seq, max_seq]` with slicing. Atenia has no
    `Slice` primitive; the only way to use a static
    `[1, 1, max_seq, max_seq]` mask would be via a costly
    custom op. Reconstructing the mask per `TinyLlamaRuntime`
    is a few KB of data and `O(seq²)` initialization — trivial.

11. **`TinyLlamaRuntime { batch, seq }` is a `build_tinyllama`
    parameter**. `Reshape` requires concrete dims (one `-1`
    wildcard allowed, but not enough to leave both `batch` and
    `seq` symbolic). Different `(batch, seq)` requires
    rebuilding the graph. M5+ (KV cache + dynamic shapes) is
    the place this becomes a real cost.

12. **`gb.silu()` uses `NodeType::Activation(SiLU)`, not the
    `NodeType::SiLU` direct variant**. The Activation arm has
    a forward but no backward closure, suitable for inference.
    M4.5 is forward-only; when training real models lands
    (M5+), switch the SwiGLU path to `NodeType::SiLU` directly.

13. **`LoadTransform` pipeline order is per-tensor configurable**,
    enumerated in a `Vec<LoadTransform>`. The TinyLlama helper
    emits `[TileGroupedDim, Transpose2D, Scale]` for K_proj and
    `[Transpose2D]` for q/o/MLP/lm_head. The order matters and
    is documented in `compute_transforms_for_name`.

14. **`WeightMapper` empty-transforms path is bit-exact
    backward-compatible with M4-c**. The pipeline runs
    decoded values through zero transforms and then performs
    the same shape check + copy as before. The M4-c roundtrip
    test (`weight_mapper_test::load_into_roundtrip_bit_exact_matches_source_graph`)
    passes unchanged — verified in the M4.5-b1 commit.

15. **`serde_json` is a regular runtime dependency, not feature-
    gated**. Used by both the `hw-probe` binary and the new
    TinyLlama config loader. A small crate (already pulled in
    transitively through `safetensors`); the previous feature
    gate added complexity without saving footprint.

---

## Empirical validation results (M4.5-d.1)

Tokens: `[1, 100, 200, 300]` (BOS + 3 arbitrary IDs, same on both
sides). Both sides load `models/tinyllama-1.1b/model.safetensors`.
PyTorch runs in BF16 (the model's native dtype); Atenia upcasts
to F32 on load.

```
                  PyTorch       Atenia
max |logit|       21.1250       21.1400
mean |logit|       2.1159        2.1058

Element-wise diff (Atenia − PyTorch):
  max |diff|       0.732295
  mean |diff|      0.059480
  diff > 1e-2      87.31 %
  diff > 1e-1      17.02 %
  diff > 1.0        0.00 %        ← no catastrophic outliers

Argmax of last-position logits:
  PyTorch          id=  595, logit = 8.0625
  Atenia           id=29871, logit = 8.0984
  Top-1 match      no   (top-2 near-tie below)

Cross-check:
  PyTorch logit at Atenia's pick (29871):  8.0625   (= its own top-1)
  Atenia  logit at PyTorch's pick (  595):  8.0651  (≈ its own top-1)
```

**Interpretation.** Drift is uniform, bounded, and concentrated
in the noise floor. Mean absolute diff of 0.06 against logit
magnitudes around 21 is ~0.3 %. The only "interesting"
disagreement is the top-1 argmax, where the two implementations
pick opposite winners of a near-tie pair separated by `~0.03`
in logit space — well inside the F32-vs-BF16 precision envelope
accumulated over 22 layers + a 2048×32000 LM head matmul.
**Atenia's TinyLlama implementation is numerically faithful.**

---

## Performance observations

Wall-clock timings, batch=1, seq=4, single-pass forward:

| Phase | Debug build | Release build |
|------|------------:|--------------:|
| `SafetensorsReader::open` (read 2 GB BF16) | ~54 s | ~12 s (combined with load below) |
| `WeightMapper::load_into` (decode + transforms, 201 tensors) | ~60 s | ~12 s |
| `graph.execute(tokens)` (22 layers + LM head, ~5 GFLOPs) | ~70 s | ~35 s |
| **Total `--ignored` test runtime** | ~185 s | ~48 s |

The 35 s release-mode forward is **slower than expected** for
~5 GFLOPs of work on a 24-thread AVX2 CPU (theoretical peak in
the hundreds of GFLOPS). The matmul dispatcher likely does not
hit the AVX2 microkernel path for every shape and falls back
to a scalar route for some tensors. Investigation deferred —
optimization is out of scope for M4.5 but is now a known
follow-up with empirical numbers attached.

Memory peak observed during M4.5-c: roughly 6.5 GB
(2 GB raw safetensors buffer + 4.4 GB F32 graph parameters +
≤ 50 MB transient per-tensor `Vec<f32>` during decode).
Comfortably inside 16 GB. Dropping the reader after load
(`drop(reader)` in the test) frees the 2 GB raw bytes before
forward activations allocate.

---

## Gaps explicitly closed in M4.5

- A real LLM (TinyLlama 1.1B Chat v1.0) loads from disk and
  executes forward end-to-end on CPU.
- 22-layer Llama-2 architecture is fully buildable from
  Atenia's `NodeType` set.
- Multi-head attention with the `[b, s, h, d] ↔ [b, h, s, d]`
  layout swap, 4D batched scores, causal masking, and softmax
  is composable in the graph.
- Grouped Query Attention is supported via load-time tile;
  the graph itself stays MHA.
- RoPE (Llama-family half-split) is a graph primitive with
  forward, backward, and PyTorch-bit-exact validation.
- BF16 → F32 conversion validated empirically against a real
  2 GB checkpoint, not just synthetic data.
- Numerical agreement against PyTorch reference, quantified.
- HF parameter naming convention (`model.layers.{i}.…`) is
  consumed directly by the loader; no per-architecture name
  tables in the engine core.

---

## Gaps explicitly NOT closed — scope deferred

- **Tokenizer integration**. M4.5 uses synthetic float-cast
  token IDs. Real text → tokens needs the HuggingFace
  `tokenizers` crate (or a hand-rolled SentencePiece /
  BPE reader). M5+ scope.

- **KV cache**. M4.5 recomputes the full sequence on every
  forward call. Token-by-token generation without KV cache
  is O(N²) in seq, quickly intolerable. M5+ scope; touches
  the graph topology (the position-offset parameter inside
  `NodeType::RoPE` is the natural interface point).

- **Dynamic seq length without rebuild**. Tied to KV cache;
  see M5+.

- **Native BF16 / F16 storage**. M4 already chose
  upcast-on-load; M4.5 inherits that. Native storage requires
  dtype-specific kernels and a parallel matmul code path —
  separate post-M5 milestone.

- **Llama 3+ family**. `rope_scaling: "llama3"` (piecewise
  scaling of RoPE frequencies) is unsupported. M4.6 candidate;
  estimated 1 day of focused work plus regression tests.

- **GPT family**. Uses `LayerNorm` (not `RmsNorm`). LayerNorm
  is not a `NodeType` yet. Out of immediate roadmap.

- **DeepSeek family**. Uses MLA (Multi-head Latent Attention)
  instead of GQA — different attention topology. Post-M5 if it
  becomes a target.

- **GQA without load-time tile**. Memory-saving optimization
  worth doing once KV cache is added (M5+) — the KV cache
  itself is what benefits most from GQA's smaller K/V state.

- **Multi-file (sharded) safetensors**. TinyLlama is single-file.
  Llama-7B+ ships as `model-00001-of-N.safetensors` plus an
  `index.json`. Extension when needed.

- **`backward` over a loaded model**. M4.5 is forward-only.
  `gb.silu()` uses the no-backward `Activation(SiLU)` arm. A
  training pipeline for real models is M5+ territory.

- **Forward optimization**. 35 s release-mode forward on
  ~5 GFLOPs is a known follow-up. Profile the matmul
  dispatcher path on production-shape matrices.

- **Tokenizer round-trip in numerical-validation tests**.
  M4.5-d.1 uses synthetic token IDs (no decoder side). Once
  a tokenizer is integrated, validate that
  `decode(argmax(logits)) ≈ decode_pytorch(argmax(logits))`
  on real prompts.

---

## Next milestones proposed

### M4.6 — Llama-family compatibility expansion

**Goal.** Extend support beyond TinyLlama 1.1B to the most
commonly-used small open LLMs in the Llama / Llama-derivative
family: Llama 3.2 1B / 3B, Qwen 2.5 0.5B / 1.5B / 3B, Phi 3.5
mini, Mistral 7B (subject to RAM), SmolLM3.

**Estimated work**:
- Llama 3.2: `rope_scaling: "llama3"` piecewise scaling
  implemented inside `NodeType::RoPE` (or as a separate
  variant if invariance becomes hard). ~1 day.
- Qwen 2.5: QK-Norm variant (an extra RMSNorm on Q and K
  before attention scores). Composable from existing
  primitives. ~0.5 day.
- Mistral 7B: identical architecture to Llama 2; only
  hardware/RAM constraint. No engine work required.
- Phi 3.5 mini: minor architectural variants documented in
  HF. Estimated 0.5 day.
- SmolLM3: Llama 2 architecture; should work via
  `tinyllama_weight_mapper` with a different `TinyLlamaConfig`.

After M4.6, Atenia executes most popular small open LLMs in
the HuggingFace catalog without per-architecture builders.

### M4.7 — Beyond VRAM (the killer demo)

**Goal.** Execute Llama 2 13B (or equivalent ~13B-parameter model)
in BF16 on notebook-class hardware (8 GB VRAM, 16 GB RAM, SSD).
This is the demo that validates the original v20 M3 reaction-loop
work against a real workload that exceeds available VRAM.

**Why it matters**. The project's core differential is
hardware-adaptive execution intelligence. Until M4.7, that thesis
is exercised only by tests with synthetic memory-pressure
injection. A real Llama 2 13B forward on 8 GB hardware exercises
real memory thrashing, real tier transitions
(VRAM ↔ RAM ↔ SSD), and real reaction-loop decisions.

**Estimated work.** 5 to 10 days, with non-trivial profiling
work. Includes layer-level granularity decisions for offload
boundaries, prefetch strategy, reaction-loop integration.

### M5 — Inference UX

**Goal.** Tokenizer integration + KV cache + token-by-token
generation = the experience of "running a real LLM".

This is the milestone that moves Atenia from "we can produce
correct logits" to "we can chat". 2 to 4 weeks.

---

## Dependencies added in M4.5

- `transformers` (Python, dev-only) — used by the M4.5-d.1
  fixture generator. Not a runtime dependency.
- `serde_json` was promoted from feature-gated optional to
  unconditional (M4.5-b1). Footprint negligible.

No new Rust crates were added in M4.5; existing
`safetensors`, `half`, `serde_json` all carried over from M4.

---

## Test coverage summary

Tests added or modified under M4.5:

### Synthetic / fast tests (run by default)

| File | Tests | Sub-phase |
|------|-------|-----------|
| `tests/rope_test.rs` | 8 | M4.5-a |
| `tests/permute_test.rs` | 9 | M4.5-b0 |
| `tests/batch_matmul_4d_test.rs` | 5 | M4.5-b0 |
| `tests/broadcast_mul_test.rs` | 6 | sub-step 3.0 |
| `tests/tinyllama_config_test.rs` | 8 (+ 1 ignored) | M4.5-b1 |
| `tests/tinyllama_weight_loading_test.rs` | 10 (+ 1 ignored) | M4.5-b1 |
| `tests/tinyllama_builder_test.rs` | 6 | M4.5-b1 |
| Internal unit tests in `src/v17/loader/weight_mapper.rs` | +3 over M4-c | M4.5-b1 |

**Total fast tests added in M4.5: 55 plus 3 ignored, all green.**

### `#[ignore]`-gated tests (real model, env var)

| File | Sub-phase | Env var |
|------|-----------|---------|
| `tests/tinyllama_end_to_end_test.rs` | M4.5-c | `TINYLLAMA_SAFETENSORS_PATH` |
| `tests/tinyllama_numerical_validation_test.rs` | M4.5-d.1 | `TINYLLAMA_SAFETENSORS_PATH` (+ committed PyTorch fixture) |

Both run green on the real `models/tinyllama-1.1b/model.safetensors`.
M4-era ignored tests (`m4_real_safetensors_validation_test`)
remain green.

### Regressions verified at every sub-phase

After each commit in M4.5, the relevant prior tests stayed green:
`weight_mapper_test` (M4-c roundtrip), `miniflux_safetensors_roundtrip_test`
(M4), `backprop_linear_test`, `apx_11_2_matmul_backward`. The
M4-c bit-exact roundtrip is the load-time invariant most at risk
during `LoadTransform` work and was checked specifically before
M4.5-b1 closed.

---

## How to resume on M4.6 or M4.7

1. Read this file. Pay particular attention to **Architectural
   decisions locked** — those are non-negotiable starting points.
2. Confirm the target model up front. Don't extrapolate from
   TinyLlama: a single `config.json` field can change everything
   (e.g. `rope_scaling`, `attention_bias: true`, `tie_word_embeddings: true`).
3. Place the real `model.safetensors` (and `config.json`) under
   `models/<target>/`.
4. Run the existing M4.5 test suite to confirm nothing has
   regressed locally:
   ```
   cargo test --test rope_test --test permute_test \
              --test batch_matmul_4d_test --test broadcast_mul_test \
              --test tinyllama_config_test --test tinyllama_weight_loading_test \
              --test tinyllama_builder_test --test weight_mapper_test \
              --test miniflux_safetensors_roundtrip_test
   ```
5. Investigation phase first. The pattern that worked in M4.5
   was: read the new model's `config.json` end to end; diff its
   field set against `TinyLlamaConfig`; locate every operation
   in the architecture that doesn't already exist in `NodeType`.
   Ship a short report before writing builder code.
6. Implementation follows the same template: single new sub-
   phase per primitive; integration sub-phase at the end; a
   PyTorch-fixture numerical validation as the close criterion.

---

## Observations from the M4.5 sprint

Recorded so the in-flight decisions are not lost.

- **Investigation-previa caught three architectural blockers
  before they became debugging sessions**. Each would have cost
  days at runtime:
  1. TinyLlama 1.1B Chat v1.0 uses GQA (4 KV heads vs 32 Q heads)
     — discovered by reading `config.json` before writing the
     builder. Resolved by tile-on-load instead of a runtime
     `repeat_kv` primitive (decision #7 above).
  2. RoPE uses the half-split (HuggingFace) layout, not the
     interleaved (original-paper) layout. PyTorch fixture
     generated before any Rust code; first canary test would
     have failed with a coherent diagnostic if this had been
     wrong.
  3. `NodeType::RmsNorm` does not multiply by γ. Discovered
     when planning the builder, not when watching wrong logits
     come out. Resolved by introducing `BroadcastMul` as an
     independent primitive (sub-step 3.0) before continuing
     M4.5-b1.

- **The agent reported architectural blockers instead of
  improvising**. Twice during M4.5-b1 it stopped, surfaced an
  unresolvable gap, and waited for a decision rather than
  hacking around the issue. Both decisions (BroadcastMul as a
  new primitive; `LoadTransform::Reshape` for γ shape
  alignment) were small and clean; both would have been costly
  to retrofit had they been worked around.

- **Empirical validation in two stages worked**. M4.5-c's
  qualitative "logits exist, are finite, magnitudes plausible"
  is a fast smoke; M4.5-d.1's quantitative element-wise diff
  against PyTorch is the rigorous gate. Splitting them meant
  the qualitative test landed in commit form before the
  quantitative comparison was even set up — useful safety net
  if the PyTorch fixture step had hit a snag.

- **Top-1 argmax disagreement was not a bug**. The first
  diagnostic instinct ("Atenia predicts a different token!") was
  wrong; the top-2 cross-check proved both implementations agree
  on the candidate set and disagree only on a near-tie. Reading
  the data carefully before reaching for a fix saved an
  unnecessary debug pass.

- **The full M4.5 milestone closed in one sprint** despite its
  scope (RoPE + Permute + 4D BMM + BroadcastMul + TinyLlama
  builder + load + execute + PyTorch validation). Dominant
  reasons: every sub-phase was investigated before being
  implemented; the existing M3 reactive infrastructure stayed
  out of the way; PyTorch was already installed for the M4.5-a
  RoPE fixture so M4.5-d.1 needed only a `pip install
  transformers`. Do not generalize — M4.6/M4.7 face different
  unknowns.

- **Test-suite hygiene paid back, again**. The benchmark-tagging
  work from M3 that moved 26 tests behind `#[ignore]` continued
  to pay off: the M4.5 regression checks ran in seconds after
  every commit, not hours, which made the "always run all
  regressions before declaring green" discipline cheap enough
  to actually maintain.
