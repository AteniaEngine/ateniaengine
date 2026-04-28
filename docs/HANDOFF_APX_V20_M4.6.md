# Handoff — APX v20 M4.6 (Llama-Family Compatibility Expansion, at M4.6 close)

**Status at handoff**: M4.6 closed. Atenia executes four members
of the Llama-family architecture end-to-end on CPU
(TinyLlama 1.1B, SmolLM2 1.7B, Qwen 2.5 1.5B, Llama 3.2 1B), each
validated against PyTorch F64 mathematical ground truth per
ADR-004. The same `build_llama` graph + `llama_weight_mapper`
loader serve all four checkpoints; per-architecture variation
rides exclusively in the parsed config.

The boundary between "we can run TinyLlama" (M4.5) and "we can
run the Llama family" (M4.6) is now crossed: the engine handles
tied embeddings, configurable RmsNorm eps, attention biases (Qwen
QKV), and Llama 3 piecewise RoPE scaling — all behind a single
`LlamaConfig` with `effective_*` helpers that resolve per-family
defaults.

**Last M4.6 commit**: `d74a7f2` (M4.6.1, retroactive F64
validation for TinyLlama). Phase C closes at `0537864`.

**Empirical baseline** (max drift Atenia F32 vs PyTorch F64,
tokens `[1, 100, 200, 300]`):

```
                 Atenia F32 vs F64    PyTorch BF16 vs F64    Ratio
TinyLlama 1.1B          0.000141              0.732367      5198x
SmolLM2 1.7B            0.001446             14.013928      9692x
Qwen 2.5 1.5B           0.000346              1.531417      4420x
Llama 3.2 1B            0.000132              0.538695      4096x
```

All four sit in the same precision class (~10⁻⁴ to 10⁻³ max
drift), consistent with F32 forward through 16–28-layer
transformer stacks. Argmax MATCH 4/4 positions on every model.
Atenia's logits are between three and four orders of magnitude
closer to mathematical truth than PyTorch BF16 inference on the
identical checkpoint.

---

## What is ready

| Sub-phase | Commit | Summary |
|-----------|--------|---------|
| **Phase A — engine extensions** | | |
| A.1 — tied word embeddings | `408f1f6` | `NodeType::RoPE` untouched; tied path lands in `build_llama` as `gb.transpose_2d(embed_w)` reusing `embed_tokens` as a multi-consumer Parameter. New `gb.transpose_2d(src) -> usize` builder method. Branch is `if config.tie_word_embeddings`. SmolLM2 (24 layers, vocab 49152) is the first checkpoint exercised end-to-end. |
| A.2 — RmsNorm eps configurable | `306e588` | `NodeType::RmsNorm` carries `eps_bits: u32` (raw `f32::to_bits`) so the variant stays `Eq + Hash`-derivable. `gb.rms_norm(x_id, eps)` requires the parameter at every call site. Closes a long-standing latent bug: `LlamaConfig::rms_norm_eps` was parsed and validated since M3 but never consumed by the graph (hardcoded 1e-5 throughout). Required by Qwen 2.5 (1e-6); transparent for the rest of the family. |
| A.3.1 — TinyLlama → Llama rename | `6c9ec15` | Mechanical rename: `TinyLlamaConfig → LlamaConfig`, `TinyLlamaRuntime → LlamaRuntime`, `TinyLlamaHandles → LlamaHandles`, `build_tinyllama → build_llama`, `tinyllama_weight_mapper → llama_weight_mapper`, `src/nn/tinyllama/ → src/nn/llama/`. Test file names retained (`tests/tinyllama_*.rs`) — they exercise the TinyLlama 1.1B checkpoint specifically, not the generic infrastructure. Formalises what was already true since A.1: SmolLM2 was already importing `nn::tinyllama::*`. |
| **Phase B — Qwen 2.5 1.5B Instruct** | | |
| B.1 — parser optional fields | `9dffeb5` | `LlamaConfig::attention_bias: Option<bool>`, new `model_type: Option<String>`, helper `effective_attention_bias()`. Qwen 2.5's `config.json` omits `attention_bias` entirely (Qwen2 hard-codes QKV biases on inside `Qwen2Attention`); the helper resolves the family default when the field is absent. New optional-parsing helpers for bool and string. |
| B.2 — builder QKV bias | `306b950` | `if config.effective_attention_bias()` branch in `build_llama`: registers `q/k/v_proj.bias` Parameters with shape `[1, hidden]` (rank 2 for `BroadcastAdd` same-rank rule) and inserts three `BroadcastAdd` nodes between the QKV projections and the multi-head reshape. Path stays a no-op when false. Legacy regressions (TinyLlama, SmolLM2) remain bit-identical. |
| B.3 — weight mapper QKV bias | `98ac1ed` | Three new handlers in `compute_transforms_for_name`: q bias `[hidden]` → `[Reshape([1, hidden])]`; k bias `[kv_dim]` → `[TileGroupedDim, Reshape, Scale]`; v bias `[kv_dim]` → `[TileGroupedDim, Reshape]`. The K bias absorbs `1/sqrt(d_k)` exactly like the K weight does — without the matching scale, biased attention scores would diverge by exactly `factor ≈ √64`. |
| B.4 — Qwen 2.5 smoke test | `25c59d3` | `tests/qwen25_end_to_end_test.rs`. Builds 338-parameter graph (1 embed + 28×12 + 1 final norm; 84 of those are bias params), loads all 338 safetensors (`skipped=0`, `missing=0`), forwards on `[1, 100, 200, 300]`. Logits `[1, 4, 151936]` finite; max=13.47, mean=2.11; predicted token at pos 3 id=1 logit=11.06. Forward 20.5 s. |
| B.5 — Qwen 2.5 F64 validation | `919b832` | `tests/qwen25_numerical_validation_test.rs`. Three-way comparison (Atenia F32 vs F64 [primary], Atenia vs BF16 [secondary], BF16 vs F64 [informative]). Atenia max drift 0.000346 vs F64; ratio vs BF16 = 4420×; argmax MATCH 4/4 (id=330, 198, 198, 1). Threshold `< 0.5` per ADR-004. Fixture under `tests/fixtures/qwen25_reference/`. |
| **Phase C — Llama 3.2 1B Instruct** | | |
| C.1 — parser rope_scaling + head_dim | `6343f0f` | `LlamaConfig::head_dim: Option<usize>` with helper `effective_head_dim()` (explicit field wins, falls back to `hidden_size / num_attention_heads`); `LlamaConfig::rope_scaling: Option<RopeScaling>` enum-typed (only `Llama3` variant today, future families add explicit branches). Parser tolerates both `rope_type` (modern) and `type` (legacy) discriminator. Unknown `rope_type` (yarn, longrope, …) parses to `None`. |
| C.2 — compute_inv_freqs_llama3 | `42ce78a` | Pure-function piecewise inverse-frequency transform mirroring `huggingface/transformers::modeling_rope_utils::_compute_llama3_parameters`. F64 internal compute, F32 output: with `base_freq = 500_000`, `b^(2i/d)` loses meaningful precision in pure F32 near `head_dim/2`. Four unit tests cover the three bands (high-freq identity, low-freq /factor, mid-band smooth interp) plus a factor=1 identity sanity check. |
| C.3 — wire scaling into NodeType::RoPE | `1366f87` | `NodeType::RoPE` extended with `scaling: Option<RopeScalingLlama3>`. New `RopeScalingLlama3` struct stores `factor`, `low_freq_factor`, `high_freq_factor` as raw `f32::to_bits` (`u32`) to keep the variant `Eq + Hash`-derivable. Forward computes `inv_freqs` once and shares the same vector with the backward closure — agreement by construction. Builder gains `gb.rope_scaled(...)`; `build_llama` routes through `config.effective_rope_scaling()`. Legacy path remains bit-identical on all three prior models. |
| C.4 — Llama 3.2 smoke test | `ea71036` | `tests/llama_3_2_end_to_end_test.rs`. 146-parameter graph (1 embed + 16×9 + 1 final norm; no `lm_head` under tied embeddings, no QKV biases). 146 tensors loaded clean. Logits `[1, 4, 128_256]`; max=11.13, mean=1.45; predicted token at pos 3 id=264 logit=9.27. Forward 44 s. Test docstring explicitly notes `seq=4` cannot detect a broken scaling; C.6 is the falsifier. |
| C.5 — Llama 3.2 F64 validation | `7060869` | `tests/llama_3_2_numerical_validation_test.rs`. Atenia max drift 0.000132 vs F64; ratio 4096×; argmax MATCH 4/4 (id=11, 12942, 863, 264). Same caveat printed at the end of the test: this passes whether or not the Llama 3 scaling is correctly wired. |
| C.6 — long-context falsifier | `0537864` | `tests/llama_3_2_long_context_validation_test.rs`. Builds a synthetic input at `seq_len = 2048` and asserts (A) the graph output bit-exactly equals `apply_rope_with_inv_freqs(x, head_dim, &compute_inv_freqs_llama3(...))` — proves the scaled vector reaches the kernel through the AMG pipeline; (B) the same input run through plain unscaled RoPE differs by max abs ≈ 1.74 — proves scaling is materially active. A second short-context test (`seq=4`) documents the canary at 2.9e-3, explaining why C.4 / C.5 cannot detect a missing scaling. |
| **M4.6.1 — retroactive F64 (TinyLlama)** | | |
| M4.6.1 | `d74a7f2` | New `tests/tinyllama_f64_validation_test.rs` mirrors the SmolLM2 / Qwen / Llama 3.2 pattern. Atenia max drift 0.000141 vs F64; ratio 5198×; argmax MATCH 4/4. Original `tinyllama_numerical_validation_test.rs` (M4.5-d.1, BF16 reference) left bit-for-bit untouched as historical record of the pre-ADR-004 methodology. ADR-004 footnote: M4.5-d.1's pos 3 "Match: false" was BF16 quantisation noise (both 595 and 29871 round to logit 8.0625, near-tie), not an Atenia bug — F64 reference confirms 29871 wins by ~0.04. |

Every commit is on `main` and pushed to `origin/main`. Phase A, B,
C, and M4.6.1 each closed with their own commit set; no commits
mix sub-phases.

Three ADRs accepted during M4.6 — `ADR-001` (numerical health
monitor deferred), `ADR-002` (mathematical ground-truth validation
strategy), `ADR-003` (methodology questioning framework),
`ADR-004` (F64 reference as default validation methodology). All
four live under `docs/decisions/` and bracket the empirical
discovery (Phase A SmolLM2 BF16 catastrophic drift) that motivated
the methodology shift.

---

## Architectural decisions locked

Treat as invariants. Future work extends rather than re-litigates.
The M4.5 invariants from `HANDOFF_APX_V20_M4.5.md` are still in
force; the list below adds M4.6's contributions on top.

16. **`f32` config scalars in `NodeType` are stored as `u32` via
    `f32::to_bits`**. Required because `NodeType` derives
    `Eq + Hash`, which `f32` does not satisfy. Pattern established
    in M4.5 for `RoPE { base_freq: u32 }` — extended in M4.6 to
    `RmsNorm { eps_bits: u32 }` and `RopeScalingLlama3 { factor_bits,
    low_freq_factor_bits, high_freq_factor_bits, … }`. New scalars
    in `NodeType` should follow the same convention. Round-trip
    via `f32::from_bits` is bit-exact, no precision lost.

17. **Per-family defaults live in `LlamaConfig::effective_*`
    helpers, not in the builder**. Three helpers exist:
    `effective_attention_bias()`, `effective_head_dim()`,
    `effective_rope_scaling()`. Each centralises the disambiguation
    between "explicit field present" vs "absent — derive from
    `model_type`". Builder code never inspects the raw `Option`
    fields; it always asks the helper. New per-family defaults add
    to this list.

18. **`attention_bias` is qwen2-aware by default**. `Some(b)` wins;
    when `None`, `model_type == "qwen2"` resolves to `true`,
    everything else to `false`. The branch is one line in
    `effective_attention_bias()`. Future families that hard-code
    biases (none currently in scope) extend the same pattern.

19. **`head_dim` is parsed when present, derived when absent**.
    `LlamaConfig::head_dim: Option<usize>` with
    `effective_head_dim()` returning `Some(field)` or
    `hidden_size / num_attention_heads`. For all four M4.6
    checkpoints the two values coincide; the field exists to keep
    architectures like Phi-3 medium (where `head_dim` and
    `hidden_size / num_attention_heads` diverge) one config field
    away from working.

20. **`rope_scaling` is an `enum`, not parsed-but-ignored**. The
    parser explicitly enumerates every recognised `rope_type`
    (currently just `"llama3"`). Unknown `rope_type` (yarn,
    longrope, dynamic, …) parses to `None`. This is deliberate:
    silent acceptance of an unsupported scaling would produce
    wrong logits without surfacing the gap. Future families add
    explicit `RopeScaling::*` branches when their scaling is
    implemented, never before.

21. **K bias absorbs `1/sqrt(d_k)` exactly like K weight does**.
    Discovered during B.3, not anticipated in the plan. PyTorch
    computes `scores = Q @ (h W_k + b_k)^T / sqrt(d_k)`. Atenia
    pre-folds the scale into the K weight at load time; the
    matching scale must apply to the K bias too, otherwise
    biased attention scores diverge by a factor of `√d_k`. The
    weight mapper handler emits `[TileGroupedDim, Reshape, Scale]`
    for K bias to mirror the K weight pipeline. V bias has no
    scale (V never enters the QK product).

22. **Llama 3 piecewise RoPE scaling lives in
    `compute_inv_freqs_llama3` as a pure function**. Internal
    math runs in `f64` and casts to `f32` at the end. With
    `base_freq = 500_000`, `b^(2i/d)` loses meaningful precision
    in pure F32 for indices near `head_dim/2`; routing through
    `f64` keeps Atenia's inverse frequencies faithful to the HF
    reference. The legacy `compute_inv_freqs` retains its
    F32-only path so non-Llama-3 checkpoints remain bit-identical.

23. **Forward and backward of RoPE share one precomputed
    `inv_freqs` vector**. The graph executor computes the vector
    once (legacy or Llama 3 schedule, dispatched on `scaling:
    Option<RopeScalingLlama3>`), uses it in the forward, and
    moves it into the backward closure. There is no path where
    forward and backward reconstruct independent copies — agreement
    by construction. New scaling families should plug into the
    same dispatch.

24. **F64 reference is the primary numerical validation
    methodology**. ADR-004 is the formal record. Concretely: every
    new model added to Atenia ships with both a BF16 fixture (for
    historical continuity / industry-drift telemetry) and an F64
    fixture; only the F64 comparison gates the test. Threshold
    `max_abs_diff < 0.5`, comfortable headroom over the empirical
    ~10⁻⁴ baseline. The pre-ADR-004 BF16-gated tests (M4.5-d.1
    TinyLlama) are preserved untouched; new tests follow the
    F64-gated pattern.

25. **Legacy path bit-identity is a regression invariant**. After
    every M4.6 sub-phase, the prior models' numerical validations
    (TinyLlama M4.5-d.1, SmolLM2 F64, then also Qwen F64) were
    re-run and required to produce stat-by-stat identical output.
    Bit-identity, not "approximately equal". Any sub-phase that
    can't satisfy this is incorrectly scoped. Verified after A.1,
    A.2, A.3.1, B.2, B.3, B.4, B.5, C.3, C.4, C.5, C.6, and
    M4.6.1 — twelve checkpoints, all clean.

26. **Investigation precedes implementation, every sub-phase**.
    Each Phase opened with an investigation report (Phase A:
    tied embeddings + SmolLM2 architecture; Phase B: Qwen 2.5
    config + safetensors layout + QK-Norm absence; Phase C:
    rope_scaling math). Sub-phases inside a Phase reused the
    investigation; primitives were named before being coded. The
    pattern caught two architectural surprises before they became
    debugging sessions (K bias scaling in B; long-context
    falsifier requirement in C). The pattern is a contract — new
    Phases should not skip it.

---

## Empirical validation results — F64 family table

Tokens `[1, 100, 200, 300]` (BOS + 3 arbitrary IDs), batch=1,
seq=4 unless otherwise specified. F64 reference generated by
loading the same checkpoint with `torch_dtype=torch.float64` and
running one forward pass; saved to
`tests/fixtures/<model>_reference/expected_logits_f64.json`.

| Model | Layers | Vocab | RoPE θ | Special | Atenia vs F64 (max) | Atenia vs F64 (mean) | BF16 vs F64 (max) | Ratio |
|-------|-------:|------:|-------:|---------|--------------------:|---------------------:|------------------:|------:|
| TinyLlama 1.1B Chat | 22 | 32_000 | 10_000 | (M4.5 baseline) | 0.000141 | 0.000015 | 0.732367 | 5198× |
| SmolLM2 1.7B Instruct | 24 | 49_152 | 130_000 | tied embed | 0.001446 | — | 14.013928 | 9692× |
| Qwen 2.5 1.5B Instruct | 28 | 151_936 | 1_000_000 | QKV bias, eps=1e-6 | 0.000346 | 0.000032 | 1.531417 | 4420× |
| Llama 3.2 1B Instruct | 16 | 128_256 | 500_000 | rope_scaling llama3 | 0.000132 | — | 0.538695 | 4096× |

Argmax MATCH 4/4 between Atenia F32 and F64 on every model.
SmolLM2 is the largest gap to BF16 because BF16's catastrophic
position-0 drift on that specific 24-layer stack inflates the
denominator (Investigation F under Phase A surfaced the underlying
BF16 instability — see ADR-004 Application Note).

The numbers above use Atenia's parameters loaded from the same
safetensors files PyTorch reads; the comparison is pure
implementation precision, not a checkpoint disagreement.

---

## Performance observations

Wall-clock timings, batch=1, seq=4, single-pass forward, release
build, 24-thread AVX2 CPU:

| Model | Reader open | Mapper load_into | Forward | Total `--ignored` test |
|-------|------------:|-----------------:|--------:|-----------------------:|
| TinyLlama 1.1B | ~12 s | ~12 s | ~35 s | ~48 s (M4.5 baseline; carried) |
| SmolLM2 1.7B | ~50 s | ~12 s | ~50 s | ~115 s |
| Qwen 2.5 1.5B | ~77 s | ~11 s | ~21 s | ~109 s |
| Llama 3.2 1B | ~48 s | ~10 s | ~44 s | ~103 s |

Reader times scale roughly with on-disk size (BF16 → F32 upcast
is the dominant cost). Forward times scale with parameter count
× layer count, not linearly; Qwen 2.5 1.5B forward is faster than
Llama 3.2 1B forward despite Qwen having more parameters because
Qwen's hidden_size is smaller (1536 vs 2048) and matmul shapes
are friendlier on this dispatcher path.

The 35–50 s release-mode forwards remain the same M4.5 follow-up:
matmul dispatcher likely missing the AVX2 microkernel on some
shapes. Out of scope for M4.6.

Memory peaks remain comfortably inside 16 GB for all four models.
F64 fixture generation is the heaviest step (Qwen ~16 GB, Llama
3.2 ~12 GB, SmolLM2 ~14 GB, TinyLlama ~9 GB peak), and each was
measured to fit. Larger models (post-M4.7 territory) will require
either dedicated hardware for fixture generation or a
layer-by-layer F64 conversion fallback.

---

## Gaps explicitly closed in M4.6

- Atenia executes four production small open LLMs end-to-end on
  CPU with F64-validated numerical fidelity: TinyLlama 1.1B,
  SmolLM2 1.7B, Qwen 2.5 1.5B, Llama 3.2 1B.
- Tied word embeddings supported via graph-level
  `gb.transpose_2d(embed_w)` reuse; no separate `lm_head.weight`
  parameter in tied checkpoints.
- RmsNorm eps configurable per-model; `LlamaConfig::rms_norm_eps`
  is finally consumed by the graph (latent bug since M3).
- Q/K/V projection biases supported when the family demands them
  (Qwen 2.5). K bias absorbs the attention scale just like the
  K weight does — discovered analytically before the smoke test
  could have surfaced it as a numerical gap.
- Llama 3 piecewise RoPE frequency scaling implemented end-to-end
  with both mathematical (C.2 unit tests) and graph-pipeline (C.6
  falsifier) verification.
- HF `config.json` fields `model_type`, `head_dim`, `rope_scaling`
  parse cleanly across all four checkpoints, including the
  Qwen2 case where `attention_bias` is absent and the parser must
  resolve the family default.
- Generic infrastructure renamed from `TinyLlama*` /
  `nn::tinyllama` to `Llama*` / `nn::llama`, formalising what was
  already true since A.1: the same builder + mapper serve the
  whole family.
- ADR-002 / ADR-003 / ADR-004 lock down the methodology shift
  from BF16-as-truth to F64-as-truth with a documented empirical
  trigger.
- M4.6.1 closes the methodology debt left by M4.5-d.1: TinyLlama
  has its F64 validation now, mirroring the rest of the family,
  without invalidating the M4.5 commit.

---

## Gaps explicitly NOT closed — scope deferred

The M4.5 deferred-scope list (tokenizer, KV cache, dynamic seq,
native BF16 storage, GPT family, DeepSeek family, sharded
safetensors, backward over a loaded model, forward optimization,
tokenizer round-trip in numerical tests) remains in force. M4.6
did not address any of those.

New deferrals introduced or re-confirmed during M4.6:

- **Phi 3.5 mini**. Originally enumerated in the M4.5 next-
  milestones list as an M4.6 candidate. Not investigated during
  M4.6. Candidate for M4.6.2 or M4.7. Per the HF model card the
  architecture is a Llama-class layout with minor variants; an
  investigation-previa will determine whether new primitives are
  required (most likely yes for the long-context grouped variants;
  the 4k-context base may work directly through `build_llama` +
  config additions).

- **Mistral 7B**. Listed as an M4.6 candidate in the M4.5 handoff
  with the caveat "subject to RAM". The architecture is identical
  to Llama 2 — no engine work is required. The blocker is purely
  hardware: a 7B BF16 model exceeds 16 GB once F32 upcast and
  activations are added. Mistral 7B is therefore not an M4.6 gap;
  it becomes accessible automatically once M4.7 (beyond-VRAM
  execution) lands.

- **SmolLM3**. Enumerated in M4.5 as an M4.6 candidate. M4.6
  shipped SmolLM2 instead (Phase A.1, single-file safetensors),
  which is architecturally equivalent. SmolLM3 is sharded and
  was deferred at the model-selection stage during the Phase A
  download decision. Multi-file safetensors loading is the
  prerequisite — same as for any other 7B+ model.

- **Long-context end-to-end F64 validation**. C.6 falsifies the
  rope_scaling wiring with a synthetic seq=2048 RoPE-only test
  plus a mathematical proof from C.2. The full-model F64 forward
  at long context is intentionally not exercised: F64 fixture
  generation at seq ≥ 2048 for a 1B model approaches RAM limits
  on 16 GB hardware, and the C.2 + C.6 combination already
  proves the only remaining gap (graph wiring threading the
  scaled vector through to the kernel). When dedicated long-
  context numerical validation becomes useful (e.g. for prompt
  caching M5+), this is the natural follow-up.

- **Position offset in RoPE**. Still implicit `[0..seq)`. KV
  cache (M5+) is the trigger; no work in M4.6.

- **`backward` over scaling-extended RoPE**. Implemented and
  tested via the existing graph-tape mechanism, but no
  finite-difference numerical check was added for the
  `Some(scaling)` path. Sufficient for inference; revisit when
  training pipelines land.

- **Forward performance optimization**. The M4.5 handoff
  documented a known follow-up: 35 s release-mode forward on
  ~5 GFLOPs is slower than expected for a 24-thread AVX2 CPU
  (theoretical peak in the hundreds of GFLOPs). The matmul
  dispatcher likely does not hit the AVX2 microkernel path for
  every shape. M4.6 added three more datapoints (SmolLM2 50 s,
  Qwen 2.5 21 s, Llama 3.2 44 s release-mode forward at seq=4)
  but no profiling work; the dispatcher gap is unchanged. M4.7
  is the natural trigger: beyond-VRAM execution on a real 13B
  model will produce the first empirical data on where the
  actual bottleneck sits — compute, memory bandwidth, or tier
  transition latency. The performance optimization milestone
  should be scoped *after* M4.7 numbers are available, not
  before; optimising on the M4.6 baseline risks chasing the
  wrong bottleneck before the killer-demo workload exposes
  what matters.

---

## Next milestones proposed

### M4.6.2 — Phi 3.5 mini (optional, sub-milestone)

**Goal.** Cover the Phi family the same way M4.6 covered the
Qwen family.

**Estimated work.** Investigation-previa first (1–2 hours):
read `config.json`, list safetensors keys, diff against
`LlamaConfig`, identify primitives. The 3.8B model's
architecture is closer to Llama than to GPT, so most
infrastructure should reuse. The unknowns are
`rope_scaling: "longrope"` (a different piecewise schema; needs
a `compute_inv_freqs_longrope` parallel to the llama3 version)
and any QK-Norm variants. Implementation 0.5–1 day if the
investigation surfaces no new primitives; up to 2 days if it
does.

**When to do it.** Optional. Useful before M4.7 only if Phi 3.5
mini is a target deployment; otherwise M4.7 (beyond-VRAM) is
strictly higher leverage.

### M4.7 — Beyond VRAM (the killer demo) — UNCHANGED FROM M4.5 HANDOFF

The M4.5 handoff's M4.7 framing remains correct. M4.6 did not
address it; the engine still upcasts everything to F32 and holds
all parameters in RAM. The thesis "hardware-adaptive execution
intelligence" remains exercised only by tests with synthetic
memory-pressure injection. M4.7 is the demo that validates the
v20 reaction-loop work against a real Llama 2 13B forward on
8 GB VRAM + 16 GB RAM hardware. 5 to 10 days, with non-trivial
profiling work. Mistral 7B falls out as a side effect of M4.7
since its architecture is already supported.

### M5 — Inference UX — UNCHANGED FROM M4.5 HANDOFF

Tokenizer integration + KV cache + token-by-token generation,
2–4 weeks. Becomes meaningfully easier post-M4.6 because every
model in scope is now numerically validated; the unknowns
collapse to UX rather than correctness.

---

## Dependencies added in M4.6

None. The Python fixture generation scripts use the same
`torch` + `transformers` stack that M4.5-d.1 already required
(dev-only, not a runtime dependency). No new Rust crates.

A `transformers >= 4.43.0` floor is implicit for the Qwen 2.5
fixture generator (the Qwen2 modeling module landed in 4.39).
Llama 3.2 needs `>= 4.45.0` for the rope_scaling "llama3"
default config-resolver. Both pinned in the per-fixture
`requirements.txt`.

---

## Test coverage summary

Tests added in M4.6, by sub-phase. All green at handoff.

### Synthetic / fast tests (run by default)

| File | Tests | Sub-phase |
|------|-------|-----------|
| `tests/rms_norm_eps_test.rs` | 2 | A.2 |
| `tests/softmax_adversarial_test.rs` | 11 | A (during Investigation F) |
| `tests/tinyllama_tied_path_regression_test.rs` | 1 (`#[ignore]`) | A.1 |
| `tests/tinyllama_config_test.rs` | +9 over M4.5 (4 inline llama3, 5 inline new + 4 ignored on-disk) | A.1, A.2, A.3.1, B.1, C.1 |
| `tests/tinyllama_weight_loading_test.rs` | +3 over M4.5 (qwen QKV bias dispatch) | B.3 |
| `src/nn/rope.rs::llama3_scaling_tests` | 4 (in-source) | C.2 |
| `tests/llama_3_2_long_context_validation_test.rs` | 2 | C.6 |

### `#[ignore]`-gated tests (real model, env var)

| File | Sub-phase | Env var |
|------|-----------|---------|
| `tests/smollm2_end_to_end_test.rs` | A.1 | `SMOLLM2_SAFETENSORS_PATH` |
| `tests/smollm2_numerical_validation_test.rs` | A (Investigation F) | `SMOLLM2_SAFETENSORS_PATH` |
| `tests/qwen25_end_to_end_test.rs` | B.4 | `QWEN25_SAFETENSORS_PATH` |
| `tests/qwen25_numerical_validation_test.rs` | B.5 | `QWEN25_SAFETENSORS_PATH` |
| `tests/llama_3_2_end_to_end_test.rs` | C.4 | `LLAMA32_SAFETENSORS_PATH` |
| `tests/llama_3_2_numerical_validation_test.rs` | C.5 | `LLAMA32_SAFETENSORS_PATH` |
| `tests/tinyllama_f64_validation_test.rs` | M4.6.1 | `TINYLLAMA_SAFETENSORS_PATH` |

Each `_end_to_end_test.rs` runs in 30–115 s. Each
`_numerical_validation_test.rs` is the same plus fixture
deserialisation; total similar.

### Fixtures committed

| Path | Files | Purpose |
|------|-------|---------|
| `tests/fixtures/smollm2_reference/` | `generate.py`, `generate_f64.py`, `expected_logits.json` (~3.0 MB), `expected_logits_f64.json` (~3.9 MB), `inputs.json`, `requirements.txt`, `README.md` | A.1 + Investigation F |
| `tests/fixtures/qwen25_reference/` | same 7 files | B.5 |
| `tests/fixtures/llama32_reference/` | same 7 files | C.5 |
| `tests/fixtures/tinyllama_reference/` | added `generate_f64.py`, `expected_logits_f64.json` (~2.6 MB) | M4.6.1 |

### Regressions verified at every sub-phase

The five-suite regression battery
(`tinyllama_config_test`, `tinyllama_weight_loading_test`,
`tinyllama_builder_test`, `weight_mapper_test`,
`miniflux_safetensors_roundtrip_test`) plus `rope_test` plus the
heavy numerical validations ran clean after every M4.6 sub-phase
commit. Three pre-existing flaky tests
(`apx_7_3_adaptive_learns_best_strategy`,
`apx_7_6_hpge_performance_sanity`,
`length_mismatch_is_error`) fail under concurrent full-suite
execution but pass in isolation; verified pre-existing on `main`
via `git stash`. They are not regressions of any M4.6 work.

**Total fast tests added in M4.6: 32 (29 inline + 3 ignored
on-disk).**
**Total `#[ignore]`-gated heavy tests added: 7.**
**Total committed fixtures: 4 reference directories
(~16.4 MB on disk in JSON).**

---

## How to resume on M4.7 (or M4.6.2)

1. **Read this file and the M4.5 handoff in that order**. The
   M4.5 invariants are still in force; M4.6's are layered on top.
   Pay special attention to the **Architectural decisions locked**
   sections.
2. **Pick one model up front, ground in its `config.json`**.
   Resist the temptation to "do all the small models at once".
   Phase A through Phase C each delivered exactly one new
   architecture per Phase, and each Phase took roughly the same
   total effort despite vastly different surface areas.
3. **Investigation-previa is non-negotiable**. The Phase B and
   Phase C investigations each surfaced exactly one architectural
   surprise that would otherwise have become a multi-hour debug
   session: B's K-bias-scale-absorption requirement and C's H7
   long-context falsifier. The investigations cost ~30 min each.
4. **Run the M4.6 regression battery to confirm a clean baseline**
   before writing new code:
   ```
   cargo test --test rope_test --test permute_test \
              --test batch_matmul_4d_test --test broadcast_mul_test \
              --test rms_norm_eps_test --test softmax_adversarial_test \
              --test tinyllama_config_test --test tinyllama_weight_loading_test \
              --test tinyllama_builder_test --test weight_mapper_test \
              --test miniflux_safetensors_roundtrip_test
   ```
   Plus optionally the four numerical validations under `--ignored`
   if the four model `*.safetensors` files are checked out:
   ```
   cargo test --release \
              --test tinyllama_f64_validation_test \
              --test smollm2_numerical_validation_test \
              --test qwen25_numerical_validation_test \
              --test llama_3_2_numerical_validation_test \
              -- --ignored --nocapture
   ```
5. **Sub-phase rhythm: parser → primitive (if any) → builder →
   mapper → smoke → numerical**. Phases B and C both followed
   exactly that order. Each sub-phase commits independently. Each
   sub-phase passes the regression battery before the next starts.
   Smoke and numerical can land together if seq=4 alone
   exercises the new path materially; otherwise pair the smoke
   with a falsifier (C.6 is the template).
6. **F64 fixture is the close criterion**. ADR-004 is binding:
   `max_atenia_vs_f64 < 0.5`. Both BF16 and F64 fixtures land
   committed; only F64 gates pass/fail.
7. **For M4.7 specifically**: M4.6 did not exercise the
   reaction-loop infrastructure at all. Plan to spend the first
   day reading the M3 reactive code and the v20 hardware
   adaptation code paths before writing any new tier-transition
   logic. The empirical numbers in M4.6's "Performance
   observations" are the baseline against which M4.7's pressure-
   triggered tier transitions will be measured.

---

## Observations from the M4.6 sprint

Recorded so the in-flight decisions are not lost.

- **Investigation-previa caught two architectural surprises that
  would have wasted multi-hour debug sessions**:
  1. **K bias scale absorption (Phase B.3)**. Reading
     `modeling_qwen2.py` while writing the weight-mapper handler
     surfaced that PyTorch's
     `scores = Q @ (h W_k + b_k)^T / sqrt(d_k)` requires the K
     bias to be divided by `sqrt(d_k)` along with the K weight,
     because the scale is pre-folded at load time. Without this,
     biased attention scores would diverge by a factor of `√64`.
     Caught analytically before the first smoke test ran.
  2. **Llama 3 long-context falsifier (Phase C, H7)**. The Phase
     C investigation explicitly identified that a `seq_len = 4`
     forward could pass even with completely-broken rope_scaling,
     because all positions sit in the high-frequency band where
     scaled and unscaled `inv_freq` coincide. C.6 was added to
     the plan as a result, with a synthetic 2048-position graph
     test that bit-exactly proves the scaled vector reaches the
     RoPE kernel. Without H7 the milestone could have closed
     green and silently broken on long contexts.

- **The "BF16 mismatch" of M4.5-d.1 was BF16, not Atenia**.
  M4.5-d.1's pos-3 argmax disagreement (Atenia 29871, PyTorch
  595) led to a careful "top-1 near-tie" interpretation at the
  time — correct as far as it went, but the real explanation
  surfaced only with M4.6.1's F64 fixture: both 595 and 29871
  round to logit 8.0625 in BF16 (a quantisation-bucket
  collision), and the BF16 argmax bounces between them
  depending on which is iterated first. F64 has no such tie
  (29871 wins by ~0.04), and Atenia's F32 result agrees with
  F64 to within 1e-5. The M4.5 framing was honest but slightly
  generous to BF16; the M4.6.1 framing is the right one.

- **The investigation → sub-phase → regression rhythm worked
  without exception**. Three Phases (A, B, C) plus M4.6.1 all
  followed the same template. No sub-phase had to be rolled
  back, no regression check ever failed, no architectural
  decision was re-litigated mid-Phase. The pattern is
  reproducible — it is not a property of the specific subject
  matter.

- **`Option<T>`-as-config-extensibility kept old configs working
  while letting new ones express more**. Every M4.6 config
  addition (`attention_bias: Option<bool>`, `model_type:
  Option<String>`, `head_dim: Option<usize>`, `rope_scaling:
  Option<RopeScaling>`) was additive and parsed `None` for the
  M4.5 baseline configs. No M4.5 test required updating to
  accommodate M4.6 fields; the four M4.5 tests that do reference
  `attention_bias` directly continued to compile because
  `Some(false)` is the natural lift of `false`. Future
  config-extensibility work should follow the same pattern.

- **F64 fixture generation is the bottleneck on the model
  selection**. SmolLM2 1.7B in F64 needs ~14 GB peak; Qwen 2.5
  1.5B needs ~16 GB; Llama 3.2 1B needs ~12 GB. The 1.5–1.7B
  range is the upper bound on a 16 GB system. Future models in
  the 3B+ range will need either dedicated hardware or a
  layer-by-layer F64 conversion strategy (out of scope for
  M4.6). The methodology is robust; the operational ceiling is
  hardware.

- **Scope discipline closed M4.6 cleanly**. Phi 3.5 mini and
  SmolLM3 were both in the original M4.6 candidate list
  (per the M4.5 handoff). Both got deferred at planning time:
  Phi for lack of urgency, SmolLM3 because it's sharded and
  multi-file loading is its own work item. The Phase A model
  swap (SmolLM3 → SmolLM2) was made on the same grounds. Each
  deferral was an explicit decision recorded at the time, not a
  drift. The M4.6.1 retroactive close was the only post-Phase
  addition, and it was scoped narrowly (one fixture + one test,
  no engine changes).

- **Three orchestrator context resets happened mid-milestone
  without methodology drift**. The investigation-previa →
  sub-phase → regression pattern survived two cold-start
  resumptions of the planning side and one inside the
  implementation side. The pattern is robust to actor changes
  because every Phase begins with the investigation report (a
  full-context document) and every sub-phase ends with a
  regression sweep (a full-context verification). No state is
  carried in conversation memory.

- **Performance optimization is deferred by design, not by
  omission**. The project principle "make it work, make it
  right, make it fast" applies in order. M4.5 closed "make it
  work" for TinyLlama. M4.6 closes "make it right" for the
  Llama family — F64 ground truth validated across four models,
  Atenia three-to-four orders of magnitude closer to mathematical
  truth than industry-default BF16 inference on every checkpoint.
  "Make it fast" is the next stage in that order, but it
  requires empirical data from a workload that exercises the
  reaction loop and tier transitions (M4.7's beyond-VRAM
  scenario) to avoid optimising the wrong bottleneck. The
  M4.6 forward timings are recorded as baselines for that
  future comparison; they are not a punch-list for
  optimisation work in isolation.

- **Test-suite hygiene continues to pay back**. The M4.6
  regression battery runs in seconds, which made the
  "always run regressions before declaring a sub-phase green"
  discipline maintainable across 14 commits. The three flaky
  tests inherited from M3/M4 were occasionally noisy under
  concurrent full-suite execution but never blocked any sub-
  phase close — verified pre-existing each time. They remain
  catalogued for future cleanup, not as M4.6 regressions.
