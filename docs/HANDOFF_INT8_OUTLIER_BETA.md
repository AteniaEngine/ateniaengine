# HANDOFF — INT8 outlier decomposition (Track β)

**Status:** β.5 FAIL — outlier decomposition does not satisfy
ADR-004 strict on the smallest M8.5 fixture model under
end-to-end forward measurement. Track β stops at β.5; β.6 CUDA
work is **not** authorised. See §7 for the verdict and §8 for the
recommended pivot.
**Predecessor:** M9 INT8 W8A16 (closed as experimental / opt-in;
failed ADR-004 strict drift gate on 4/4 models under absmax).
**See also:** `HANDOFF_APX_V20_M9.md` §4 (the M9 drift cliff
analysis that motivated Track β), §10 "Option β".

---

## 0. Why this milestone

M9 shipped a working INT8 W8A16 infrastructure with a ~2×
microbench speedup but failed ADR-004 strict (`max_abs_diff < 0.5`
vs F64 truth) on every model in the four-checkpoint fixture under
all three absmax strategies (per-channel, per-group g=32, per-group
g=128). Root cause analysis in M9 §4 identified per-column **weight
outliers** as the dominant source of cascade drift: a handful of
high-magnitude columns fix the per-column absmax scale and crush
the effective resolution of the remaining elements.

Track β tests the hypothesis that **removing those outlier columns
from the INT8 base and preserving them exactly in a small dense
sidecar** lifts the per-element envelope enough to satisfy
ADR-004 once the path is wired end-to-end.

---

## 1. The sub-phase ledger

| Phase | Title | Status | Commit |
| ----- | ----- | ------ | ------ |
| β.1   | CPU outlier-decomposition spike + quantizer API | ✅ | `0d40e3f` |
| β.2   | `TensorStorage::CpuInt8Outlier` + constructor + reconstruction | ✅ | `1354e7d` |
| β.3   | CPU MatMul integration (via β.2 ensure_decoded) | ✅ | `7d6dab5` |
| β.4   | F64-fixture-model per-tensor validation | ✅ | `c963d39` |
| β.5   | `SharedParam::CpuInt8Outlier` + end-to-end forward F64 | ⚠️ FAIL | (this commit) |
| β.6   | CUDA mixed-precision kernel (INT8 base + BF16 correction) | ❌ NOT AUTHORISED | — |
| β.7   | Docs / release / production routing | ❌ NOT AUTHORISED | — |

---

## 2. β.4 design

The β.4 question — "does CpuInt8Outlier reduce real-world drift
versus M9 plain INT8?" — has two layers:

1. **Per-tensor reconstruction error.** For each `_proj.weight`
   in the four M8.5 fixture models, compute the max-abs error
   between the original F32 weight and (a) its plain
   per-group absmax INT8 round-trip vs (b) its β
   outlier-decomposed round-trip. **This is what β.4 measures.**

2. **End-to-end logit drift vs F64.** ADR-004's actual gate.
   Requires injecting `CpuInt8Outlier` into `WeightStore` and
   running the full forward; the store today exposes only
   `F32 / Bf16 / Cuda / Disk` variants. Wiring the new variant
   into the store and the loader is **β.5**, not β.4 (scope
   rule: no productive-loader changes in β.4).

The per-tensor signal **is the necessary condition** for the
end-to-end signal: if the outlier reconstruction does not strictly
beat plain INT8 at the tensor level, the cascaded drift cannot be
better either. β.4 therefore acts as a **kill-switch gate** — if
the per-tensor reconstruction had regressed on any single layer
the milestone would abort Track β before paying β.5's
store-rewiring cost.

The harness lives in `tests/int8_outlier_f64_validation_test.rs`.
It is gated behind `#[ignore]` because it reads multi-gigabyte
safetensors files. Lightweight tests (no real model) ship in the
same file and run under default `cargo test --lib`.

---

## 3. Setup

```powershell
$models = "F:\Proyectos\artenia_engine\atenia-engine\models"
$env:TINYLLAMA_SAFETENSORS_PATH = "$models\tinyllama-1.1b\model.safetensors"
$env:SMOLLM2_SAFETENSORS_PATH   = "$models\smollm2-1.7b-instruct\model.safetensors"
$env:QWEN25_SAFETENSORS_PATH    = "$models\qwen2.5-1.5b-instruct\model.safetensors"
$env:LLAMA32_SAFETENSORS_PATH   = "$models\llama-3.2-1b-instruct\model.safetensors"
cargo test --release --test int8_outlier_f64_validation_test `
    -- --ignored --nocapture
```

Configuration parameters (locked for β.4):

- `group_size = 128` — M9.4 production value, empirical winner
  of the M9 fixture matrix.
- `outlier_k = 64` — audit recommendation. Aligns with typical
  Llama-family `head_dim ∈ {64, 128}` boundaries.

---

## 4. Results

Per-tensor max-abs reconstruction error across every
`_proj.weight` of each model (630 tensors total):

```
========================================================
β.4 SUMMARY (per-tensor max-abs reconstruction error)
group_size = 128,  outlier_k = 64
--------------------------------------------------------
Model                   tensors      plain_max    outlier_max   median_x    worst_x
TinyLlama 1.1B              154      1.2215e-2      1.5624e-3      2.54x      1.22x
SmolLM2 1.7B                168      3.6377e-2      2.5298e-2      2.56x      1.21x
Qwen 2.5 1.5B               196      1.2029e-2      2.3054e-3      1.93x      1.25x
Llama 3.2 1B                112      4.6262e-3      2.3943e-3      2.27x      1.22x
========================================================
```

`plain_max` / `outlier_max` are the worst max-abs reconstruction
error across all `_proj.weight` tensors of the model.
`median_x` / `worst_x` are the median / worst per-tensor
improvement ratio (plain ÷ outlier).

### Headline findings

- **All 4 models pass the per-tensor gate.** Worst-case
  per-tensor improvement is ≥ 1.21× on every model; outlier
  strictly dominates plain INT8 on every single one of the 630
  measured tensors. No regression.
- **Median improvement ≈ 2×** across all models, consistent with
  the β.1 synthetic prediction (≥ 5×) accounting for real-world
  outlier sparsity being lighter than the synthetic stress-test.
- **TinyLlama is the cleanest case** (outlier_max 1.56e-3,
  ~8× headline improvement).
- **SmolLM2 remains the sentinel.** Its M9 g=128 drift was 11.25
  (22× over the gate); β reduces the worst per-tensor envelope
  from 3.64e-2 to 2.53e-2 (~1.4× headline). Better, but still
  the largest absolute envelope of the four.

### Projection to ADR-004

Using the M9 §4 drift cascade equation
`end-to-end ≈ envelope_per_op · sqrt(num_ops · alpha)`:

| Model        | Envelope (outlier) | Matmuls | sqrt(N) | Projected drift |
| ------------ | ------------------ | ------- | ------- | --------------- |
| TinyLlama    | 1.56e-3            | ~154    | ~12     | ~0.019          |
| Llama 3.2 1B | 2.39e-3            | ~112    | ~11     | ~0.026          |
| Qwen 2.5 1.5B| 2.31e-3            | ~196    | ~14     | ~0.032          |
| SmolLM2 1.7B | 2.53e-2            | ~168    | ~13     | **~0.33**       |

The projection puts three of four models comfortably under the
ADR-004 0.5 gate. **SmolLM2 lands at ~0.33** — under the gate
in the linear projection, but the M9 fixture showed real cascade
amplification factors well above sqrt(N) for this model
specifically (11.25 vs ~0.5 linear projection). So:

- **Confidence Track β passes for TinyLlama, Llama 3.2 1B,
  Qwen 2.5 1.5B:** high.
- **Confidence Track β passes for SmolLM2 1.7B:** moderate — the
  per-tensor improvement is real but the cascade behaviour on
  this model has historically over-amplified. β.5 measurement
  will be decisive.

---

## 5. Verdict

**Avanzar a β.5.** The per-tensor signal is strictly positive on
all four production models; there is no model where the policy
regresses. β.5 (`SharedParam` extension + loader integration)
unlocks the end-to-end F64 measurement that ADR-004 actually
gates against.

If β.5 shows SmolLM2 cascade-amplifies above 0.5, the next pivot
options (in increasing cost) are:

- Bump `outlier_k` from 64 → 128 or 256 for SmolLM2 specifically
  (per-model policy, light change).
- FFN-only routing (skip attention proj weights — M9.5 Option α).
- GPTQ / AWQ calibration (Option γ in the M9 handoff).

We commit to revisiting this branch point with β.5 numbers before
investing in β.6 CUDA work.

---

## 6. β.5 — end-to-end forward measurement

### What β.5 added

- `SharedParam::CpuInt8Outlier(Arc<Tensor>)` — additive variant
  on `src/amg/weight_store.rs`, with `to_tensor` / `shape` /
  `resident_bytes` / `strong_count` arms.
- `WeightStore::quantize_param_to_outlier(idx, group_size, k)` —
  opt-in conversion helper. Reads the existing F32 or BF16
  shared variant, materialises F32, runs
  `decompose_outliers_topk_by_absmax`, and replaces the slot
  in place. Cuda / Disk variants panic by contract (β.5 is
  CPU-only experimental).
- Heavy F64 forward test `beta5_tinyllama_outlier_forward_reports_adr_004`
  in `tests/int8_outlier_f64_validation_test.rs`. Loads
  TinyLlama on a CPU-only `WeightStore` (`kernel_dtype = F32`),
  records the certified-baseline drift, converts all 154
  `_proj.weight` parameters to `CpuInt8Outlier`, runs the
  forward, and prints the ADR-004 verdict.
- Exhaustiveness arms in legacy tests (`tensor_storage_test.rs`,
  `tensor_storage_cuda_test.rs`, the four `m3_e_11_*` migration
  tests, `m5_db_tinyllama_pipeline_test.rs`,
  `m5_dc_llama2_13b_coherence_test.rs`) — panic / unreachable
  matchings the existing pattern for variants the legacy tests
  do not exercise. No behaviour change.

### β.5 results — TinyLlama 1.1B end-to-end (CPU F32 forward, 4 positions)

```
========================================================
β.5 TinyLlama 1.1B — full forward F64 comparison
========================================================
                              certified           outlier
group_size                          n/a               128
outlier_k                           n/a            64 / 256
drift vs F64                   6.3e-5         1.200 / 1.143
argmax 4/4                       true               false
ADR-004 (< 0.5)                  PASS                FAIL
========================================================
```

- **Certified baseline** (F32 throughout): drift 6.3e-5, argmax
  4/4 MATCH, ADR-004 PASS by a ~8000× margin. Confirms the
  harness, the model, the F32 path and the F64 fixture are all
  internally consistent.
- **Outlier path k=64** (the audit's default): drift 1.200,
  ADR-004 FAIL by 2.4×. Argmax 0/4 MATCH.
- **Outlier path k=256** (sensitivity probe): drift 1.143,
  ADR-004 FAIL by 2.3×. Quadrupling the sidecar buys ~5%
  improvement.

### Why the β.4 projection was wrong

β.4 measured per-tensor max-abs reconstruction error and applied
the linear cascade approximation
`end-to-end ≈ envelope · sqrt(num_ops)`. The TinyLlama outlier
envelope was 1.56e-3 → projected drift ~0.019.

The actual measured drift was **1.20** — roughly **60× the linear
projection**. The β.4 sentinel SmolLM2 already hinted this could
happen (its M9 cascade amplified ~22× over the linear projection);
in TinyLlama (a model β.4 considered "the cleanest case") the
amplification is even larger.

Root cause analysis: weight-only INT8 with absmax-driven outlier
removal does not see *which* per-element errors will propagate
through RMSNorm gain × attention softmax × residual sum
amplification. The errors that survive into the matmul output
correlate with activation patterns, not with raw weight outlier
magnitude. This is the empirical confirmation of the calibration
gap that motivated GPTQ / AWQ in the first place: they use real
activations to pick the per-weight scale, not the weight
statistics in isolation.

---

## 7. Verdict

**STOP Track β before β.6.** The regression rule from the β.5
spec fires verbatim:

> 11. Si aparece cualquier regresión numérica fuerte:
>     - STOP
>     - reportar
>     - NO continuar hacia CUDA.

Drift 1.20 on the smallest, cleanest β.4 model against a 0.5 gate
is a strong regression. Investing β.6 CUDA work on a path that
does not satisfy ADR-004 would be money set on fire.

The β.5 harness itself is **kept** — the `SharedParam` variant,
the conversion helper and the end-to-end test are correct
scaffolding that any future calibration-based approach (GPTQ,
AWQ, SmoothQuant) can reuse. They cost ~600 LOC and unlock
two-line replacement experiments.

---

## 7.quater. β-pivot.3 — real-text AWQ calibration (mixed result)

After β-pivot.2 demonstrated that activation-aware scaling helps
(synthetic-cal α = 0.3 → drift 0.78, the best magnitude number
in the investigation), β-pivot.3 replaces the synthetic prompts
with real natural-language text tokenised via the existing
`AteniaTokenizer`.

### What β-pivot.3 changes

- New helper `build_real_text_calibration_tokens(seq, model_dir)`
  in the test harness. Loads `tokenizer.json` from the model
  directory, encodes 8 hardcoded English sentences with BOS,
  takes the first `seq = 4` token ids of each, casts to F32 to
  match the existing token-input convention.
- `run_calibration` swaps the 4 synthetic sequences for the 8
  real-text sequences when the tokenizer is reachable; falls
  back to the synthetic batch otherwise (CI safety).
- Activation walker, AWQ scale derivation, and perturbation
  helper are unchanged.

### Calibration prompts

```text
Hello, how are you?
The capital of France is Paris.
Rust is a systems programming language.
Machine learning models require careful numerical validation.
Once upon a time there was a small dragon.
The quick brown fox jumps over the lazy dog.
Artificial intelligence systems can drift numerically.
Quantization changes the numerical properties of inference.
```

8 prompts × first-4-tokens-with-BOS = 32 real-text token
positions feeding the calibration forward.

### β-pivot.3 results — TinyLlama 1.1B (real-text α sweep)

```
========================================================
β-pivot.3 TinyLlama — real-text calibrated AWQ α sweep
group_size = 128,  prompts = 8 (real text), seq = 4
--------------------------------------------------------
α       drift       argmax 4/4    note
0.20    1.337593       4/4 ✓      argmax preserved
0.25    0.889217       4/4 ✓      best PASS-PARTIAL with argmax
0.30    0.925051       0/4 ✗      magnitude drift, argmax broken
0.40    1.192629       4/4 ✓      argmax preserved
========================================================
```

### Comparison against the full investigation

```
                                           drift    argmax 4/4
certified F32 baseline                   0.000063     4/4 ✓
M9 plain INT8 (≡ AWQ α = 0)              1.260771     0/4 ✗
β.5  outlier k = 64                      1.200260     0/4 ✗
β.5  outlier k = 256                     1.142564     0/4 ✗
β-pivot.1 weight-norm AWQ α = 0.5        2.478883     0/4 ✗
β-pivot.2 synthetic-cal AWQ α = 0.3     *0.782183*    0/4 ✗   ← min magnitude
β-pivot.3 real-text  AWQ α = 0.25       *0.889217*    4/4 ✓   ← min with argmax intact
β-pivot.3 real-text  AWQ α = 0.30        0.925051     0/4 ✗
ADR-004 gate                              < 0.5     argmax-irrelevant
========================================================
```

### Honest interpretation

- **Real-text calibration did NOT reduce magnitude drift.**
  β-pivot.2 synthetic α = 0.3 remains the best magnitude number
  at 0.78; β-pivot.3 real-text α = 0.25 is 0.89, ~14 % worse.
- **Real-text calibration DID restore argmax matches.**
  β-pivot.2 synthetic at every α gave 0/4 argmax; β-pivot.3 at
  α ∈ {0.20, 0.25, 0.40} gives 4/4 argmax. The synthetic
  prompts apparently push the BOS-position activations into a
  regime where the AWQ scale flips the relative ordering of
  large-logit tokens, even though the absolute magnitudes were
  closer to truth.

This is a real signal but it splits the metrics: **drift**
(ADR-004) and **argmax** (generative correctness) diverge under
the two calibration modes. ADR-004 is the formal gate; argmax
recovery is what would matter for real generation quality.

### Why drift did not improve

Two probable causes:

1. **`seq = 4` is too short to average out BOS noise.** The
   first token (BOS) lands the embedding in a distinct
   region of activation space; with only 3 real-text tokens
   following, the max-of-max statistic over-weights that
   transient. The β-pivot.2 synthetic prompts had every
   position carry "real-magnitude" ids so the BOS bias was
   not as pronounced relative to the rest.
2. **8 short prompts may underdetermine per-channel stats.**
   AWQ literature uses ~128 calibration samples of seq ≥ 512.
   We are 1–2 orders of magnitude under both axes.

Either or both could be addressed without leaving the AWQ
family. Neither requires a pivot to GPTQ today.

### Verdict

⚠️ **PASS PARTIAL — continue AWQ refinement. Do NOT pivot to
GPTQ yet.**

The β-pivot.3 contract rule:
> 13. Si TinyLlama sigue > 0.9: recomendar GPTQ seriamente.

The best β-pivot.3 number is 0.89, **just under** the 0.9
threshold. Combined with the argmax 4/4 recovery, this is a
mixed signal — not strong enough to declare progress, not weak
enough to fire the GPTQ rule.

Ladder of next experiments, in order of cost-to-confidence:

1. **β-pivot.4 — per-tensor α grid search.** For each
   `_proj.weight` independently, pick the α that minimises
   post-perturbation reconstruction loss against the
   un-perturbed F32 weight. The 0.78 / 0.89 / 1.19 range from
   the global-α sweep suggests different layers want
   different α; per-tensor search should close ~10–30 % of
   the gap. ~1 day of work.
2. **β-pivot.5 — AWQ + outlier hybrid.** The β.5 outlier
   carve-out and the β-pivot.2/.3 activation scaling target
   orthogonal failure modes (high-magnitude weight columns
   vs. activation-sensitive channels). Stacking should be
   additive on the modes' independent components. ~½–1 day
   of harness work (storage already exists).
3. **β-pivot.6 — longer / more prompts.** Increase
   `runtime.seq` to 16–32 and add 16–32 more prompts. Forces
   a graph rebuild (different runtime) but no new code.
4. **GPTQ fallback.** Only if (1) + (2) + (3) all exhaust
   without crossing the gate.

---

## 7.ter. β-pivot.2 — calibrated AWQ (PASS PARTIAL)

After β-pivot.1 falsified the *no-calibration* simplification,
β-pivot.2 replaces the weight-norm proxy with **real activation
statistics captured during a calibration forward**.

### Activation capture

A graph walker iterates `Graph::nodes` after each calibration
forward, finding every `NodeType::MatMul` whose RHS is a store
parameter ending in `_proj.weight`. For each such matmul it reads
the LHS node's `.output` tensor and accumulates per-input-channel
absmax. The walker is non-invasive — it adds no hooks inside the
engine; it relies on the executor leaving `node.output`
populated post-forward, which the current implementation does.

```text
ActivationStats {
    absmax: Vec<f32>,    // length K (input-channel count)
    sample_count: usize, // for diagnostics
}
```

Stats from multiple prompts are merged with max-of-max semantics
(the conservative envelope).

### Calibration prompts

Four small synthetic token sequences. No tokenizer dependency,
no external dataset. The experiment measures whether *any* real
activation signal beats the β-pivot.1 weight-norm proxy:

```text
[1.0, 100.0, 200.0, 300.0]
[50.0, 250.0, 450.0, 650.0]
[10.0, 20.0, 30.0, 40.0]
[500.0, 600.0, 700.0, 800.0]
```

### Scale formula

```text
s[k] = (activation_absmax[k])^α     (per input channel)
s[k] /= mean(s)                      (normalise to unit mean)
W'[k, n] = W[k, n] · s[k]            (pre-scale)
(q, group_scales) = absmax_per_group(W', g)
W_recon[k, n] = (q[k, n] · group_scales[g, n]) / s[k]
```

### β-pivot.2 results — TinyLlama 1.1B (α sweep)

```
========================================================
β-pivot.2 TinyLlama — calibrated AWQ α sweep
group_size = 128,  prompts = 4 (synthetic), captured tensors = 154
--------------------------------------------------------
                drift     vs prior best (β.5 = 1.20)
α = 0.0       1.26        ≈ plain INT8 control
α = 0.1       1.12        −7 %  vs β.5
α = 0.2       0.84       −30 %
α = 0.3       0.78       −35 %  ← minimum
α = 0.5       1.04       −13 %
α = 0.7       1.25         0 %  ≈ plain INT8 again
========================================================
```

### Headline

- **α = 0.3 gives the best drift: 0.78** on TinyLlama against
  the F64 fixture.
- That is the **best number across every quantisation
  experiment in this whole investigation** (plain INT8 / β
  outlier / β-pivot.1 weight-norm AWQ / β-pivot.2 calibrated AWQ).
- The U-shaped curve around α = 0.3 confirms that
  activation-aware scaling is the right *direction*: the
  weight-norm proxy (β-pivot.1) actively distorts, real
  activation stats actively help.
- ADR-004 gate is **still not met** (0.78 vs the 0.5 cap, FAIL
  by ×1.56). The β-pivot.2 spec calls this regime "PASS
  PARTIAL": 1.20 → 0.78 is a 35 % drift reduction; AWQ remains
  viable but the simplified spike does not close the gap alone.

### Why it does not (yet) clear the gate

- Calibration uses 4 **synthetic** token sequences, not real
  text. Real activation distributions on natural language
  vocabulary are more spread; better calibration data should
  shrink drift further.
- A single global `α` is applied to every `_proj.weight`. The
  paper's recipe optimises per-tensor `α` via a small grid
  search per layer (`α ∈ {0.0, 0.05, ..., 1.0}` minimising
  per-layer reconstruction loss). β-pivot.2 picks one `α` for
  the whole model.
- The stats use `max_abs` only. Mean-abs + percentile fusion
  is the established choice and tightens the envelope.
- No outlier-column carve-out is combined with AWQ scaling. The
  β.5 sidecar and β-pivot.2 scale are orthogonal mechanisms;
  combining them is a separate experiment.

### Verdict

⚠️ **PASS PARTIAL — continue AWQ refinement, do NOT pivot to GPTQ
yet.**

The β-pivot.2 contract says:

> PASS parcial: TinyLlama: 1.20 → 0.6~0.9 → AWQ sigue viable.

0.78 lands inside that band. The natural next experiments
(ordered by cost-to-confidence ratio):

1. **β-pivot.3** — real calibration text. Replace the 4
   synthetic sequences with a fixed Pile-style mini-batch
   tokenised via the existing CLI tokenizer (8–32 prompts).
   ~½ day of work; expected drift reduction ~10–20 % based on
   the literature.
2. **β-pivot.4** — per-tensor α grid search. For each
   `_proj.weight` independently, pick the `α` that minimises
   the post-perturbation reconstruction loss against the
   un-perturbed F32 weight. ~1 day of work.
3. **β-pivot.5** — combine β outlier-column carve-out with
   β-pivot.2 calibrated scaling. The two mechanisms target
   orthogonal failure modes; stacking them might close the
   remaining ×1.56 gap.

GPTQ remains the fallback if these three exhaust without
passing the gate.

---

## 7.bis. β-pivot.1 — no-calibration AWQ spike

After β.5 failed on TinyLlama (drift 1.20, k=64), the audit
recommended AWQ as the first follow-up. β-pivot.1 implements a
**simplified, no-calibration variant**: per-K-row scales derived
from the weight L2 norm itself rather than from a real activation
calibration pass. The math is the standard AWQ pre/post-scale
identity collapsed to a CPU F32 perturbation:

```text
    W'[k, n] = W[k, n] · s[k]                    (pre-scale)
    (q, group_scales) = absmax_per_group(W', g)
    W_recon[k, n] = (q[k, n] · group_scales[g, n]) / s[k]

    where s[k] = (||W[k, :]||_2)^α, normalised to unit mean.
```

### What this implementation ships

- `quantizer::awq_per_row_scales_from_weight_norm(weights, shape, α)`
  → unit-mean per-K-row scales from the weight L2 norm.
- `quantizer::apply_awq_perturbation_inplace(weights, shape, g, scales)`
  → pre-scale → INT8 absmax → dequant → inverse-scale, all on a
  mutable F32 buffer. The runtime sees a plain F32 weight; no
  new storage variant is introduced.
- `WeightStore::perturb_param_with_awq(idx, group_size, α)` —
  opt-in helper that drives the perturbation through the
  existing F32/Bf16 `SharedParam` slots and reinserts as F32.
- Heavy F64 forward test
  `beta_pivot1_tinyllama_awq_forward_reports_adr_004` reusing
  the β.5 harness verbatim.

### β-pivot.1 results — TinyLlama 1.1B end-to-end

```
========================================================
β-pivot.1 TinyLlama 1.1B — AWQ vs F64 fixture
========================================================
Mode                       drift     argmax    ADR-004 (< 0.5)
certified F32 baseline   0.000063     4/4      PASS (8000x margin)
β.5  outlier k=64        1.200260     0/4      FAIL (2.4x over gate)
β.5  outlier k=256       1.142564     0/4      FAIL (2.3x over gate)
β-piv AWQ α=0.0          1.260771     0/4      FAIL (≡ plain INT8)
β-piv AWQ α=0.5          2.478883     0/4      FAIL — *worse than plain*
========================================================
```

### Interpretation

The α=0 row is the control: AWQ degenerates to plain per-group
absmax INT8 there, and the drift (1.26) matches the β.5 outlier
path (1.20) within the residual sidecar effect. This is the
**floor** of any weight-only INT8 strategy on TinyLlama.

α=0.5 is **worse** by ~2×. Without real activation statistics,
the weight-norm proxy actively distorts the model: rows with
large weight norm get amplified pre-quantisation, which gives
them more INT8 headroom, but they are not necessarily the rows
whose dequant errors will propagate most through RMSNorm gain
× attention softmax × residual sum. The proxy is uncorrelated
with the actual sensitivity.

### Verdict

**STOP β-pivot.1 AWQ (no-calibration variant) before further
investigation.** The numerical evidence is unambiguous:

- Outlier removal alone does not close the gap (β.5).
- Weight-norm AWQ scaling actively makes it worse (β-pivot.1).
- The α=0 floor (1.26) is intrinsic to plain INT8 cascade
  amplification on this model.

Any further INT8 weight-only progress requires **real activation
statistics**. Two viable next steps:

1. **AWQ with real calibration** — extend the forward harness
   with hooks that capture activation absmax per-K-axis from
   each `_proj.weight`'s LHS input during a calibration forward,
   feed those into `apply_awq_perturbation_inplace` instead of
   the weight-norm proxy. ~3–5 days of work (graph hook
   plumbing + per-tensor stats aggregation + per-weight scale
   table propagation).

2. **GPTQ (Hessian-inverse weight-only calibration)** — the
   established state-of-art for outlier-heavy LLMs.
   Mathematically different from AWQ (computes weight updates
   that minimise the local matmul error against a Hessian
   estimate, not a per-channel scale). ~1–2 weeks.

The audit's original recommendation order (AWQ first, GPTQ as
fallback) **is preserved** — β-pivot.1 falsified only the
*no-calibration* simplification, not the AWQ family itself. A
real-calibration AWQ retry is the appropriate next experiment.

---

## 8. Recommended pivot

In increasing cost / depth:

1. **GPTQ** (Hessian-inverse weight-only calibration). Standard
   answer for outlier-heavy Llama-class weights. Needs a small
   calibration set (~128 examples) and one pass of activation
   recording per model. Drift profile on Llama 7B in the
   literature: ~0.05 vs F64 on 4-bit, ~0.005 on 8-bit. Cost:
   ~2 weeks new infrastructure (Hessian inverse op,
   calibration runner, per-tensor scale optimisation).
2. **AWQ** (activation-aware weight scaling). Lighter than
   GPTQ: detects salient channels from activation magnitude
   statistics, applies a per-channel scaling factor before
   quantisation. No Hessian. Drift profile is similar to GPTQ
   in practice and the implementation is shorter (~1 week).
3. **FFN-only mixed precision** (Option α from the M9
   handoff). Keep attention proj weights BF16; quantise only
   `mlp.*_proj.weight`. Halves the cascade depth without any
   new calibration concept. Cheaper but may still leave
   SmolLM2 above the gate, and offers half the memory win.
4. **Accept M9 INT8 as opt-in only.** Stop pursuing strict
   ADR-004 for INT8 weights; document the M9 drift envelope
   as the operator-visible cost. The infrastructure already
   ships at v0.1.0.

The audit's recommendation is **(2) AWQ** as the first
follow-up: lowest cost-to-confidence ratio, well-trodden
implementation path (Tencent / MIT-HAN papers + open source).
If AWQ also fails on SmolLM2, escalate to GPTQ.

---

## 9. Scope confirmations

β.4 touches **none** of:

- runtime core
- CUDA dispatch
- production loader
- tier planner
- adapters
- generation
- CLI
- numcert manifest schema
- M9 INT8 path (bit-identical when the β.4 harness is not
  invoked)

The harness lives entirely in `tests/` + this document.
