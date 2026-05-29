# Atenia Quantization Search (AQS) — Architecture Audit

**Status:** design audit only. No code changes, no commits, no branches.

**Author of this audit:** consolidates the β / β-pivot investigation
(M9 INT8, outlier decomposition, calibrated AWQ, real-text calibration,
hybrid). Numerical references throughout cite the empirical results
recorded in [`HANDOFF_INT8_OUTLIER_BETA.md`](./HANDOFF_INT8_OUTLIER_BETA.md).

**Reader:** anyone evaluating whether to commit engineering time to AQS
as the next strategic surface for Atenia Engine.

---

## 0. Executive summary

**AQS is a search-and-certification harness, not a quantization
algorithm.** Its job is to take a set of weight quantization *techniques*
(plain INT8, GPTQ, AWQ, hybrid, BF16 fallback, FP8) and a model, then
**automatically discover and certify** the per-tensor (or per-layer)
policy that minimises a memory/latency objective subject to a hard
ADR-004-class numerical drift bound.

**Why this is worth building.** Every other inference runtime ships
quantized checkpoints with hand-picked methods (`Q4_K_M`, `AWQ-4bit`,
`bitsandbytes`) and *no measured drift envelope*. Atenia already has:

  - a working F32 / BF16 / INT8 runtime that can be selectively replaced
    per-parameter at the `SharedParam` level (β.5 proved this),
  - an F64 reference fixture per model (ADR-004 contract) that makes
    drift measurable, not estimated,
  - the activation-capture walker and calibration loop from β-pivot.3,
  - the numcert manifest schema v2.0.0 with per-tensor policy support
    (already in production for `Certified` / `Fast` / `Quantized` modes).

That triple — *measurable drift*, *per-tensor policy substrate*,
*calibration harness* — is what no competitor has. AQS would turn it
into a product: **"every Atenia checkpoint ships with a manifest the
runtime can verify against F64 ground truth, and the manifest was
discovered automatically."**

**What is NOT worth building (yet).** A fully autonomous, hardware-
aware, multi-technique greedy search across all tensors of all models
on all platforms. That is the marketing fantasy. The first three AQS
milestones must be **brutally narrow**: integrate one new technique
(GPTQ), wire the search-and-verify loop for one model on one
platform, ship one manifest format with reproducibility guarantees.

**Bottom line.** AQS is the strongest product differentiator Atenia
can ship in 2026 *if* scoped surgically. It is also the easiest way
to ship 18 months of unmaintainable code if scoped optimistically.
This audit recommends the surgical path.

---

## 1. Phase 1 — Current state audit

### 1.1 Inventory: what already exists

| Component | File / type | β / β-pivot use | AQS fit |
|---|---|---|---|
| `quantizer::absmax_per_channel_symmetric` | `src/tensor/quantizer.rs` | M9, β.5 base | **reusable** |
| `quantizer::absmax_per_group_symmetric` | `src/tensor/quantizer.rs` | M9.4, β.5, AWQ | **reusable** |
| `quantizer::decompose_outliers_topk_by_absmax` | `src/tensor/quantizer.rs` | β.1, β-pivot.5 | **reusable** |
| `quantizer::apply_awq_perturbation_inplace` | `src/tensor/quantizer.rs` | β-pivot.1/2/3 | **reusable** |
| `quantizer::apply_hybrid_awq_outlier_perturbation_inplace` | `src/tensor/quantizer.rs` | β-pivot.5 | **reusable** |
| `quantizer::awq_per_row_scales_from_weight_norm` | `src/tensor/quantizer.rs` | β-pivot.1 | **reusable as baseline** |
| `quantizer::awq_per_row_scales_from_activations` | `src/tensor/quantizer.rs` | β-pivot.2/3 | **reusable** |
| `TensorStorage::CpuInt8` + `CpuInt8Outlier` | `src/tensor/tensor.rs` | M9, β.2 | **reusable** |
| `SharedParam::CpuInt8Outlier` | `src/amg/weight_store.rs` | β.5 | **reusable** |
| `WeightStore::perturb_param_with_*` (4 variants) | `src/amg/weight_store.rs` | β.5, β-pivot.{1,2,5} | **reusable** |
| F64 fixture loader + drift measurement | `tests/m8_5_full_family_validation_test.rs` | M8.5 / β.4 / β.5 | **reusable** |
| Activation walker `capture_proj_activation_stats` | `tests/int8_outlier_f64_validation_test.rs` | β-pivot.2/3 | **reusable** (move out of `tests/`) |
| Calibration loop `run_calibration` | `tests/int8_outlier_f64_validation_test.rs` | β-pivot.2/3 | **reusable** (same) |
| Real-text calibration via `AteniaTokenizer` | `tests/int8_outlier_f64_validation_test.rs` | β-pivot.3 | **reusable** |
| numcert manifest schema (v1.0.0 + v2.0.0) | `src/nn/llama/numcert.rs` | production | **needs schema bump** |
| `MatmulMode::{Certified, Fast, Quantized}` | `src/nn/llama/numcert.rs` | production | **needs extension** |
| `PerTensorPolicy` + glob matching | `src/nn/llama/numcert.rs` | production | **reusable** |
| Adapter Toolkit v2 (DSL) | `src/adapter_toolkit/` | production | **orthogonal** (model identity, not quant) |
| Tier planner cost model | `src/gpu/tier_plan.rs` | production | **needs extension** (per-policy bytes) |
| Loader weight mapping | `src/v17/loader/weight_mapper.rs` | production | **needs extension** (manifest-driven quant routing) |

### 1.2 Inventory: what is missing

| Component | Why it does not exist | AQS need |
|---|---|---|
| GPTQ (Hessian-inverse) | β-pivot family stopped at the plateau | **first new algorithm to integrate** |
| Persistent search state (e.g. SQLite) | every β experiment was one-shot | **needed for any non-trivial search** |
| Per-prompt/per-token-position drift breakdown | M8.5 only records max + argmax | **needed to debug per-layer sensitivity** |
| Engine-side activation hooks | β-pivot used post-execute walker | **needed for long-sequence calibration** |
| Latency / memory measurement harness | β tracked only drift | **objective function input** |
| Hardware probe → policy resolver | every test runs on the dev box | **needed for "validated_on" claims** |
| FP8 / W4 paths | none implemented | **search-space placeholders only** |
| Sweep CLI (`atenia search`) | not a planned CLI surface | **product UX** |

### 1.3 Inventory: what needs redesign

| Component | Current shape | Why redesign | AQS shape |
|---|---|---|---|
| `WeightStore::perturb_param_with_*` family | four ad-hoc methods, each panicking on unsupported variants | proliferating helper sprawl, ad-hoc validation | unify under a `QuantizationPolicy` trait that owns its own state and validation |
| Activation walker (currently in `tests/`) | non-invasive post-execute walk | usable only for inference-only short forwards; loses LHS to executor cleanup in some paths | replace with explicit engine-side hooks (opt-in, gated by env or builder flag) |
| numcert `MatmulMode` enum | closed enum: `{Certified, Fast, Quantized}` | every new technique requires an enum variant + arm-update blast | open registry / tagged-string with strict schema validation per known tag |
| F64 fixture coverage | 4 models × 4 token positions | very coarse for per-layer sensitivity attribution | per-layer activation snapshots + per-position drift slices |

### 1.4 Single-sentence audit verdict

**~70 % of the AQS substrate already exists; the missing pieces are
(a) one new technique (GPTQ), (b) the search-and-verify loop, and
(c) the manifest schema evolution.** The β investigation incidentally
built most of the infrastructure for a real product without anyone
naming it.

---

## 2. Phase 2 — Problem definition

### 2.1 Objective function

AQS optimises a **constrained multi-objective** problem:

```text
minimize:    α₁ · memory_bytes(W, policy)
           + α₂ · latency_ms(model, policy, hardware)
           + α₃ · accuracy_delta(model, policy)

subject to:  drift(model, policy)        <= drift_cap   (hard, ADR-004-style)
             argmax_match(model, policy) >= argmax_floor (hard)
             every_per_tensor_policy_valid(policy)        (hard)
             reproducible(policy, hardware_class)         (hard)
```

### 2.2 The three obvious tradeoffs

| Pair | Direction | Atenia stance |
|---|---|---|
| Memory ↔ drift | smaller bytes → larger drift | drift is the *hard* cap; memory is what we minimise underneath |
| Latency ↔ drift | aggressive quant kernels (INT4, FP8) faster but less stable | same — drift cap is sacred |
| Search cost ↔ result quality | longer search → better policy | bounded by wall-clock; expose as a knob |

### 2.3 Primary vs secondary metrics

| Metric | Role | Source |
|---|---|---|
| `max_abs_diff` vs F64 | **primary hard cap** (ADR-004) | per-model F64 fixture |
| `argmax_match` per position | **primary hard cap** | per-model F64 fixture |
| `memory_bytes_per_tensor` | **primary minimisation target** | computed from policy |
| `latency_ms_per_token` | **secondary minimisation** (only when hardware probe available) | runtime measurement |
| KL-divergence on logits | **diagnostic only** (not gating) | F64 fixture |
| Per-layer per-channel drift | **debug only** | engine-side hook |

### 2.4 Hidden assumption to make explicit

The F64 fixture is the *gold standard for drift but not for accuracy*.
A policy may pass the drift cap and the argmax match yet still degrade
downstream generation quality (e.g. long-form coherence, math, code).
**AQS does not solve that.** A second-tier "behavioural certification"
(perplexity / lm-eval-harness style) is an explicit non-goal of v1.

---

## 3. Phase 3 — Search space design

### 3.1 The dimensions

```text
Technique       T ∈ {plain_int8, awq, gptq, hybrid, bf16_fallback,
                     fp8_future, int4_future}

Per-technique parameters:
  plain_int8:    group_size ∈ {32, 64, 128, 256}
  awq:           group_size, α ∈ [0.0, 1.0],
                 calibration_source ∈ {weight_norm, synthetic,
                                       real_text_8, real_text_128}
  gptq:          group_size, calibration_source, blocksize, percdamp
  hybrid:        all of awq + outlier_k ∈ {0, 32, 64, 128, 256}
  bf16_fallback: none
  fp8_future:    fp8_format ∈ {E4M3, E5M2}

Per-tensor selection: T_tensor_i for each i ∈ tensors_to_quantize
```

### 3.2 Combinatorial explosion analysis

Naive product space, TinyLlama (154 candidate tensors):

```
|T| = 6 techniques × ~10 param combinations each ≈ 60 per-tensor options
total search space = 60^154 ≈ 10^274 configurations
```

**This is the trap.** Brute force is impossible; greedy per-tensor is
~9000 evaluations per model per technique (still hours on CPU);
per-layer is ~22 evaluations (manageable).

### 3.3 Recommended dimension fixing

| Dimension | Fix | Vary | Rationale |
|---|---|---|---|
| `group_size` | **fix at 128** (M9.4 winner across 4 models) | — | varying gave ≤5% in β.4, not worth blowing up the space |
| `α` | — | **per-tensor sweep** in `{0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5}` | β-pivot.3 showed U-curve in this range |
| `outlier_k` | — | **per-tensor sweep** in `{0, 32, 64, 128}` | β.5 showed marginal sensitivity |
| `technique T` | — | **per-tensor** | the actual point of the search |
| `calibration_source` | **fix globally per search** (e.g. `real_text_128`) | — | per-tensor calibration is overfitting territory |
| `tensor_selection` | **fix at "`*_proj.weight`"** | — | proven across β / β-pivot |

**Effective space per tensor:** `T ∈ {plain_int8, awq, gptq, hybrid,
bf16_fallback}` × `α × outlier_k ≈ 7 × 4 = 28` options × 5 techniques
≈ **140 per-tensor evaluations**. For 154 tensors → ~21 600 individual
matmul drift evaluations. Tractable if the drift evaluator is
per-layer not per-forward (see §3.4).

### 3.4 The killer simplification: per-layer drift attribution

A full F64 forward is ~30 s on CPU for TinyLlama. Doing 21 600 of
those is ~7 days. **Unacceptable.**

**The right abstraction:** measure per-tensor reconstruction drift in
isolation (matrix-level, milliseconds each) and accept the linear
projection as the search signal. Reserve full forwards for final
policy validation. This is the β.4 approach repurposed as the *search*
signal, with full-forward β.5 as the *certification* signal.

```
Search loop (cheap):
  for each tensor:
    for each (T, α, k) option:
       measure per-tensor reconstruction max_abs_diff vs F32 truth
       record (option, drift, bytes)
    pick option minimising bytes s.t. drift <= per-tensor budget

Certification loop (expensive, once per candidate policy):
  full F64 forward, compare logits, decide PASS/FAIL
```

The per-tensor budget is derived backwards from the global drift cap
divided by the cascade amplification factor measured on the model
(β.5 showed this factor is ~60× for TinyLlama, ~25× for SmolLM2).
This is the load-bearing heuristic of AQS and deserves its own
sub-milestone (AQS-2).

---

## 4. Phase 4 — Policy granularity

### 4.1 Comparison

| Granularity | Manifest size | Search cost | Maintenance | Expected quality |
|---|---|---|---|---|
| Global (one mode for whole model) | tiny | trivial | trivial | same as M9 / M10.3.1.0 today |
| **Per-layer** (mode per transformer block) | small | manageable | manageable | **>90 % of per-tensor benefit** |
| Per-tensor (mode per parameter) | medium | high | painful | marginal vs per-layer |
| Per-channel (mode per row/column inside a tensor) | huge | unbounded | unmaintainable | only marginal in the literature |

### 4.2 Recommendation

**Per-layer for v1 of AQS.** Per-tensor as an opt-in mode for the
~5–10 % of weights where per-layer fails the drift cap (`lm_head`,
first/last layer, attention output projection).

Justifications:
- Per-layer manifest fits in a single page of YAML; humans can read and
  audit it.
- The β data already showed per-tensor improvement is marginal at best
  in our regime (the plateau is global, not concentrated in specific
  tensors).
- Per-tensor opens the door to overfitting the calibration set, which
  is the most plausible failure mode for AQS (see Phase 10).
- Per-channel is GPTQ-territory only; it is internal to the technique,
  not a knob the manifest exposes.

---

## 5. Phase 5 — Certification model

### 5.1 Manifest schema v3.0.0 (proposed)

```yaml
schema_version: "3.0.0"

model:
  identity:
    family: llama
    config_sha256: "..."           # SHA of the config.json bytes
    weights_sha256: "..."          # SHA of the safetensors bytes
  reference:
    f64_fixture_sha256: "..."      # SHA of expected_logits_f64.json
    token_positions: 4

policy:
  default_mode: bf16_certified
  rules:
    - match: "model.layers.*.mlp.*_proj.weight"
      mode: awq
      params: { group_size: 128, alpha: 0.25, calibration: real_text_128 }
    - match: "model.layers.{0,1,15}.self_attn.q_proj.weight"
      mode: bf16_certified            # high-sensitivity guards
    - match: "model.layers.*.self_attn.*_proj.weight"
      mode: gptq
      params: { group_size: 128, calibration: real_text_128, percdamp: 0.01 }
    - match: "lm_head.weight"
      mode: bf16_certified

metrics:
  measured_on:
    aqs_version: "0.1.0"
    timestamp: "2026-..."
  drift:
    max_abs_diff_vs_f64: 0.41
    argmax_match: [true, true, true, true]
    adr_004_pass: true
  memory:
    bytes_total: 712_000_000
    bytes_saved_vs_bf16: 408_000_000
    saving_fraction: 0.36
  reproducibility:
    rng_seeds_used: []                 # AQS is deterministic; documented
    calibration_corpus_sha256: "..."   # the exact prompts
    search_log_sha256: "..."           # for re-execution

hardware:
  validated_on:
    - { os: linux, arch: x86_64, cpu_features: [avx2], cuda: null }
    - { os: linux, arch: x86_64, cpu_features: [avx2], cuda: { sm: "89", driver: ">=550" } }
  expected_drift_envelope:
    same_class_hardware: { tol: 1e-6 }
    cross_class_hardware: { tol: 1e-3, note: "BF16 TC rounding differences" }
```

### 5.2 Properties this schema gives you

| Property | Mechanism |
|---|---|
| **Reproducible** | every input hashed (weights, config, F64 fixture, calibration corpus, search log) |
| **Portable across hardware** | `validated_on` is an explicit list, not a wildcard |
| **Deterministic** | RNG seeds documented, calibration corpus pinned, AQS itself runs deterministically |
| **Backward compatible** | v3.0.0 is a superset of v2.0.0; legacy manifests still parse and resolve to the same modes |
| **Human-auditable** | per-layer rules are a few dozen lines; the rule glob set is small |
| **Re-verifiable on receive** | runtime can re-run the F64 fixture on first load and abort if it does not match the manifest's drift number |

### 5.3 Anti-patterns this schema rules out

- ❌ Manifests that bind to specific tensor names (rigid against
  refactors). Manifests use globs.
- ❌ Manifests without a hardware envelope. Cross-hardware drift is
  real (TF32 vs F32 vs BF16-TC rounding).
- ❌ Manifests without a calibration corpus hash. "We used AWQ with
  α=0.3" is unreproducible without the prompts.
- ❌ Black-box quality claims ("verified on MMLU"). Out of scope for
  v1; mentioned only as future "behavioural certification".

---

## 6. Phase 6 — Search algorithm

### 6.1 Survey

| Algorithm | Cost | Quality | Determinism | Fit |
|---|---|---|---|---|
| Brute force | infeasible (10^274) | optimal | deterministic | rejected |
| Greedy per-tensor (fix others, sweep one) | O(tensors × options) | local optimum | deterministic | **good fit** |
| Hill climbing with restarts | O(K · greedy) | better local | deterministic with seed | good fit |
| Beam search (top-B at each layer) | O(B · greedy) | better but expensive | deterministic | maybe v2 |
| Simulated annealing | O(steps) | needs tuning | RNG-dependent | rejected (non-deterministic without seed pinning) |
| Evolutionary | O(generations · population) | unpredictable | RNG-dependent | rejected |
| RL / policy search | huge | unpredictable | non-deterministic | hard reject |

### 6.2 Recommendation: greedy per-layer with one cycle of revisit

**Algorithm AQS-Greedy-v1** (single-pass per-layer):

```
1. Compute per-layer drift budget from cascade-amplification estimate.
2. For each layer L in topological order:
     a. For each technique T in {bf16, plain_int8, awq, gptq, hybrid}:
          measure per-tensor reconstruction drift for L's _proj.weight tensors
          aggregate to per-layer drift contribution
          compute bytes saved vs bf16
     b. Pick the T that minimises bytes subject to per-layer drift budget.
3. Run one full F64 forward against the policy.
4. If global drift > drift_cap:
     a. Find the layer(s) contributing most to drift overrun.
     b. Force those layers to the next-safer mode (e.g. gptq → bf16).
     c. Repeat step 3.
5. Emit manifest.
```

**Cost estimate (TinyLlama):**
- Step 2: 22 layers × 5 techniques × 7 tensors × (~50 ms reconstruction
  drift eval) ≈ **40 s**.
- Step 3: 1 full F64 forward ≈ **30 s**.
- Step 4: 1–3 retries at the same cost.
- **Total: ~3 minutes per model.** Acceptable.

For SmolLM2 (24 layers, harder cascade) the budget step may iterate
more; budget ~10 minutes per model.

### 6.3 Reproducibility contract

- **No RNG** in the search itself. All ties broken by tensor name
  alphabetical order.
- Calibration prompt set hashed and embedded in the manifest.
- Hardware fingerprint embedded; replay aborts if the recorded
  drift cannot be reproduced within `1e-6` on the same hardware class.

---

## 7. Phase 7 — Data / storage strategy

### 7.1 What to persist

| Artifact | Per | Volume | Format |
|---|---|---|---|
| Manifest | model | KB | YAML (the certified deliverable) |
| Search log | search run | MB | JSONL (one line per evaluated config) |
| Calibration corpus | search run | KB | embedded text or referenced by hash |
| Per-tensor activation stats | calibration run | tens of MB | binary (one f32 per K-channel per tensor) |
| F64 fixture | model | low MB | existing JSON format (no change) |
| Drift history | per (model, hardware) pair | small | JSONL append-only |
| Benchmark history | per (manifest, hardware) | small | JSONL append-only |

### 7.2 Recommended persistence layout

```
~/.atenia/aqs/
├── models/
│   └── tinyllama-1.1b/
│       ├── manifest.v3.yaml                   # the deliverable
│       ├── f64_fixture.json                   # reference logits
│       ├── search/
│       │   ├── 2026-xx-xx-greedy-v1.log.jsonl # one search run
│       │   └── ...
│       └── activations/
│           └── real_text_128.bin              # calibration cache
└── hardware_profiles/
    └── linux-x86_64-avx2-rtx4070/
        └── benchmarks.jsonl
```

### 7.3 Storage technology

| Tech | Verdict | Rationale |
|---|---|---|
| **YAML for manifest** | ✅ | human-auditable, the certified surface |
| **JSONL for logs / history** | ✅ | append-only, line-grep-able, no DB |
| **Binary for activation caches** | ✅ | volume justifies it, internal format |
| SQLite | ❌ for v1 | no query workload that benefits |
| Parquet | ❌ for v1 | no columnar analytics planned |
| Cloud / DB | ❌ | violates the "no infra cloud" rule |

**File-system-only is the right answer for AQS v1.** No databases, no
servers, no daemons. Same posture as the rest of Atenia.

---

## 8. Phase 8 — Hardware-aware search

### 8.1 Does the optimal policy depend on the hardware?

**Yes, but more weakly than one might fear.** The β empirical data
(reconstruction-level) is hardware-independent — the drift is in the
math, not the kernel. What *does* depend on hardware:

| Aspect | Hardware sensitivity |
|---|---|
| Per-element drift of a specific (technique, params) combo | **none** (deterministic math) |
| Latency / throughput | **high** (CPU SIMD vs CUDA TC vs AMD CDNA) |
| Memory budget feasibility | **high** (VRAM size dictates which layers must offload) |
| BF16 TC rounding vs F32 CPU | **small** (~1e-3 cross-class) |
| Available kernels (INT4? FP8?) | **binary** (kernel exists or it does not) |

### 8.2 Modelling without explosion

**Two-tier approach:**

1. **Hardware class** (a small enum): `{cpu_avx2, cpu_avx512,
   cuda_sm_70_80, cuda_sm_89_90, rocm_cdna3, metal_m3}`. The search
   runs once per hardware class and the manifest records all classes
   it validated against.
2. **Per-class kernel availability** is the only thing that affects
   technique availability (e.g. no FP8 on AVX2). The optimisation
   landscape (drift, memory) is the same across classes; only the
   *technique menu* changes.

**Practical implication:** AQS searches on the developer's CPU, the
resulting manifest is portable to any hardware class whose technique
menu is a superset of the search's. CUDA additions just unlock more
techniques; they do not require re-search.

### 8.3 What is explicitly out of scope

- Heterogeneous fleets (mixed-hardware inference). Single-host per
  manifest.
- Cost-model driven search ("optimise for AWS g5"). The objective is
  bytes; latency is secondary.
- JIT-style hardware re-search at runtime. Cold offline search only.

---

## 9. Phase 9 — Product strategy

### 9.1 What AQS sells

> **"Every Atenia checkpoint ships with a manifest the runtime can
> verify against F64 ground truth, and the manifest was discovered
> automatically."**

That single sentence is the differentiator. Decomposed:

| Claim | Real today? | After AQS-1 | After AQS-3 |
|---|---|---|---|
| Per-checkpoint drift envelope vs F64 | ✅ (numcert v1/v2) | ✅ | ✅ |
| Per-tensor / per-layer policy | ✅ (numcert v2 PerTensorPolicy) | ✅ | ✅ |
| Auto-discovered, not hand-picked | ❌ | partial | ✅ |
| Runtime re-verifiable | partial (drift is loaded but not re-checked on load) | ✅ | ✅ |
| Multi-technique (GPTQ + AWQ + ...) | ❌ | GPTQ added | full |
| Hardware-portable manifests | partial | ✅ | ✅ |

### 9.2 Three product flavours, in honest categories

| Flavour | Realism | Notes |
|---|---|---|
| **Auto-certified quantization** | ✅ realistic | the v1 deliverable; nothing magical |
| **Numerically auditable inference** | ✅ realistic | already true today; AQS makes it more legible |
| **Hardware-adaptive manifests** | ⚠️ realistic but narrow | covered by §8.2 two-tier approach |
| **"Discover the best quant for your model"** as a user-facing CLI | ⚠️ realistic but expensive | `atenia search <model_dir>` is a 3-30 min wait |
| **"Self-searching runtime" (online adaptation)** | ❌ marketing | runtime never re-searches; that is a research direction |
| **Behavioural certification** (MMLU / lm-eval) | ❌ for v1 | requires lm-eval-harness infra; explicit non-goal |
| **Cross-model policy transfer** | ❌ for v1 | tempting but unproven; explicit non-goal |

### 9.3 What this means for the README / website

AQS does **not** sell "we made AWQ better". It sells "we made
quantization auditable, reproducible and automatic, and the manifest
ships with the model". That framing survives the GPTQ/AWQ next-paper
churn because the contribution is the *certification layer*, not the
underlying technique.

---

## 10. Phase 10 — Risks (brutal honesty mode)

### 10.1 Numerical / methodological risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Calibration prompt overfitting** — the manifest passes drift on the search corpus but degrades on real workloads | **HIGH** | (a) hash and ship the calibration corpus; (b) keep a held-out validation corpus; (c) document the calibration sentence count in the manifest so operators can re-search with their own data |
| **Per-tensor / per-layer drift is not additive** — the cascade amplification factor varies by layer | **HIGH** | (a) the search loop already re-checks via full forward; (b) the per-layer budget is heuristic, not contractual; (c) document this clearly |
| **F64 fixture is per-model + 4 token positions** — a manifest can pass while drifting on positions / sequences not in the fixture | **MEDIUM** | (a) extend the fixture to ~32 positions for AQS-certified models; (b) re-verification on load with a small held-out sample |
| **Argmax-stability does not equal generation quality** | **MEDIUM** | (a) explicit non-goal in v1; (b) layer 2 "behavioural certification" as future work |
| **Cross-hardware drift exceeds the recorded envelope** | **LOW** | already addressed by §5.1 `validated_on` list |

### 10.2 Engineering risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Combinatorial explosion** — the search space drifts towards 10^N variants | **HIGH** | (a) fix `group_size` and `calibration_source` globally; (b) per-layer not per-tensor; (c) cap search wall-clock |
| **Maintenance nightmare** — every new technique requires touching the search, the manifest, the runtime | **HIGH** | (a) `QuantizationPolicy` trait so adding a technique is one impl + one enum variant; (b) manifest schema uses tagged-string mode, not closed enum |
| **CI matrix explosion** — F64 fixtures for N models × M hardware classes | **MEDIUM** | (a) per-model F64 fixture is one-time generation; (b) hardware classes use small reference matrices for CI; full F64 forwards remain `#[ignore]` |
| **Search cost too high for casual operators** | **MEDIUM** | (a) `atenia search` ships a cached manifest from Atenia's own searches; (b) re-search is an opt-in not the default |
| **Irreproducible search** — different machines produce different manifests | **MEDIUM** | (a) deterministic search algorithm; (b) hardware-class fingerprint in the manifest |

### 10.3 Product risks

| Risk | Severity | Mitigation |
|---|---|---|
| **AQS becomes the project's tar pit** — every β technique demands AQS support | **HIGH** | (a) AQS-0 scoped tight; (b) v1 supports exactly 4 techniques; (c) explicit non-goals doc |
| **The plateau extends to GPTQ** — even GPTQ does not satisfy ADR-004 strict on some models | **MEDIUM** | (a) document the plateau as an empirical envelope; (b) accept it; (c) ADR-004-strict remains a F32/BF16 contract for those models |
| **Marketing temptation** — "AQS searches your manifest in real time as you generate" | **HIGH** | (a) explicit non-goal: no online search; (b) keep README brutalist |

### 10.4 The most likely failure mode

**Not** that AQS produces bad manifests. **That AQS produces excellent
manifests for TinyLlama only**, and the search-loop heuristics
(per-layer budget, cascade amplification estimate) collapse on larger
models (SmolLM2 1.7B, Qwen 2.5 1.5B, Llama 3.2 1B).

The β.4 / β.5 / β-pivot data already shows SmolLM2 is the worst-case
sentinel. AQS v1 must validate against all four M8.5 fixture models,
not just TinyLlama.

---

## 11. Phase 11 — Proposed roadmap

### 11.1 Eight milestones, in dependency order

```
AQS-0  Architecture                  — this audit + a written API surface design (0 LoC)
AQS-1  QuantizationPolicy trait      — unify the four perturb_param_* helpers into one extensible substrate
AQS-2  Per-tensor drift evaluator    — fast (no full forward) reconstruction-drift harness reusable as search signal
AQS-3  GPTQ integration              — the missing technique; first new entry in the policy registry
AQS-4  Greedy per-layer search       — the AQS-Greedy-v1 algorithm; outputs a candidate policy
AQS-5  Full forward certification    — wires the existing β.5 F64 forward harness into the search-and-retry loop
AQS-6  Manifest schema v3.0.0        — additive on v2; emits the AQS-discovered manifest
AQS-7  CLI surface — `atenia search` — opt-in command, 3–30 min wall-clock, writes manifest + log
AQS-8  Hardware envelope             — `validated_on` list, cross-class drift assertion on load
```

### 11.2 What each milestone takes

| Milestone | Cost | Risk | Dependency |
|---|---|---|---|
| AQS-0 | days | — | — |
| AQS-1 | ~1 week | low (additive refactor of β code) | AQS-0 |
| AQS-2 | ~1 week | medium (per-layer budget heuristic) | AQS-1 |
| AQS-3 | ~2 weeks | high (GPTQ is genuinely new code) | AQS-1, AQS-2 |
| AQS-4 | ~1 week | medium (search-loop bookkeeping) | AQS-3 |
| AQS-5 | ~0.5 week | low (reuses β.5 harness verbatim) | AQS-4 |
| AQS-6 | ~0.5 week | low (additive on numcert v2) | AQS-5 |
| AQS-7 | ~1 week | low (CLI-style work) | AQS-6 |
| AQS-8 | ~1 week | medium (hardware fingerprinting) | AQS-7 |

**Critical path:** AQS-0 → AQS-1 → AQS-3 → AQS-4 → AQS-5 → AQS-6.
About **6 weeks** of focused work for a shippable v1 (single
technique addition + single-model search + manifest + CLI). AQS-7/8
extend the surface.

### 11.3 What v1 deliberately does NOT include

- AWQ-with-real-text-cal as a search dimension (β-pivot.3 underperformed
  vs synthetic at the same α; keep it opt-in but not searched).
- FP8, INT4, W2 paths (no kernel today; placeholders in the policy
  registry only).
- Cross-model policy transfer.
- Online / runtime search.
- Behavioural (MMLU-class) certification.
- Heterogeneous fleet support.
- A search-result database.
- A GUI.

---

## 12. Phase 12 — Final verdict

### 12.1 Is AQS viable?

**Yes.** ~70 % of the substrate already exists, the search problem is
tractable with greedy per-layer at ~3 minutes per model, and the
manifest schema is an additive bump on the v2 production format.

### 12.2 Is it genuinely differentiating?

**Yes, narrowly.** The differentiator is **automated certification +
hardware-portable manifests + per-layer policy**, not the search
algorithm itself. Every other inference runtime ships quantized
checkpoints without measured drift envelopes. Atenia's F64 contract
makes the envelope possible; AQS productises it.

### 12.3 Where is the smoke?

| Idea | Verdict |
|---|---|
| "Self-searching runtime" | ❌ marketing |
| "Discover the optimal policy for any model" | ⚠️ only true for models in the F64 fixture; otherwise needs fixture generation first |
| "AQS replaces GPTQ / AWQ" | ❌ AQS *uses* them; it does not replace |
| "Hardware-adaptive" | ⚠️ true within a class enum; not true continuously |

### 12.4 Where is the real innovation?

| Innovation | Verdict |
|---|---|
| F64 reference as the search target (not just validation) | ✅ unique to Atenia |
| Per-layer manifest in human-auditable YAML | ✅ unique product format |
| Runtime re-verification on load against the manifest's drift number | ✅ unique to Atenia |
| Calibration corpus hashed and shipped with the manifest | ✅ reproducibility unique to Atenia |
| Greedy per-layer search bounded by per-layer cascade budget | ⚠️ heuristic; load-bearing; deserves its own validation milestone |

### 12.5 What should be built first?

**AQS-0 (this audit) → AQS-1 (`QuantizationPolicy` trait) → AQS-3
(GPTQ)** is the unavoidable critical path. AQS-2 (drift evaluator)
can be deferred or built in parallel with AQS-3.

### 12.6 What should NOT be built yet?

- Anything in §11.3.
- Per-tensor granularity beyond the per-layer opt-in escape.
- The hardware envelope, until at least one CUDA + one CPU search has
  been completed and the cross-class drift has been measured (AQS-5
  consequence).

### 12.7 Where is the biggest technical risk?

**The cascade amplification heuristic** (per-layer drift budget derived
from the global drift cap). If this heuristic is wrong on any of the
four M8.5 fixture models, AQS-4 retries explode and the search wall-
clock balloons. AQS-2 must include a falsification experiment for the
heuristic on all four models before AQS-4 lands.

### 12.8 Where is the biggest opportunity?

**Making "this checkpoint is numerically certified for ADR-004 on
your CPU class" a single command output.** No other runtime can say
this today. AQS is what turns Atenia's F64 contract from an internal
correctness story into a shipping product feature.

---

## 13. Appendix A — Explicit non-goals (v1)

For the avoidance of doubt, AQS v1 does **not** attempt:

- Quantization-aware training, fine-tuning or weight retraining.
- Dataset downloads or any internet I/O during search.
- Activation quantization (W8A8, FP8 activation) — weight-only.
- Sub-4-bit weights.
- Distillation.
- Per-sequence-length policy.
- Runtime adaptive policy switching.
- Mixed-vendor manifests (one Cuda + one ROCm in one manifest).
- "AutoAI" framing.

---

## 14. Appendix B — Suggested first implementation milestone (AQS-1)

If you authorise AQS-1 after this audit, here is the surface that
should land:

```rust
// src/quant/policy.rs (new module, no src/ surgery outside this folder)

pub trait QuantizationPolicy: Send + Sync {
    /// Stable identifier for the manifest's `mode` field.
    fn id(&self) -> &'static str;

    /// Validate that the (shape, group_size, params) tuple is
    /// compatible with this policy.
    fn validate(&self, shape: &[usize]) -> Result<(), PolicyError>;

    /// Apply the in-place F32 perturbation that simulates the
    /// quantisation round-trip. CPU-only.
    fn apply_inplace(
        &self,
        weights: &mut [f32],
        shape: &[usize],
        cal: &CalibrationContext<'_>,
    ) -> Result<(), PolicyError>;

    /// Memory cost in bytes if this policy were realised in the
    /// final storage (used by the search objective).
    fn memory_bytes(&self, shape: &[usize]) -> u64;
}

pub struct CalibrationContext<'a> {
    pub activation_absmax: Option<&'a [f32]>,  // length K = shape[0]
    pub corpus_hash: [u8; 32],
}

pub struct Bf16Fallback;
pub struct PlainInt8 { pub group_size: usize }
pub struct AwqPolicy { pub group_size: usize, pub alpha: f32 }
pub struct HybridPolicy { /* ... */ }
pub struct GptqPolicy { /* AQS-3 */ }
```

The four existing `WeightStore::perturb_param_with_*` helpers become
the bodies of `PlainInt8::apply_inplace`, `AwqPolicy::apply_inplace`,
`HybridPolicy::apply_inplace`. `Bf16Fallback` is a no-op. `GptqPolicy`
lands in AQS-3.

**Files touched in AQS-1:** new `src/quant/policy.rs` + a small
delegation refactor in `src/amg/weight_store.rs`. No engine, no CLI,
no manifest, no CUDA, no loader.

---

## 15. Appendix C — Glossary

| Term | Meaning |
|---|---|
| ADR-004 | Atenia decision record: drift cap of `max_abs_diff < 0.5` vs F64 reference |
| F64 fixture | per-model JSON of expected_logits_f64 from a PyTorch double-precision forward |
| numcert | the existing per-checkpoint quantization manifest (v1.0.0, v2.0.0) |
| `SharedParam` | runtime parameter slot; storage variant decides residency (F32, BF16, Cuda, Disk, CpuInt8Outlier) |
| Cascade amplification | empirically-measured factor by which per-tensor reconstruction drift grows through the forward stack |
| Calibration corpus | the set of token sequences whose activations drive AWQ scale derivation |
| Policy | the (technique, params) tuple chosen for one tensor |
| Manifest | the YAML file the runtime loads alongside the model to know which policy applies to which tensor |

---

*End of audit. No code was modified, no commits made, no branches
created. This document is the deliverable.*

---

## Post-implementation results (epilogue)

> **Added after AQS-1 → AQS-10 were implemented.** The audit above is the
> *pre-implementation* plan and is preserved verbatim as a historical
> record. This epilogue records what actually happened, because parts of
> the plan (notably the optimism about GPTQ) were tested and came back
> negative. For the consolidated overview see
> [AQS_OVERVIEW.md](./AQS_OVERVIEW.md); per-milestone detail is in
> `docs/HANDOFF_AQS_1.md` … `docs/HANDOFF_AQS_10.md`.

**What shipped.** The full pipeline the audit proposed was built, isolated
and opt-in: `QuantizationPolicy` (AQS-1) → drift evaluator (AQS-2) →
end-to-end harness (AQS-4) → certification report + `3.0.0-draft` manifest
(AQS-6) → search engine (AQS-7) → callback runner (AQS-8) → real-TinyLlama
wiring (AQS-9) → `atenia search` CLI (AQS-10). GPTQ was implemented both as
a diagonal surrogate (AQS-3) and as **real blockwise GPTQ** with a full
`K×K` Hessian, Tikhonov damping, and Cholesky-based inverse-Hessian error
compensation (AQS-5).

**GPTQ outcome (the audit's key assumption, now falsified).** The audit
treated GPTQ as the most promising path to ADR-004 strict. Measured
end-to-end on TinyLlama against the F64 fixture:

- GPTQ **surrogate**: 12.5 drift — far worse than plain INT8.
- GPTQ **real**: 1.405 drift — still worse than plain INT8 (1.261) and
  AWQ (0.889), with broken argmax. Most likely cause: calibration
  starvation (the Hessian was severely rank-deficient with the available
  calibration set). Real GPTQ also cost ~7.8 h on CPU for one model.

**AWQ outcome.** AWQ (α=0.25) is the best *useful-lossy* policy — 0.889
drift, argmax-stable, ~1.94× compression — but **above** the ADR-004
strict gate (0.5), so it is **not certified**.

**Accepted plateau.** Five weight-only mechanisms (plain INT8, β outlier,
AWQ, hybrid, GPTQ) all fail ADR-004 strict on TinyLlama. **BF16 is the only
certified policy.** The weight-only plateau is **accepted**: AQS's
delivered value is the certification / search / reporting layer, not a new
quantization technique. The runtime's production numeric certification is
unchanged and remains governed by ADR-004 / ADR-005.

**Risk register accuracy.** The audit's risk discussion flagged "the
plateau may extend to GPTQ" as a real possibility; that risk materialised.
The audit's scope safeguards (experimental, opt-in, no productive
integration) held throughout — no runtime, loader, generation, CUDA,
tier-planner, or productive-manifest code was touched by any AQS milestone.
