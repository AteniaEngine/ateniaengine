# HANDOFF — AQS-4: end-to-end policy evaluation harness landed

Milestone: **AQS-4** (first real TinyLlama forward + F64 comparison over
`QuantizationPolicy` candidates; no automatic search).
Predecessors:
- `docs/AQS_ARCHITECTURE_AUDIT.md` (`6e1ead7`)
- `docs/HANDOFF_AQS_1.md` — `QuantizationPolicy` (`c5e39bb`)
- `docs/HANDOFF_AQS_2.md` — local drift evaluator (`602901b`)
- `docs/HANDOFF_AQS_3.md` — experimental GPTQ (`e31edd2`)

## Objective

Close the gap AQS-3 identified: the *local* per-tensor drift signal
cannot judge GPTQ, because GPTQ trades per-element drift for column-level
error cancellation that only appears through a real forward. AQS-4 is the
first orchestration that perturbs **real** TinyLlama weights with a
policy, runs a **real** CPU forward, and compares **logits** against the
certified F64 fixture (ADR-004 gate, `max_abs_diff < 0.5`).

It is a **deterministic, reproducible orchestrator** — NOT the automatic
search (that is AQS-5+).

## Architecture

```
candidate policy
  → clone the loaded WeightStore           (original never mutated)
  → WeightStore::perturb_all_proj_with_policy(&dyn QuantizationPolicy, act)
  → build_llama_with_store + forward       (CPU, kernel_dtype = F32)
  → logit_drift_metrics(logits, F64 fixture)
  → EndToEndEvalResult row
```

* **Reusable, model-agnostic** pieces in `src/quant/end_to_end.rs`:
  `PolicyEvalCandidate`, `EndToEndEvalResult`, `logit_drift_metrics`,
  `render_result_table`.
* **Weight perturbation** through the new experimental
  `WeightStore::perturb_param_with_policy` /
  `perturb_all_proj_with_policy` (the AQS-1-deferred "minimal delegation",
  landed now because AQS-4 needs it). Single dependency edge
  `amg → quant::policy`; CPU-only; Cuda/Disk variants panic.
* **Model-specific harness** in `tests/aqs4_end_to_end_test.rs`, reusing
  the β / β-pivot TinyLlama load + calibration + forward pattern. Env-var
  / fixture / safetensors I/O stays out of `src/`.

### How weights are cloned / applied

The loaded store is never mutated. Each candidate runs on a fresh
`WeightStore { params: store.params.clone(), names: store.names.clone() }`.
`SharedParam::F32` holds an `Arc<Vec<f32>>`; perturbation **replaces** the
Arc (copy-on-write), so cloned stores are fully isolated. Verified by the
CI test `orchestrator_preserves_original_weights`.

## Metrics produced (end-to-end, real logits)

`max_abs_diff` (L∞, the ADR-004 metric), `mean_abs_diff`, `rmse`,
`argmax_match` (true logits argmax across all 4 positions),
`memory_bytes` (Σ `policy.memory_bytes(shape)` over perturbed
`_proj.weight`).

## TinyLlama results (real forward, F64 fixture)

```
group_size=128  alpha=0.25  outlier_k=64  gptq_damp=0.01
calibration: 8 real-text sequences, 154 _proj.weight tensors, 5.5s

candidate          max_diff    mean_diff       rmse   argmax   memory_bytes   ADR-004
-------------------------------------------------------------------------------------
certified(f32)     0.000063     0.000008   0.000010    true             0      PASS
bf16               0.000063     0.000008   0.000010    true    2260729856      PASS
plain_int8         1.260771     0.145249   0.202987   false    1165688832      FAIL
awq                0.889217     0.074738   0.105758    true    1165688832      FAIL
hybrid             0.831786     0.061254   0.091544   false    1266614272      FAIL
gptq              12.510114     1.171387   1.671700   false    1167265792      FAIL
```

### Harness is validated by reproducing prior independent results

This is the credibility anchor — AQS-4 was written from scratch yet
reproduces three earlier β measurements **exactly**:

| Policy     | AQS-4 max_diff | Prior β result            | Match |
|------------|----------------|---------------------------|-------|
| AWQ        | 0.889, argmax 4/4 | β-pivot.3: 0.89, argmax 4/4 | ✅ |
| Hybrid     | 0.832          | β-pivot.5: 0.83           | ✅ |
| plain INT8 | 1.261          | β.5 reference: ~1.20–1.26 | ✅ |
| BF16       | 0.000063 (= certified) | (identity)        | ✅ |

The forward, the calibration, and the metric are therefore trustworthy.

## Local drift vs end-to-end drift

| Policy | AQS-2 local max_abs_diff (random 64×64) | AQS-4 end-to-end max_abs_diff |
|--------|------------------------------------------|-------------------------------|
| AWQ    | ≈ 0.016                                  | 0.889                         |
| GPTQ   | ≈ 0.048 (worse locally)                  | **12.510** (far worse)        |

## Honest verdict on GPTQ

**The science question — "does GPTQ win end-to-end despite worse local
drift?" — answers a clear NO for this implementation.**

The simplified GPTQ surrogate is the **worst** candidate end-to-end
(12.51), an order of magnitude worse than plain INT8 (1.26) and ~14×
worse than AWQ (0.89). Crucially:

* **AQS-2 did NOT underestimate GPTQ in the helpful direction.** The local
  signal already ranked GPTQ *worse* than AWQ (0.048 > 0.016), and
  end-to-end confirmed it is worse — dramatically so. The local→
  end-to-end relationship here is monotone in the bad direction, not the
  hoped-for inversion.

* **This condemns the *surrogate*, not GPTQ-the-algorithm.** The AQS-3
  implementation is explicitly a *simplified* GPTQ: a diagonal-Hessian
  clip-search scale + sequential error diffusion. Two of its choices are
  now shown to be actively harmful on real LLM weights:
  1. the Hessian-weighted clip search picks sub-absmax scales that *clip*
     structured weight outliers;
  2. per-column error diffusion redistributes that clipped energy in a way
     that, summed across the matmul's K reduction, amplifies rather than
     cancels.
  Real GPTQ's benefit comes from the **full inverse-Hessian per-weight
  compensation** that the AQS-3 surrogate explicitly does *not* implement
  (it is degenerate under a diagonal Hessian). So **real GPTQ remains
  untested** — it would need the inverse-Hessian machinery, which is out
  of AQS-4 scope.

**No goalposts were moved and no result was cherry-picked.** The surrogate
failed; that is the honest outcome.

## Which policy is best (today)

**AWQ** (`alpha = 0.25`): lowest end-to-end drift among lossy policies
(0.889), the only lossy policy with **argmax 4/4**, ~2× compression
(1.17 GB vs 2.26 GB BF16). It is still **above** ADR-004 strict (0.5) — no
weight-only policy in the AQS family crosses the gate on TinyLlama. This
is consistent with the entire β / β-pivot plateau (0.78–0.89).

## Tests

* CI (no model): `orchestrator_preserves_original_weights`,
  `bf16_policy_is_endtoend_noop_on_store`,
  `policies_perturb_deterministically_on_store`,
  `awq_on_store_requires_activation_stats`,
  `result_table_renders_all_candidates` + 4 light unit tests in
  `src/quant/end_to_end.rs`.
* Heavy (`#[ignore]`): `aqs4_tinyllama_policy_comparison` — the table
  above; asserts only that the certified baseline PASSes ADR-004 and that
  every drift is finite. **Passed in 150.5s.**

Local validation: `cargo test --lib --release` → **577 passed / 0 failed /
1 ignored**. `cargo test --test aqs4_end_to_end_test` → 5 passed / 1
ignored. Heavy run reported above.

## Files modified

* `src/quant/end_to_end.rs` — new (types + metrics + table + light tests).
* `src/quant/mod.rs` — `pub mod end_to_end;` + re-exports.
* `src/amg/weight_store.rs` — added experimental
  `perturb_param_with_policy` + `perturb_all_proj_with_policy` (edge
  `amg → quant::policy`).
* `tests/aqs4_end_to_end_test.rs` — new (CI mechanics + heavy harness).
* `docs/HANDOFF_AQS_4.md` — this file.

No other files touched. No CUDA, tier-planner, CLI, generation, loader, or
manifest changes.

## Next step (needs authorisation)

Two honest directions, both AQS-5:
1. **Real GPTQ** — full inverse-Hessian per-weight compensation, to test
   GPTQ-the-algorithm rather than the failed surrogate. Heavier, but the
   only way to actually answer whether GPTQ beats the AWQ plateau.
2. **Accept the plateau** — declare AWQ (0.889, argmax 4/4) the best
   weight-only option, stop chasing ADR-004 strict with weight-only
   methods, and pivot the AQS value proposition to *certified search +
   verifiable manifests* over the policies we already have (the original
   audit's actual differentiator).
