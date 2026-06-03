# HANDOFF — NUMERIC-POLICY-3: certification governance

The NUMERIC-POLICY audit ([docs/NUMERIC_POLICY_AUDIT.md]) found the architecture
*functionally* complete but the *governance of correctness* not production-grade:
certification was manual/offline, **not persisted**, and there was **no runtime
guard**. This milestone closes those gaps — turning the fast modes from "opt-in,
operator-verified" into "**certificate-backed with a safe runtime fallback**" —
**without** changing kernels, the `Certified` default, or making `Fast` default.
Predecessor: `8e7b61e`.

## FASE 1 — Audit of the current cert path

`PolicyCertificate` (`numeric_policy.rs`) computes `max_abs_diff` / `mean_abs_diff`
/ `rmse` / `argmax_match_rate` / `tokens_match` and `passes(τ)`. It was used only
in **unit tests** and an **ad-hoc int8 sim**, never persisted, never read at load
time. Policy is selected by `ATENIA_NUMERIC_POLICY` (cached) / in-process
override; tier dtype by `ATENIA_MOE_TIER_QUANT` / `ATENIA_MOE_TIER_BF16`. **Gap:
nothing connected "a non-Certified mode must have a passing certificate".**

## FASE 2 — Certificate artifact (`src/moe/cert.rs`)

`NumericCertificate` (JSON), one per (model, policy, tier dtype):
`certificate_version`, `created_at_unix`, `code_version`, `model_id`,
`manifest_version`, `numeric_policy`, `tier_dtype`, `validation_set_id`,
`tolerance`, per-case `cases[]` (prompt, max_new, **expected** vs **observed**
token ids, `tokens_match`, `argmax_match_rate`, `max_abs_diff`/`mean_abs_diff`/
`rmse`), `pass`, `worst_max_abs_diff`. Stored next to the tier manifest:
`<tier_dir>/moe_tier/<model_id>/numeric_cert_<policy>_<tier>.json`.

## FASE 3 — Validation prompt set

`validation_set()` — 6 deterministic greedy cases, varied prompts + lengths,
**token ids in `0..=9`** (valid for any real vocab and the tiny test fixtures),
**no network / no external model / nothing downloaded**. Versioned by
`VALIDATION_SET_ID` (`"moe-greedy-v1"`); a change bumps it so stale certs are
rejected.

## FASE 4 — Cert runner

`run_certification(rt, model_id, policy, tier_dtype, manifest_version)` runs the
set under `Certified` (reference) vs the candidate **on one loaded (lossless)
model**, toggling the compute policy and the **int8 sim** (so the int8 tier's
numerical effect is modelled without a second cold load / tier rebuild),
comparing tokens + logits, and aggregating into a `NumericCertificate`. The
orchestrator `runtime::certify_model(dir, policy, tier_dtype)` loads losslessly,
runs it, and persists the cert. **CLI:** `atenia moe-certify --model DIR
--policy strict --tier qint8 --experimental-moe` (prints pass/fail + worst drift
+ path; exit 0 pass / 1 fail). The runner clears its in-process overrides on exit.

## FASE 5 — Loader / validator

`load_certificate(path)` + `NumericCertificate::is_valid_for(model_id, policy,
tier_dtype, manifest_version, validation_set_id)` → valid **iff** every field
matches **and** `pass`. Any mismatch (model / policy / tier / manifest version /
prompt-set id / cert version) or a failed cert → invalid.

## FASE 6 — Runtime guard (`runtime.rs::apply_certification_guard`)

Enforced **only** under `ATENIA_NUMERIC_REQUIRE_CERT=1` (default off →
behaviour unchanged). At load, given `model_id`:

- **Lossy tier (qint8)** — cannot be downgraded at run time (it is baked on
  disk): a valid passing cert for the requested policy is **required**, else the
  load is **refused** with an actionable error.
- **Lossless tier (bf16/f32)** — a non-`Certified` compute policy without a valid
  cert **falls back to `Certified`** (runtime-switchable, free).
- `Certified` is always allowed. The default (flag off) keeps the existing
  opt-in/experimental behaviour intact.

## FASE 7 — Effective numeric-mode descriptor

`cert::numeric_mode_descriptor(policy, tier, cert_status)` →
`policy=… tier=… cert=…`, logged under `ATENIA_MOE_CACHE_STATS=1` at load
(`[ATENIA] numeric mode: …`) so a run's effective precision is never silent.

## FASE 8 — Tests

- `moe::cert` units (6): serialization round-trip; `is_valid_for` accept/reject on
  every field; failed cert never valid; guard fallback/allow logic; save+load
  round-trip; descriptor format.
- `tests/moe_cert_governance_test` (new): `certify_model` writes a 6-case
  certificate; under `REQUIRE_CERT` an **int8 tier with a valid cert loads**, the
  **same tier without a cert is refused**, and **Certified + lossless is always
  allowed**.
- Full MoE regression + lib suite green; **default path unchanged** (the only
  intermittent failure is the unrelated environmental GPU-VRAM probe).

## Architecture notes / limits

- **Compute policy vs tier dtype.** The guard treats them differently *by
  necessity*: the compute policy is runtime-switchable (free fallback); the tier
  dtype is a build-time on-disk choice (refuse, don't silently downgrade).
- **Single-process env.** `numeric_policy()` caches the env once (correct for a
  CLI process); the cert runner / tests drive it via the in-process override.
- **Sim-based int8 certificate.** The int8 tier's numerical effect is certified
  via the per-row sim (shown identical to the real int8 tier in NUMERIC-POLICY-2).
  A future option is to certify against a real int8 tier directly (needs a tier
  dir keyed by dtype to avoid bf16/int8 collision).
- The certificate's metric is greedy token-id + per-step logit drift on 6 short
  cases; broadening to longer text / perplexity is a future enhancement.

## Deliverable answers

1. **Implemented:** persisted `NumericCertificate` + validation set + runner
   (`certify_model` / `atenia moe-certify`) + loader/validator + runtime guard
   (`ATENIA_NUMERIC_REQUIRE_CERT`) + effective-mode descriptor.
2. **Certificate:** JSON, all required fields incl. expected/observed tokens,
   argmax/tokens match, drift metrics, manifest + code + cert + prompt-set
   versions, pass/fail.
3. **Where:** `<tier_dir>/moe_tier/<model_id>/numeric_cert_<policy>_<tier>.json`.
4. **How generated:** `atenia moe-certify` (or `certify_model`) runs the
   validation set Certified-vs-candidate on one lossless load (policy toggle +
   int8 sim) and persists the result.
5. **How validated:** `is_valid_for` requires every field to match + `pass`.
6. **Validation set:** 6 deterministic greedy cases, ids `0..=9`, offline.
7. **Fallback:** lossy tier w/o valid cert → refuse; non-Certified policy on a
   lossless tier w/o valid cert → `Certified`; flag off → unchanged opt-in.
8. **Default Certified:** unchanged — always allowed, still the default + the
   universal fallback; bit-exact path untouched.
9. **Effective mode reporting:** `[ATENIA] numeric mode: policy=… tier=… cert=…`.
10. **Tests:** cert units + governance integration + full regression — green.
11. **Suite:** lib (801 pass, 1 unrelated VRAM-probe flake) + MoE integration.
12-13. Commit / CI: see git log + the push.
14. **Risks:** sim-based int8 cert (vs a real int8-tier cert); single-process env
    caching (correct for CLI, override for tooling); size-only tier integrity
    (pre-existing).
15. **Limitations:** small validation set (6 short greedy cases); no perplexity /
    long-text metric yet; DeepSeek/MLA still out of scope.
16. **Recommend next:** (a) **AV exclusion** (operational, biggest win); (b)
    broaden the validation set + add a real-int8-tier cert path; (c) **int4
    sim-cert** (evidence-gated, never on intuition); (d) manifest content
    checksums; (e) DeepSeek coverage. **Do not** pursue more expert-matmul / CUDA
    / tier-consolidation (measured dead-ends).

## Files

- `src/moe/cert.rs` (new) — certificate + validation set + runner + guard.
- `src/moe/numeric_policy.rs` — (unchanged API; reused).
- `src/moe/residency.rs` — in-process int8-sim override.
- `src/moe/runtime.rs` — `generate_full`, `model_id`, `certify_model`,
  `apply_certification_guard`, manifest unchanged (v5).
- `src/bin/atenia.rs` — `moe-certify` subcommand.
- `tests/moe_cert_governance_test.rs` (new) + `docs/HANDOFF_NUMERIC_POLICY_3.md`
  (this) + `docs/STATUS.md` + `docs/NUMERIC_POLICY_AUDIT.md` (gap #1 closed).
