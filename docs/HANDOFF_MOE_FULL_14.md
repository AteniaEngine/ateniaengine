# HANDOFF — MOE-FULL-14: controlled production MoE path

Milestone: **MOE-FULL-14** — turn the experimental MoE runtime into a
**controlled production path**, **without** declaring general support. A model
directory is gated through detection → family classification → unsupported-
variant check → certification manifest → explicit opt-in → the MoE runtime.
**No new architecture / families / graph ops.** The dense path is untouched.
Predecessor: MOE-FULL-13 (`0a703a1`).

## MoE support status

| Layer | Status |
|---|---|
| **Experimental runtime** | ✅ Mixtral / Qwen-MoE (graph) + DeepSeek-MoE (MLA), tiny fixtures, certified vs HF |
| **Controlled product path** | ✅ `moe::controlled_moe_generate` + `atenia moe-generate` (opt-in, manifest-gated) |
| **Unsupported variants** | Qwen3-MoE (QK-norm), DeepSeek Q-LoRA (refused with a clear message) |
| **General production support** | ❌ NOT declared — tiny fixtures only, no large-scale certification, no tokenizer/text CLI, no fail-loud lift on the *dense* loader |

### Remaining blockers to drop the "experimental" label

1. Large-checkpoint certification (Mixtral-8x7B, Qwen2-57B-A14B) via the MOE-1
   partial/sub-reference methodology at scale.
2. Tokenizer-backed text CLI (current `moe-generate` is token-id based).
3. Qwen3-MoE (QK-norm) + DeepSeek Q-LoRA/YaRN coverage.
4. A reviewed lift of the **dense** loader's fail-loud for certified families.
5. Throughput (decode-graph reuse, residency hot-path, VRAM tier).

## 1. Controlled production path (`src/moe/production.rs`)

`controlled_moe_generate(dir, prompt_ids, max_new)` gates a model directory:

```text
detect MoE → unsupported-variant check → classify family
           → manifest scope (runnable?) → opt-in flag → MoeRuntime → generate
```

Each gate returns a clear error: `NotMoe`, `UnsupportedVariant(..)`,
`NotCertified{family,scope}`, `NotEnabled{family,scope}`, `Runtime(..)`.
`diagnose_moe(dir)` is the **read-only** counterpart (status + actionable
message, never executes).

## 2. Opt-in flags

- **Env**: `ATENIA_ENABLE_MOE=1` (the MOE-FULL-14 production flag), accepted
  alongside the legacy `ATENIA_EXPERIMENTAL_MOE=1` (so prior callers/tests keep
  working). **Chosen** because env is automation-friendly and pairs with a CLI
  flag.
- **CLI**: `atenia moe-generate --experimental-moe` (sets the env for the run).
- Without either, every MoE path refuses with an actionable message; the dense
  loader's fail-loud guard is unchanged.

## 3. Families that can run

| Family | Scope (manifest) | Runs via controlled path? |
|---|---|---|
| Mixtral | `certified_partial` | ✅ (with opt-in) |
| Qwen-MoE | `certified_partial` | ✅ (with opt-in) |
| DeepSeek-MoE | `certified_fixture` | ✅ (with opt-in) |
| Qwen3-MoE (QK-norm) | `unsupported` | ❌ clear "unsupported variant" |
| DeepSeek Q-LoRA | `unsupported` | ❌ clear "unsupported variant" |

## 4. Numcert / MoE manifest (`src/moe/manifest.rs`)

A machine-readable manifest, embedded from `fixtures/moe/moe_cert_manifest.json`
(single source of truth, no runtime file dependency). Records per family:
family, scope (`certified_fixture` / `certified_partial` / `experimental` /
`unsupported`), runtime path, attention type, MoE layout, end-to-end drift,
argmax match, generate/eos, limitations — plus the unsupported-variant
fingerprints (QK-norm, Q-LoRA). Kept **separate** from the dense
`nn::llama::numcert` matmul-precision manifest. The controlled path consults
`scope_for(family)` to decide runnability.

## 5. Fail-loud (controlled)

The **dense** loader still refuses to load MoE as dense (unchanged). The
controlled path is the only door, and it is doubly gated: a family must be
**certified** (manifest scope runnable) **and** the opt-in flag set. Unsupported
variants are refused with the manifest's reason. Example messages:

```text
MoE checkpoint detected.
Family: Mixtral.
Status: certified_partial.
The controlled MoE runtime is opt-in: set ATENIA_ENABLE_MOE=1 (or pass
--experimental-moe) to run it. The dense loader still refuses MoE.
```
```text
MoE checkpoint detected.
Family: Qwen3-MoE variant.
Status: unsupported (QK-norm attention not implemented).
```

## 6. Partial / sub-reference certification (FASE 6)

The `*_moe_layer0` fixtures are layer-0 MoE blocks sliced from **real**
HuggingFace checkpoints, each with a committed HF f64 reference. New
`moe_partial_cert_test.rs` certifies them:

| Real checkpoint | layer-0 MoE block vs HF |
|---|---|
| Mixtral | **1.164e-10** |
| Qwen1.5-MoE | **2.910e-11** |
| Qwen2-MoE | **5.821e-11** |
| Qwen3-MoE (block only) | **5.821e-11** |

So the family-distinguishing MoE block is certified against **real** checkpoints
at machine precision (the MOE-1 sub-reference methodology — a full f64 forward at
scale is infeasible, a single real layer is not). DeepSeek-MoE has no real-
checkpoint layer-0 fixture; its block is certified on a synthetic fixture
(MOE-FULL-11, 2.196e-04) — documented in the manifest.

## 7. Robustness

`controlled_path_gates...` (opt-in on/off), `dense_dir_is_not_moe`,
`qwen3_qk_norm_is_unsupported_variant`, `diagnose_reports_certified_mixtral_status`;
CLI smoke `moe_generate_runs_with_opt_in_flag` / `moe_generate_refuses_without_opt_in`.
Plus MOE-FULL-13's corrupt-router / corrupt-MLA checks. Existing CLI suites
(errors/ux/diagnostics) unchanged → dense CLI unaffected.

## Tests

- `src/moe/manifest.rs` — 2 unit; `src/moe/production.rs` — 1 unit.
- `tests/moe_partial_cert_test.rs` — 4 (real-checkpoint layer-0 blocks).
- `tests/moe_production_test.rs` — 4 (gating + dispatcher).
- `tests/cli_moe_generate_test.rs` — 2 (CLI smoke: runs with flag, refuses
  without).

Local validation (real output, exit 0): full lib suite
`cargo test --lib --release -- --test-threads=1` → **787 passed / 0 failed /
1 ignored** (was 784; +3). MoE integration: production 4/4, partial-cert 4/4
(1e-10..1e-11), cli-moe 2/2, certification 4/4, deepseek 4/4, qwen 3/3, mixtral
3/3, robustness 6/6. CLI regression: errors 10/10, ux 9/9, diagnostics 8/8
(dense CLI unaffected).

## Files modified

* `src/moe/manifest.rs` — new (MoE cert manifest + parser).
* `src/moe/production.rs` — new (controlled dispatcher + diagnose + gating).
* `src/moe/runtime.rs` — `MoeRuntime::load_from_dir`.
* `src/moe/family.rs` — `ATENIA_ENABLE_MOE` accepted by the opt-in.
* `src/moe/mod.rs` — module registration + re-exports.
* `src/bin/atenia.rs` — `atenia moe-generate` subcommand (additive; dense
  `generate` untouched).
* `fixtures/moe/moe_cert_manifest.json` — new (embedded).
* `tests/moe_partial_cert_test.rs`, `tests/moe_production_test.rs`,
  `tests/cli_moe_generate_test.rs` — new.
* `docs/HANDOFF_MOE_FULL_14.md` — this file; plus `docs/MOE_OVERVIEW.md`,
  `docs/MOE_FULL_PATH_AUDIT.md`, `docs/MODEL_FAMILY_VALIDATION.md`,
  `docs/CLI.md` — updated.

No new architecture / families / graph ops. Dense path + dense CLI untouched;
fail-loud preserved except the explicit, manifest-gated, opt-in MoE path.

## Recommendation

The MoE campaign should **continue** but the *correctness* track is essentially
complete: three families certified (tiny + real-checkpoint blocks), a controlled
opt-in product path, a manifest, and clear gating. The remaining work is
**scale + integration** (large-checkpoint certification, tokenizer CLI, the
fail-loud lift) and **variant coverage** (Qwen3 QK-norm, DeepSeek Q-LoRA/YaRN) —
each a separate, bounded milestone. MoE remains **experimental** until at least
large-checkpoint certification (#1) lands.
