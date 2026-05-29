# Numeric Certification — Atenia Engine

## What this is

Atenia Engine ships with **per-checkpoint numeric certificates** that document, with empirical data, the drift envelope of every supported model under each execution mode (`certified`, `fast`). The certificate is a JSON manifest versioned alongside the model checkpoint, named `<model>.numcert.json`.

The contract is simple: the engine offers two execution modes; a manifest tells you, for a specific checkpoint on a specific Atenia version, what numerical drift each mode produces against an F64 ground-truth reference. Operators choose modes with data, not folklore. Auditors verify the manifest with a single `cargo test` invocation.

This is the production form of the value proposition the project carries on its README: *correctness as a per-checkpoint certificate, not a runtime tax*. ADR-004 (F64 reference) provides the truth; ADR-005 (fast mode envelope) provides the trade-off framework; this document operationalises both into a per-checkpoint artefact.

## The two execution modes

| Mode | Activated by | GPU GEMM | Drift profile | When to use |
|------|--------------|----------|---------------|-------------|
| `certified` (default) | (no flag) or `ATENIA_M8_BF16_KERNEL=1` without `ATENIA_FAST_MODE` | `cublasGemmEx(F32, F32, F32)` with `CUBLAS_COMPUTE_32F_FAST_TF32` (TF32 Tensor Cores) — Path B M8.4c + M10.2.0 | ADR-004 strict (`max_abs_diff < 0.5` vs F64) holds on the 4-model fixture | Research, audited deployments, any output whose numerical equivalence to F64 ground truth must be defensible |
| `fast` | `ATENIA_FAST_MODE=1` | `cublasGemmEx(BF16, BF16, F32)` — native BF16 Tensor Cores (M10.2.1) | Industry-standard BF16 drift profile, ADR-005 envelope, may fail ADR-004 strict per-model | Chat / generation workloads where argmax-equivalence at near-tie positions is acceptable and the per-matmul speedup matters |

Both modes share storage (BF16 weight in VRAM via M4.7.2 / M8.3) and tier-aware loading (M6 / M7 / M8.7). The mode swap only changes the GEMM dispatch.

## Schema

A manifest is JSON with the following top-level fields. Reference manifests for the four fixture models are in `docs/numcert/`.

```json
{
  "schema_version": "1.0.0",
  "model": "Human-readable checkpoint name",
  "model_family": "Builder family (llama-2, llama-3-rope-scaling, llama-2-with-qkv-bias, ...)",
  "model_path_hint": "Suggested relative path inside an Atenia checkout",
  "atenia_version": "Atenia version that produced these numbers",
  "f64_fixture_version": "Version of the F64 reference fixture used",
  "f64_fixture_test": "Path to the test that consumes this fixture",
  "generated_at": "YYYY-MM-DD",
  "generated_on": "Hardware string (GPU + RAM + OS)",

  "drift_envelope": {
    "certified_mode": {
      "max_abs_diff_vs_f64": <number>,
      "argmax_match_4_of_4": <bool>,
      "adr_004_strict_pass": <bool>,
      "adr_004_threshold": 0.5,
      "adr_004_margin": <number>,        // threshold / drift; >1 means inside the gate
      "kernel_path": "Description of the kernel chain used"
    },
    "fast_mode": {
      "max_abs_diff_vs_f64": <number>,
      "argmax_match_4_of_4": <bool>,
      // When 4_of_4 is false, additional fields document the partial match:
      "argmax_match_count": <int>,
      "argmax_match_total": <int>,
      "argmax_mismatch_positions": [<int>, ...],
      "adr_004_strict_pass": <bool>,
      "adr_004_threshold": 0.5,
      "adr_004_margin": <number>,        // when pass = true
      "adr_004_overshoot_factor": <number>, // when pass = false; drift / threshold
      "kernel_path": "Description of the kernel chain used"
    }
  },

  "recommended_mode": "certified" | "fast" | "quantized",
  "recommended_mode_rationale": "Why this checkpoint is recommended for this mode",

  "per_tensor_policy": null,
  "per_tensor_policy_status": "Reserved for M10.3+ dispatcher",

  "verification": {
    "command": "cargo test command that reproduces these numbers",
    "env_for_certified": "Env vars for certified mode",
    "env_for_fast": "Env vars for fast mode",
    "expected_dispatch_counter_certified": "What vram_bf16_matmul_count should advance by",
    "expected_dispatch_counter_fast": "What vram_bf16_native_matmul_count should advance by"
  }
}
```

### Why two drift numbers per mode are not enough on their own

`max_abs_diff_vs_f64` is the worst per-element drift across the 4-position fixture forward output. It is a useful end-to-end summary but it does not tell you **which token decision the drift might flip**. The `argmax_match_*` fields capture that: they record whether the engine, under the mode in question, would have produced the same next-token argmax as the F64 reference at every position. A model can satisfy `max_abs_diff < 0.5` and still flip an argmax at a near-tie position — Llama 3.2 1B under fast mode is exactly this case (drift 0.27, but the position-2 argmax differs from F64).

For chat workloads the argmax match is the structurally relevant gate; for scientific workloads `max_abs_diff` matters in its own right.

## How a manifest is generated

Today (M10.3 v1) manifests are generated **manually** from the data the F64 fixture prints. The process for each model in the 4-model fixture:

1. Run the fixture in `certified` mode:
   ```sh
   cargo test --release --test m8_5_full_family_validation_test -- \
     --ignored m8_5_<model>_under_bf16_kernel_matches_f64 --nocapture
   ```
   (with the appropriate `*_SAFETENSORS_PATH` and `ATENIA_M8_BF16_KERNEL=1` env vars).

2. Re-run with `ATENIA_FAST_MODE=1` added.

3. Collect from the test output:
   - `max drift vs F64 = <number>` (one per mode)
   - The 4 argmax MATCH/MISMATCH lines (one per mode)
   - The `BF16 matmul calls: certified=<n> native=<m>` line (verifies dispatch routing)

4. Fill in the manifest schema with the numbers; calculate `adr_004_margin` (= 0.5 / drift) when drift < 0.5, or `adr_004_overshoot_factor` (= drift / 0.5) when drift >= 0.5.

5. Pick `recommended_mode` based on:
   - Fast mode passes ADR-004 strict AND argmax 4/4 → `fast`
   - GGUF / quantized checkpoints validated by functional smoke + documented drift → `quantized`
   - Otherwise → `certified`
   The rationale field explains the trade-off in plain language so an operator reading the manifest can override the recommendation knowingly.

`quantized` is a functional certification mode for GGUF-style checkpoints whose source format intrinsically loses precision. It is not an alias for ADR-004 `certified` execution or ADR-005 `fast` execution: the loader decodes GGUF tensors into Atenia's resident storage and validates the checkpoint through smoke tests plus explicit drift documentation.

A future milestone is expected to automate this: a `cargo run --bin atenia-certify --model <path>` subcommand that produces the manifest as a structured artefact instead of hand-editing JSON. The schema is stable enough today (`schema_version: "1.0.0"` for safetensors/F64 manifests, `schema_version: "2.0.0"` with `schema_variant: "gguf-functional"` for GGUF manifests) that automation is primarily an ergonomics upgrade, not a contract change.

## How an operator verifies a manifest

The whole point of the manifest is that **anyone can verify it** without trusting the publisher. The `verification` block is the contract: it specifies the exact `cargo test` invocation that reproduces the numbers, with the env vars and expected counter deltas. An auditor:

1. Clones the Atenia repo at the version the manifest claims (`atenia_version` field).
2. Sets the fixture-checkpoint env var (`*_SAFETENSORS_PATH=<their copy of the model>`).
3. Runs the verification command for `certified` mode.
4. Reads the printed drift; compares to `drift_envelope.certified_mode.max_abs_diff_vs_f64`.
5. Repeats for `fast` mode with `ATENIA_FAST_MODE=1`.
6. Confirms the `BF16 matmul calls:` line shows the expected counter delta — proving the dispatcher actually routed through the claimed kernel, not silently fell back to a different path.

If any of those checks disagree with the manifest, the manifest is invalid. There is no opaque infrastructure between the manifest claim and the verification command.

## How the manifest is versioned

A manifest is produced for a specific `(checkpoint, atenia_version)` pair. When either side changes the manifest must be regenerated. Concretely:

- **Same checkpoint, new Atenia version**: re-run the fixture and check the drift numbers; if the new version changes the kernel or the GEMM compute mode (e.g. M10.2.0's TF32 swap), the manifest needs to be updated to reflect the new envelope. The verification block must point to the new commit / version.
- **Same Atenia version, new checkpoint**: a fine-tuned variant of an existing model is structurally a new checkpoint; the manifest of the base model does not transfer. Re-run the fixture from scratch.
- **Atenia version where the fixture itself changed**: bump `f64_fixture_version`. The schema's `f64_fixture_version` field is precisely so a v2 fixture (e.g. a 10-model expansion in M11) can coexist with v1 manifests until they are regenerated.
- **GGUF / quantized checkpoint**: use `schema_variant: "gguf-functional"` and `recommended_mode: "quantized"`. ADR-004 strict thresholds remain documented as context, but they are not the pass/fail gate for lossy quantized formats.

The recommended distribution flow: the manifest is a sibling file to `model.safetensors` in the model directory. A model card (HuggingFace or otherwise) can publish the manifest as a downloadable artefact alongside the weights. Atenia's `models/` checkout convention (already used by the F64 fixture) makes this trivial to wire.

## What this does NOT promise (yet)

- **Per-tensor policy dispatch** is reserved for M10.3+ in v1 safetensors manifests. GGUF functional manifests may use `per_tensor_policy.default = "quantized"` to make the model-level quantized path explicit; this is not a promise of ADR-004 strict equivalence.
- **Coverage beyond the 4-model fixture**: M10.3 v1 ships manifests for TinyLlama 1.1B, SmolLM2 1.7B, Qwen 2.5 1.5B, Llama 3.2 1B (the M4.6 family). Llama 2 7B and 13B Chat have end-to-end smoke evidence but no F64 reference (size). M11 expands to the top-10 model list with manifests for each.
- **Statistical claims at scale**: each manifest is a 4-position end-to-end snapshot. Adversarial prompts or long-context sequences may surface drift profiles outside the fixture's coverage. The manifest's `recommended_mode_rationale` flags this where relevant (SmolLM2 and Llama 3.2 explicitly).

## Reference manifests

The four manifests shipped in this repo (`docs/numcert/`) are reference copies for the M4.6 family. Operators bringing their own checkpoint should generate a fresh manifest using the procedure above. The fixture models double as the regression sentinels for any future change to the kernel chain — if a manifest's drift would change under a kernel update, that is the signal to revisit ADR-004 / ADR-005 with new evidence.

## Relationship to AQS (experimental — not production certification)

**AQS (Atenia Quantization Search)** is a separate, experimental research
subsystem (see [AQS_OVERVIEW.md](./AQS_OVERVIEW.md)). It produces its own
artefacts — a **search report** (classifying candidate quantization
policies as `certified` / `useful_lossy` / `failed` against the ADR-004
gate) and a **`3.0.0-draft` manifest** — via the `atenia search` CLI.

These AQS outputs are **experimental and must not be confused with the
production manifests described in this document**:

- The productive numeric certification described here (`schema_version`
  `1.0.0` / `2.0.0`) is the contract the **runtime** and operators rely on,
  governed by **ADR-004 / ADR-005**. It is unchanged by AQS.
- The AQS `3.0.0-draft` manifest is a **draft** — never consumed by the
  runtime, never a production certificate. The `-draft` suffix is
  deliberate.
- AQS confirmed empirically that no weight-only policy (plain INT8, β
  outlier, AWQ, hybrid, GPTQ) crosses ADR-004 strict on TinyLlama; BF16
  remains the only certified policy and AWQ is the best *useful-lossy*
  option. This does not alter any production certification claim.

If/when AQS graduates a manifest format for runtime use, it will go through
the same ADR process and schema-versioning as the manifests above.

## Related

- [ADR-004 — F64 reference as default](./decisions/ADR-004-f64-reference-as-default.md)
- [ADR-005 — Fast mode (BF16-TC) drift envelope](./decisions/ADR-005-fast-mode-bf16-tc-envelope.md)
- [ROADMAP.md §"Numeric contract strategy"](../ROADMAP.md#numeric-contract-strategy)
- [AQS_OVERVIEW.md — experimental quantization search](./AQS_OVERVIEW.md)
- The 4-model F64 fixture: [`tests/m8_5_full_family_validation_test.rs`](../tests/m8_5_full_family_validation_test.rs)
