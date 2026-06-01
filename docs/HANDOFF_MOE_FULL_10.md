# HANDOFF — MOE-FULL-10: controlled productive Mixtral runtime (opt-in)

Milestone: **MOE-FULL-10** — the first **productive** MoE path in Atenia. Behind
an explicit opt-in, it loads a real tiny **Mixtral** checkpoint and generates to
EOS, reusing every certified MoE component. **No general MoE support is
declared; only Mixtral is enabled, only with the opt-in.** The dense loader's
fail-loud guard is unchanged. Predecessor: MOE-FULL-9 (`d3775c3`).

## 1. Controlled opt-in lift of fail-loud

`ATENIA_EXPERIMENTAL_MOE=1` (`moe::family::experimental_moe_enabled`) gates the
new `moe::runtime::MixtralRuntime`:

- **With** the opt-in: `MixtralRuntime::load_from_files(config.json, safetensors)`
  loads a recognised Mixtral checkpoint.
- **Without** it: `load_from_files` returns `MixtralRuntimeError::OptInDisabled`
  — refuses exactly as before.
- The **dense loader (`weight_mapper.rs`) is untouched** — it still fails loud on
  any MoE checkpoint (a dense load of MoE is always wrong). The opt-in selects a
  *separate* runtime; it does not lift the dense guard. The loader's family-aware
  message now points at the opt-in runtime.

## 2. Productive runtime wiring (reuses the certified pipeline)

`load_from_files` threads the certified components in order — no duplicated
logic, no test helpers:

```text
opt-in gate
  → parse HF config.json (dims, experts, top-k, rope_theta, eos_token_id)
  → classify_family  → must be Mixtral            (MOE-FULL-9)
  → MixtralAdapter::recognize + validate_family_config  (MOE-FULL-3/9)
  → per layer: RealMoeLayer::assemble              (MOE-11)
               gqa::to_mha_kv  (K/V tiling)        (MOE-FULL-9)
               ResidentExpertLayer(RAM) + ExpertCache, self-validated (MOE-FULL-8/9)
  → TinyMixtralWeights
generate(prompt, max_new)
  → generate_greedy_tiny_eos  (prefill + KV cache + decode + EOS)  (MOE-FULL-6/7)
```

- **Residency + expert cache are wired and self-validated at load**: for each
  layer the runtime builds a `ResidentExpertLayer` (RAM tier) + an `ExpertCache`
  and asserts `forward_cached` reproduces the certified `RealMoeLayer::forward_auto`
  block (≤1e-5) — the MOE-FULL-8/9 invariant, now a runtime self-check, not just
  a test. The generation hot path runs the certified RAM MoE block (bit-identical
  to the residency RAM tier); routing decode *through* residency+cache is a perf
  optimisation, deliberately out of scope (correctness first).
- **GQA** is handled transparently by `gqa::to_mha_kv` (identity for the MHA
  fixture; real GQA validated at 5.960e-08 in MOE-FULL-9).

## 3. Minimal productive Mixtral path

A real `load → generate → EOS` over a tiny Mixtral checkpoint
(`fixtures/moe/mixtral_tiny_config.json` + the committed
`full_mixtral.safetensors`, MOE-FULL-6). `eos_token_id=20`.

## 4. End-to-end validation

```text
ATENIA_EXPERIMENTAL_MOE=1
  MixtralRuntime::load_from_files(mixtral_tiny_config.json, full_mixtral.safetensors)
  generate([22,25,29], 8) = [17, 20]      ← greedy [17,20,10,17] stopped at eos=20
```

- **Loader real**: parses the HF `config.json`, opens the safetensors, recognises
  Mixtral, validates the adapter + config.
- **Runtime real**: certified prefill/KV-cache/decode loop.
- **Generation + EOS real**: stops on token 20 before `max_new_tokens=8`.
- **Determinism**: identical output on repeat.
- **Opt-in**: without `ATENIA_EXPERIMENTAL_MOE=1` the same call returns
  `OptInDisabled`.
- **No dense regression**: dense generation suite green; the dense loader guard
  and the legacy fail-loud message are unchanged.

## What stays protected by fail-loud

- The **dense loader** refuses every MoE checkpoint, always.
- The MoE runtime itself refuses **without** the opt-in.
- **Not enabled**: Qwen-MoE, DeepSeek-MoE, Mixtral-8x7B (only tiny Mixtral
  exercised), VRAM tier, quantised experts, batching, CLI, general support.

## What family is enabled

**Mixtral only**, and only behind `ATENIA_EXPERIMENTAL_MOE=1`. Qwen-MoE /
DeepSeek-MoE are recognised (MOE-FULL-9) but **not** runnable.

## Remaining risks / work (MOE-FULL-11)

- Decode hot path materialises the MoE block in RAM each step (no residency
  routing, no perf work) — fine for tiny, not for Mixtral-8x7B.
- No VRAM tier, no real large-checkpoint certification, no CLI, no Qwen-MoE/
  DeepSeek productive enablement.
- The runtime reuses the experimental tiny generation loop (graph rebuilt per
  step); productive decode-graph reuse is later work.

## Tests

- `src/moe/runtime.rs` — 1 unit (`opt_in_disabled_refuses`).
- `tests/moe_mixtral_runtime_test.rs` — 3 integration
  (`controlled_mixtral_load_generate_eos` [opt-in gate + load + generate + EOS +
  determinism], `dense_is_not_a_mixtral_family`).

Local validation (real output, exit 0): full lib suite
`cargo test --lib --release -- --test-threads=1` → **781 passed / 0 failed /
1 ignored** (was 780; +1). MoE integration: runtime 2/2 (`generate([22,25,29],8)
= [17,20]`), decode 5/5, GQA 3/3, residency 4/4, family-loader 4/4,
**moe_loader_failloud 3/3**. Dense regression: `m5_da_generation_loop_test` 4/4.

## Files modified

* `src/moe/runtime.rs` — new (`MixtralRuntime`, opt-in load + generate + EOS +
  residency self-check; 1 unit test).
* `src/moe/generate.rs` — `generate_greedy_tiny_eos` (EOS-aware); 3-arg
  `generate_greedy_tiny` delegates with empty EOS (MOE-FULL-7 behaviour
  unchanged).
* `src/moe/family.rs` — `experimental_moe_enabled` / `EXPERIMENTAL_MOE_ENV`;
  fail-loud message references the opt-in runtime.
* `src/moe/mod.rs` — `pub mod runtime;` + re-exports.
* `tests/moe_mixtral_runtime_test.rs` — new (3 integration tests).
* `fixtures/moe/mixtral_tiny_config.json` — new HF-style Mixtral config for the
  committed weights (no weight duplication).
* `docs/HANDOFF_MOE_FULL_10.md` — this file.
* `docs/MOE_FULL_PATH_AUDIT.md`, `docs/MOE_OVERVIEW.md`,
  `docs/MODEL_FAMILY_VALIDATION.md` — MOE-FULL-10 recorded; remaining work
  renamed MOE-FULL-11.

No CLI / dense-loader / runtime-productive (dense) / Adapter-Toolkit / CUDA
change. Dense models unaffected; fail-loud preserved everywhere except the
explicit, Mixtral-only, opt-in runtime.
