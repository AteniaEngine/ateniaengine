# HANDOFF — MOE-PRODUCT-2: opt-in DeepSeek-V2-Lite in productive `generate`

Milestone: **MOE-PRODUCT-2** — enable the **opt-in productive routing of DeepSeek-V2-Lite**
(MLA) in `atenia generate`, via `MoeSpecResolver` + the unchanged `MoeRuntime`. **V3 stays
non-runnable**, the **dense path stays intact**, **fail-loud stays the default**, and **no
numerics / thresholds / cert manifest** are touched.

## Result

`atenia generate <deepseek-v2-lite>` now, **with `ATENIA_ENABLE_MOE=1`**, routes to the
controlled MoE runtime — joining Mixtral and Qwen-MoE as a productively-routable family.
Without the opt-in it fails loud (exit 2). The DeepSeek-V3 routing mechanism and the
Q-LoRA variants (DeepSeek-V2-236B / V3) remain refused.

| Checkpoint | Route (opt-in) | Why |
|---|---|---|
| Dense | Dense (unchanged) | not MoE |
| Mixtral / Qwen-MoE | RunMoe | MOE-PRODUCT-1 |
| **DeepSeek-V2-Lite** (MLA, `q_lora_rank=null`, no V3 router) | **RunMoe** | **MOE-PRODUCT-2** |
| DeepSeek-V2-236B / V3 (Q-LoRA `q_a_proj`) | Refused | unsupported variant (Q-LoRA) |
| DeepSeek-V3 routing (`e_score_correction_bias`) | Refused | V3 routing = L0 mechanism-only / non-runnable |
| MoE without opt-in | fail-loud (exit 2) | default protection |
| unknown / uncertified | Refused | fail-loud |

## What blocked DeepSeek before, and what changed

DeepSeek-V2-Lite was never a runtime problem — `MoeRuntime::load_from_dir` +
`controlled_moe_generate` already load and `generate` it (the MLA family; `deepseek_scale`
certifies generate→EOS, and `atenia moe-generate` already ran it). The only deferral was in
the **routing deciders**, which refused `DeepSeekMoe`. MOE-PRODUCT-2:

1. `src/adapter_toolkit/moe_resolver.rs` — `arch_for_productive_routing(DeepSeekMoe)` now
   returns `Some(MoeArch::DeepSeekV2Lite)` (was `None`); `MoeSpecResolver::route` therefore
   resolves it (runnable) and routes it behind the opt-in.
2. `src/moe/production.rs` — `decide_route` no longer special-cases `DeepSeekMoe` to
   `Refused` (it flows through the generic runnable + opt-in branch), and a new
   **unsupported-variant guard** flags the **DeepSeek-V3 routing marker**
   (`e_score_correction_bias`) as non-runnable (defence-in-depth; real V3 is also caught by
   the existing Q-LoRA `q_a_proj` guard). The cert manifest is **not** touched (the new
   message is hardcoded; no manifest entry required).
3. `src/cli_generate.rs` — comment only (it already routes via `MoeSpecResolver::route`).

## Safety: only the clean V2-Lite shape reaches the runtime

`diagnose_moe` refuses, **before** routing, any DeepSeek with Q-LoRA (`q_a_proj`/`q_b_proj`)
or the V3 routing marker (`e_score_correction_bias`) → `certified_runnable=false` → Refused.
So the only DeepSeek shape that reaches `RunMoe` is **DeepSeek-V2-Lite** (MLA,
`q_lora_rank=null`, softmax router, no V3 bias) — the MoE-certified L3 family. YaRN +
dense-first are already supported by the certified runtime (MLA-0/MLA-3).

## Guardrails (FASE 5 of the spec)

- **Dense generate untouched** (no numerics / thresholds / loader change).
- **Fail-loud default** (MoE without `ATENIA_ENABLE_MOE=1` → exit 2, family-aware message).
- **V3 cannot run as a real model:** Q-LoRA + the V3 routing marker are both refused; the
  V3 routing arch is `is_runnable_model()=false` and is never produced from a checkpoint.
- **No non-certified family enabled; cert manifest + ADR-007 untouched.**
- **No Q-LoRA, no latent cache, no performance work** (out of scope, untouched).

## Validation

- New tests: `moe::production` — `deepseek_v2_lite_runnable_routes_with_opt_in`,
  `v3_router_marker_is_unsupported_variant`; `adapter_toolkit::moe_resolver` —
  `route_deepseek_v2_lite_runnable_with_opt_in`, updated `productive_arch_mapping` /
  refuse test; integration `tests/moe_product_routing_test.rs` — DeepSeek-V2-Lite shape
  (`deepseek_scale` fixture) routes `NeedsOptIn` → `RunMoe` with the opt-in.
- Regressions: `moe_integrate_routing` + `moe_production` green; full `cargo test --lib` green.
- **No heavy real-model generate in CI** (FASE 5 manual run skipped — the ~15.7B
  DeepSeek-V2-Lite real generate is long/heavy; EOS/tokenizer is covered by the
  `deepseek_scale` topology generate→EOS cert + the routing tests on the V2-Lite-shaped
  fixture).

## Files

- `src/moe/production.rs` (un-defer DeepSeek in `decide_route` + V3-router unsupported
  guard + tests), `src/adapter_toolkit/moe_resolver.rs` (DeepSeek arch mapping + tests),
  `src/cli_generate.rs` (comment), `tests/moe_product_routing_test.rs` (DeepSeek case).
- docs: this handoff + `CLI.md`, `ADAPTER_TOOLKIT_V2.md`, `STATUS.md`.

## Operating it

```
ATENIA_ENABLE_MOE=1 atenia generate --model models/DeepSeek-V2-Lite --prompt "..." --max-tokens 32
# Without the flag: fail-loud (exit 2). DeepSeek-V2/V3 (Q-LoRA) or V3 routing: refused.
```

Caveats: experimental, opt-in, CPU-only, **slow** (MLA + disk-tier expert streaming);
**MoE-certified L3 = active-path-certified, NOT dense ADR-004 `CERTIFIED`**; **L4
reserved/unreachable**; the **latent/absorb KV cache is NOT implemented** (correctness path,
not the perf cache).

## Next recommended step

- **MOE-PERF-1:** MoE-generate throughput (CPU forward + bounded-cache disk-tier; today it
  is slow) — performance only, no numerics change.
- **MOE-PRODUCT-3 / CLI UX:** surface MoE status + the opt-in hint in `--help` / a
  `diagnose`-style command; optional progress for the long MLA prefill.

## Risks / caveats

- DeepSeek-V2-Lite productive generate now reachable behind the opt-in but **not exercised
  end-to-end in CI** (heavy model) — covered structurally (routing + topology generate→EOS
  cert). A real run is the optional MOE-PRODUCT-2 follow-up on a host with the weights.
- The V3-routing guard keys on the `e_score_correction_bias` tensor name; a future V3 point
  release renaming it would need the marker updated (real V3 is still caught by Q-LoRA).
