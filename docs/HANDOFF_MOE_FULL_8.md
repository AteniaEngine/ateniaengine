# HANDOFF — MOE-FULL-8: experimental expert residency (RAM / NVMe)

Milestone: **MOE-FULL-8** (demonstrate the MoE architecture can sit on
Atenia's real residency infrastructure — the `SharedParam` tiers used by the
productive `WeightStore` — **without materialising all experts in RAM**). This
closes the "large-model residency" blocker called out in `docs/MOE_OVERVIEW.md`
and `docs/MOE_FULL_PATH_AUDIT.md`, at the experimental/CPU+NVMe level.
**Experimental, CPU+NVMe, test-only.** No productive loader / runtime / Adapter
Toolkit / CLI / pipeline / WeightStore / CUDA change, no fail-loud lift.
Predecessor: MOE-FULL-7 (`307065e`).

## Problem

The certified MoE block (MOE-FULL-7) keeps **every** expert's weights in RAM:
`RealMoeLayer.routed.experts: Vec<MoeDenseExpert>`, each a trio of f32 `Vec`s.
Fine for tiny fixtures; the headline blocker for Mixtral-8x7B / large Qwen-MoE /
DeepSeek-MoE, where experts dominate the parameter count and cannot all be
resident at once.

## Scope built (honest, bounded)

`src/moe/residency.rs` — a `ResidentExpertLayer` whose expert weights live in a
chosen **residency tier** and are resolved **on demand**:

```text
route(x)  ──► softmax weights            (router stays RAM-resident)
top_k(weights, k)  ──► selected indices + renormalised weights
for e in selected:                       (NOT all experts)
     resolve expert e from its tier  ──► transient MoeDenseExpert
     y_e = expert.forward(x)  ;  out += w · y_e
     drop the materialised weights
(+ optional shared expert per convention)
```

- **Tiers** (`ExpertTier`): `Ram` (`SharedParam::F32`) or `Disk`
  (`SharedParam::Disk` via `crate::tensor::disk_tier`, **zero host RAM** until
  requested). Built from a certified `RealMoeLayer` with
  `ResidentExpertLayer::from_real_layer(&layer, tier)`.
- **On-demand resolution**: a forward routes, selects top-k, and materialises
  **only** those experts (`to_tensor()` → `ensure_cpu()` reads NVMe for the
  Disk tier), runs the SwiGLU, accumulates, and drops the weights. The router
  weight and shared-expert gate stay RAM-resident (small).
- **Reuses the certified math unchanged**: the same `route` softmax, the same
  `top_k_routing_with` selection/renormalisation, the same
  `MoeDenseExpert::forward`, the same weighted combine and shared-expert
  convention as `RealMoeLayer::forward_with`. Only *where the weights live* and
  *when they are materialised* differ → output is bit-identical by
  construction.
- **Consumes, does not modify**, `SharedParam` / `WeightStore` / `disk_tier`.
  No new storage variant, no infra change.

### Honest simplifications (documented)

- CPU + NVMe only. The `Cuda` (VRAM) tier exists in `SharedParam` and the
  design admits it, but it is not exercised here (needs a device).
- No quantised experts, no GQA, no productive load-path wiring. This is a
  *residency mechanism* demo, not Mixtral support.
- Per-forward materialise-then-drop (no expert cache / prefetch). A real
  implementation would add an LRU expert cache; out of scope here.

## Evidence of residency

| Metric (128 experts, d_model 64, d_ff 128, NVMe tier) | Value |
|---|---|
| Host RAM resident (`resident_ram_bytes`) | **32 768 B** (router only) |
| Full materialisation (`full_materialization_bytes`) | 12 615 680 B |
| **Residency saving** | **385×** |
| Experts materialised per forward (top-k=2) | **2** (not 128) |
| Bytes materialised per forward | 2 × 98 304 B |

The host-RAM footprint is the router alone (`n · d_model · 4`); the 128 experts
cost **zero RAM** on NVMe until a token routes to them, and each token resolves
exactly its top-k. Routing varies across tokens (the layer genuinely selects,
not pins a fixed pair).

## Numerical results (correctness)

Bit-identical to the certified path on both tiers:

- `ResidentExpertLayer(Ram).forward(x) == RealMoeLayer::forward_auto(x)` —
  exact equality (synthetic 4-expert layer).
- `ResidentExpertLayer(Disk).forward(x) == RealMoeLayer::forward_auto(x)` —
  exact equality (synthetic 6-expert layer **and** the real committed
  `full_mixtral.safetensors` layer 0, 6 seeds).
- NVMe tier == RAM tier for the 128-expert layer (exact equality).

Compatibility with MOE-FULL-6/7 is therefore direct: the residency layer is a
drop-in for the certified MoE block (same output), so the full-forward and
decode paths would compute identically if backed by it.

## What is NOT implemented

- No VRAM tier exercised, no quantised experts, no expert cache / prefetch.
- No GQA, no config-driven topology, no productive load-path integration, no
  fail-loud lift. Real MoE checkpoints still fail loud (MOE-2).
- Not wired into the full-forward graph or the decode loop (those still use the
  RAM-resident `RealMoeLayer`); the residency layer is validated as an
  equivalent provider.

## What remains (MOE-FULL-9)

- GQA (load-time K/V tile or graph repeat-kv) for real Mixtral-8x7B.
- VRAM expert tier + an LRU expert cache / prefetch for throughput.
- Productive integration: a Mixtral family adapter on the load path, an
  explicit opt-in fail-loud lift, config-driven topology, and backing the
  full-forward/decode MoE node with the residency provider.

## Tests

- `src/moe/residency.rs` — 6 unit tests: `ram_tier_matches_real_layer_bit_for_bit`,
  `disk_tier_matches_real_layer_bit_for_bit`, `disk_tier_keeps_experts_out_of_ram`,
  `ram_tier_keeps_experts_in_ram`, `only_top_k_experts_materialized_per_forward`,
  `forward_is_deterministic_across_tiers`.
- `tests/moe_residency_test.rs` — 4 integration tests:
  `residency_matches_real_fixture_layer_on_nvme` (real `full_mixtral` layer 0),
  `large_scenario_residency_keeps_ram_flat` (128 experts, prints the 385×
  saving), `large_scenario_nvme_matches_ram_tier`, `fail_loud_still_active`.

Local validation (real output, exit 0): full lib suite
`cargo test --lib --release -- --test-threads=1` → **762 passed / 0 failed /
1 ignored** (was 756; +6). Integration: residency 4/4. No regressions across
the MoE integration targets (full_forward 7/7, decode 5/5, decoder_layer 6/6,
graph_op 7/7).

## Files modified

* `src/moe/residency.rs` — new (`ResidentExpertLayer`, `ExpertTier`,
  `ResidencyInfo`, on-demand tier resolution + 6 unit tests).
* `src/moe/mod.rs` — `pub mod residency;`.
* `tests/moe_residency_test.rs` — new (4 integration tests).
* `docs/HANDOFF_MOE_FULL_8.md` — this file.
* `docs/MOE_FULL_PATH_AUDIT.md` — MOE-FULL-8 marked DONE; remaining renamed
  MOE-FULL-9.
* `docs/MOE_OVERVIEW.md` — large-model residency mitigation noted; readiness
  table rows added for MOE-FULL-6/7/8.

No new fixtures, no model download. Reuses `SharedParam` / `disk_tier`
unchanged. No loader / runtime / Adapter Toolkit / WeightStore / CUDA / CLI
change. Fail-loud preserved.
