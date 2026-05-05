# TECH DEBT — known pre-existing failures

Last reviewed: tier-aware loader default flip (commit `afaa975`).

## `exec_gpu_segment` (APX 4.3 legacy) deprecation

**Status (commit after the M8.7 / `execute_inference` fix)**: a
runtime guard now skips `exec_gpu_segment` whenever any operand
in the planned segment is `Cuda(BF16)` or `Disk(_)`. The legacy
path is APX 4.x and assumes F32 GPU tensors — calling
`cuda_matmul_inplace` (the F32 kernel) on a BF16 device buffer
asserts inside `TensorGPU::to_cpu` (`tensor_gpu.rs:179`).

The guard lives in `src/amg/graph.rs::execute_single_inner` plus
the helper `Graph::segment_has_bf16_or_disk_operands`. With the
guard, the modern dispatch (`gpu/dispatch/hooks.rs::try_gpu_matmul`
+ M8.4c `cuda_matmul_bf16_inplace` + M8.7.0
`cuda_matmul_disk_streamed_bf16`) handles every M8 / M8.7
operand correctly and the killer demo's 13B forward stays at
~13.5 s for seq=4.

**Removal plan**: `exec_gpu_segment` is invoked from a single
call site (`src/amg/graph.rs::execute_single_inner`) and reads
`self.gpu_plan` populated by `apx4_3::GpuPlan`. Auditing the APX
4.x test surface (`tests/apx_4_*`) before removing the path
needs a dedicated cycle — for now the guard is sufficient and
the path is documented as **superseded** by the M6 / M8 / M8.7
modern dispatch chain. Open as deprecation candidate alongside
`ATENIA_TIER_AWARE_LOADER` in M9 / M10.

## M8.7.0 disk-streamed matmul not firing through the killer demo *(RESOLVED)*

**Resolution**: H3 confirmed via `[M8.7.0 DEBUG]` instrumentation —
`record_tape == true` was reaching the MatMul arm because
`graph.execute(...)` defaulted to `record_tape=true`. Fixed by
introducing `Graph::execute_inference` (`record_tape=false`)
and migrating the killer demo to it. The fix surfaced a
secondary panic in `exec_gpu_segment` (legacy F32 GPU path)
because `record_tape=false` re-enables that path against M8 BF16
operands; addressed by the new `segment_has_bf16_or_disk_operands`
guard documented under the *`exec_gpu_segment` deprecation*
section above.

Post-fix smoke (`ATENIA_M8_BF16_KERNEL=1 ATENIA_M8_7_ENABLED=1`,
13B Llama 2 Chat, seq=4):

    Tier plan: vram=78 tensors (6.46 GiB), ram=0, disk=325
    Forward: 13.50s
    BF16-resident matmuls (M8.4c): 281
    Disk-streamed matmuls (M8.7.0): 203
    Per-position argmax matches the M8 baseline.

**Original symptom (post commit `4e126f0`)**: with
`ATENIA_M8_7_ENABLED=1`
and 159 Disk-tier weights in the 13B tier plan,
`disk_streamed_matmul_count` stays at 0 across the full 4-token
forward of `examples/llama2_13b_demo.rs`. Other counters confirm
the rest of the pipeline is healthy: `vram_bf16_matmul_count = 80`
(M8.4c path firing for VRAM-resident weights),
`gpu_matmul_resident_count = 80` (mixed-residency dispatch live),
logits sane, argmax matches the M8 close baseline.

**Why it matters**: M8.7.0 was unit-tested with a synthetic
`SharedParam::Disk` tensor (drift gate 3.28e-3 vs ADR-004 0.5,
counter advances by 1 per dispatch — see
`tests/cuda::matmul::cuda_matmul_disk_streamed_tests`), but the
full-stack smoke through the demo is the first time the path was
exercised against a real `WeightStore` produced by the tier-aware
loader's Disk arm. The unit tests exercise the dispatch helper in
isolation; this issue is in the **routing** that decides whether
to call the helper.

**Hypotheses (to investigate next session)**:

- **H1** — `register_param_from_store` produces a `Tensor` whose
  `storage` does not preserve `TensorStorage::Disk(handle)`. The
  builder calls `p.to_tensor()` on the `SharedParam::Disk`; if
  that path materialises the handle into RAM/Cpu storage instead
  of forwarding it as `TensorStorage::Disk`, the M8.7.0 hook in
  `src/amg/graph.rs:3110-3168` never sees a Disk-storage `b`.
  This is the highest-probability hypothesis given the unit test
  passes with a hand-built `Tensor::from_disk`.
- **H2** — A pre-MatMul operator (Reshape, Transpose, etc.) in
  the Llama graph consumes the parameter `Tensor` and triggers
  `ensure_decoded`, materialising the Disk handle to F32 CPU
  before the MatMul arm runs. The dispatch then sees a regular
  `Cpu` operand and falls through to the legacy AVX2 path.
- **H3** — `record_tape == true` at execute time. The M8.7.0 hook
  is gated on `!record_tape` (inference only). If the demo's
  `graph.execute(...)` defaults to record_tape=true, the hook is
  silently skipped.
- **H4** — `gpu_can_run_matmul(m, k, n)` rejects the specific
  shapes of Disk-tier weights (e.g. K-dim or N-dim outside the
  gate) before reaching the M8.7.0 hook, **or** the hook is
  placed after a `return` from another MatMul arm that fires
  first.

**Recommended investigation order**:

1. Add a single `eprintln!` in the M8.7.0 hook arm reporting
   `(record_tape, b.storage.variant())` and run a one-token 13B
   smoke. Pins H3 and H1 in one shot.
2. If H3 confirmed, push the gate inside the dispatch helper or
   relax it — the helper itself is inference-friendly (no
   gradient capture).
3. If H1 confirmed, audit `SharedParam::Disk::to_tensor` and
   `register_param_from_store` for the storage forwarding.
4. Re-run the smoke; expect `disk_streamed_matmul_count` to
   advance to 159 (one per Disk-tier weight in the 13B forward).

**Validation gate when fixed**: 13B smoke with the demo and
`ATENIA_M8_BF16_KERNEL=1 ATENIA_M8_7_ENABLED=1` should report
`disk_streamed_matmul_count = 159`, total forward time below the
20.85s baseline (since CPU AVX2 matmul on 159 disk weights
dominates today), `cargo test --lib` 185/185 + the existing
M8.7.0 ignored CUDA test still passing.

## `ATENIA_TIER_AWARE_LOADER` deprecation

The opt-in flag from D74 of `HANDOFF_APX_V20_M6.md` was inverted at
commit `afaa975`: tier-aware is now the default and
`ATENIA_LEGACY_LOADER=1` is the new opt-out. The deprecated flag
`ATENIA_TIER_AWARE_LOADER` is still recognised — it becomes a no-op
because the path is default — and the pipeline emits a one-line
deprecation warning on stderr when it is set:

    [ATENIA] ATENIA_TIER_AWARE_LOADER is now the default and will be
    removed in a future version. Use ATENIA_LEGACY_LOADER=1 to opt
    out instead.

**Removal plan**: keep the warning-only stub through one more major
milestone (M9 or M10, whichever lands first) so external scripts
that still reference the flag have a chance to migrate. Then drop
the read entirely and remove the dedicated `if` block in
`src/nn/llama/pipeline.rs`. The opt-out variable
`ATENIA_LEGACY_LOADER` stays indefinitely — it preserves the pre-M6
CPU-resident path for hardware where the tier-aware loader cannot
be exercised (no CUDA, RAM-only setups). Adding a new entry here
when that grace period closes.

## `cargo test --tests` non-determinism

Running the full integration suite (`cargo test --tests`) produces a
**different** failure on each run, depending on which test binary
finishes first under cargo's parallel-runner. None of the failures
reproduce when the affected binary is run in isolation.

Verified by running on clean `e8b2ec3` (M8.7 prereq stashed) — the
breakage **pre-dates** the M8.7 work. The M-milestone regression
flow has masked it because every milestone gate uses
`cargo test --lib` plus targeted `cargo test --test <name>` invocations
rather than the full integration build.

| Test | Failure | Repro |
|---|---|---|
| `apx_4_16_qkv_backward_integration_test::test_qkv_backward_4_14_vs_4_16_match` | numerical mismatch `a=0.6 b=0.3 diff=0.3 tol=1e-6` | always (when the binary runs) |
| `apx_6_12_scheduler_test::apx_6_12_default_bias_is_none` | passes alone, fails under `cargo test --tests` parallel scheduling | `cargo test --tests` (not in isolation) |
| `cli_explain_smoke_test` | `assertion failed: output.status.success()` — assumes the `atenia` binary is on PATH | `cargo test --tests` under load |

The first one is now `#[ignore]`'d in `e8b2ec3` with a doc comment
linking back to this file. The other two are not yet annotated —
they're cross-binary parallel-scheduling problems and the right fix
is not to silence them but to remove the cross-binary state sharing.

## Probable root cause

Several integration tests mutate process-wide state via environment
variables (`ATENIA_APX_MODE`, `ATENIA_M8_BF16_KERNEL`, etc.) inside
`unsafe { std::env::set_var(...) }` blocks. Cargo runs separate test
binaries in **parallel processes**; within a single binary, tests
run on multiple threads inside the same process. Both layers can
race on `std::env`:

1. **Within a binary** — concurrent `set_var`/`remove_var` against
   reads (e.g. `pipeline.rs` loading `ATENIA_M8_BF16_KERNEL`). Tests
   that read the env at runtime see whatever the most-recently-running
   test left there.
2. **Across binaries** — separate processes don't share env, but they
   do share files, ports, and the PATH-installed `atenia` binary. The
   `cli_explain_smoke_test` failure is consistent with `atenia` not
   being installed (or being mid-rebuild) on the runner's PATH while
   another test binary is locking the `target/release/` artefact.

## Fix strategy (deferred — not in M8.7 scope)

1. Replace each `std::env::set_var` in the test binaries with a
   per-test override that doesn't touch process state. The pattern is
   either:
   - Pass the override as a function parameter rather than reading
     `std::env` at runtime; or
   - Use a test-scoped lock (like the `BF16_COUNTER_TEST_LOCK` already
     in `cuda::matmul::tests`) to serialise tests that *must* mutate
     the env.
2. Update `cli_explain_smoke_test` to either depend on the binary
   being built (via `cargo test --no-default-features --bin atenia`)
   or skip when the binary is unavailable.
3. Re-enable `apx_4_16_qkv_backward::test_qkv_backward_4_14_vs_4_16_match`
   after fixing the underlying APX 4.14 vs 4.16 backward gradient
   divergence (the two paths produce different values, not different
   precision — likely a fused-QKV backward implementation bug).

## Validation flow used today

Until the integration suite is repaired, the M-milestone validation
contract is:

```bash
cargo test --lib                                    # 183 tests, must be 0 failed
cargo test --test <relevant_test> [--ignored]       # per milestone
cargo install --path . --force --bin atenia         # smoke install
atenia probe                                        # safety gate
atenia run --mode c --model <path> --output json    # smoke generation
```

Adding `cargo test --tests` to that list requires the cleanup above.
