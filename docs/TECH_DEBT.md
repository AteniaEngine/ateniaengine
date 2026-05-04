# TECH DEBT — known pre-existing failures

Last reviewed: M8.7 prerequisite (commit after `e8b2ec3`).

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
