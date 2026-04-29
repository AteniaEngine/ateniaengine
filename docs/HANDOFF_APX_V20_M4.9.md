# Handoff — APX v20 M4.9 (Public CLI Demo, at M4.9 close)

**Status at handoff**: M4.9 closed. The v20 killer demo is
now reproducible via a single command on any machine with
the model checkpoint and the right hardware. After
`cargo install --path .` and an `huggingface-cli download`,
running `atenia run --mode c --model <path> --cache-dir <path>`
loads Llama 2 13B Chat, runs the warmup forward, forces a
50 % LRU spill, runs the post-spill forward, and prints a
`[PASS] ✓ argmax(pre-spill) == argmax(post-spill)
bit-exactly` block — in **6.9 minutes wall-clock** on the
dev box.

The boundary between "the killer demo is real" (M4.7) and
"anyone can reproduce it" (M4.9) is now crossed. The CLI
consumes the same engine APIs the test suite consumes — no
"demo-only" code paths that could drift. The reproduction
recipe is locked into `README.md` and `docs/CLI.md`; a
community member with a 32 GB / 8 GB box, an NVMe drive, and
the Llama 2 13B Chat checkpoint reaches the `[PASS] ✓` line
in well under 30 minutes from `git clone`.

**Last M4.9 commit**: `56152a3` (M4.9.f, docs + README +
drop legacy `hardware_probe` binary). M4.9 closes at this
commit; M5 (tokenizer + KV cache + token-by-token generation)
is the next active milestone.

**Empirical baseline — Mode C via the CLI** (the canonical
killer-demo path; same harness as M4.7.6.e Mode C, now via
`atenia run --mode c`):

```
=== Atenia v20 Killer Demo — Llama-family — Mode C ===
Phases:
  Build graph    ....   ~1 s
  Load weights   .... ~165 s   (~156 MB/s on NVMe)
  Warmup forward .... 200 s
  Force LRU spill ..  19 s     (866 / 1732 LRU entries spilled = 50.0 % exact)
  Post-spill fwd ..   23 s
Transparency contract:
  argmax(pre)  = 1, logit 4.7747
  argmax(post) = 1, logit 4.7747
  [PASS] ✓ argmax(pre-spill) == argmax(post-spill) bit-exactly
Total wall-clock: 6.9 minutes.
```

The argmax `id = 1, logit = 4.7747` is bit-exact identical
to:
- The M4.7.6.e Mode C test (the original empirical baseline,
  ~24 min on its first run).
- The M4.7.6.d Mode A baseline (pos 0 of seq=4).
- The M4.8.f Mode A re-run after the perf upgrade (5.38 min
  forward).
- This Mode C run via the public CLI (6.9 min total).

The transparency contract holds across the entire M4.7 /
M4.8 / M4.9 stack at 13 B parameter scale, on the same
hardware that runs Chrome and a code editor in the
background.

---

## What is ready

| Sub-phase | Commit | Summary |
|-----------|--------|---------|
| **M4.9.a — clap skeleton + probe migration** | `8655832` | New `src/bin/atenia.rs` rewrites the v13-era manual-argv binary into a clap-4 derive-based CLI with three subcommands: `probe`, `run`, `explain`. `probe` arm gated at runtime via `cfg!(feature = "hw-probe")` rather than `required-features` on the `[[bin]]` entry, so `cargo install --path .` produces a working `atenia` binary on hosts that cannot build wgpu (default-build users invoking `atenia probe` get an actionable "feature not enabled" message). `run` arm parses cleanly but each mode returns a not-yet-implemented stub. `explain` arm preserves the v13 `LearningContextSnapshot` + `SelfTrainer::explain_decision` semantics under clap's surface. Probe parity verified bit-exact vs the legacy `hardware_probe` binary on the dev box — only the timestamps differ. |
| **M4.9.b — `atenia_engine::demo` extraction** | `a18c64a` | New `src/demo/mod.rs` lifts the M4.7.6.e helpers (pressure probes, reactive-context factory, sharded BF16 load helper, argmax reduction, cache-dir helper) out of `tests/m4_7_6_e_llama2_13b_modes_b_c_test.rs` into a public module. New `demo` Cargo feature (default-on) gates the module so power users can build minimal libraries via `--no-default-features`. The test file refactored to consume the public module via a 25-line `build_and_load_13b` wrapper that preserves the pre-M4.9.b call sites; `cargo test --no-run --test m4_7_6_e_*` clean. Pure refactor — no engine logic changed. |
| **M4.9.c — `atenia run --mode a`** | `5851bec` | New `src/cli_run.rs` library module behind the `demo` feature. `DemoReport` schema (serde-serializable; same struct drives text and JSON renderers), `Heartbeat` helper for the live progress dots (~2 s flush on stderr, gated by `--no-progress`), `ram_soft_warning()` (warns below 28 GB RAM but does not abort), `cache_dir_disk_probe()` (100 MB sequential write benchmark, warns below 200 MB/s), text + JSON renderers, and `run_mode_a()` — the actual Mode A runner. Reproduces M4.8.f bit-exact: `argmax id=1, logit=4.7747` at pos 0, forward 278 s, total 7.5 min on the dev box. |
| **M4.9.d — `atenia run --mode c`** | `d554996` | `run_mode_c()` in `src/cli_run.rs`. Warmup forward at low pressure → captures `argmax(pre)`. Force `Graph::deep_degrade_with_lru(&cache_dir)` → spill 50 % of the LRU. Post-spill forward → captures `argmax(post)`. `bit_exact = (pre.id == post.id) && (pre.logit == post.logit)`. Exit code 0 on PASS, 3 on contract violation. Smoke-tested on 13B: argmax(pre) = argmax(post) = 1, logit 4.7747; 866 / 1732 entries spilled (50.0 % exact); subsequent dev-box reproductions land at 6.9 min total (warmup 200 s + spill 19 s + post-spill 23 s) as caches warm up. |
| **M4.9.e — `atenia run --mode b`** | `6bed661` | `run_mode_b()` in `src/cli_run.rs`. Attaches the high-pressure reactive context (one-shot probes from M4.7.6.e), wraps `graph.execute(...)` in `std::panic::catch_unwind` with `AssertUnwindSafe`. The first guard checkpoint fires `Degrade → DeepDegrade`; a downstream activation hits the documented M4.7.5.e gap and panics; `catch_unwind` absorbs it; the verbatim panic message prints under "Forward absorbed:". Counters validated: 4 DeepDegrade events fired, 26 031.7 MB spilled. Exit 0 if `dd_events > 0 && spilled_bytes > 0`, 1 otherwise. Wall-clock 8.1 min on the dev box. |
| **M4.9.f — docs + release polish** | `56152a3` | `docs/CLI.md` documents every subcommand, every flag, every exit code, and the JSON schema (stable across M4.9.+, breaking changes are major bumps). `README.md` ships the "Reproduce the killer demo in one command" section near the top with the exact shell sequence and expected output. Legacy `src/bin/hardware_probe.rs` removed; `[[bin]] hardware_probe` entry stripped from `Cargo.toml` (parity verified at .a, the `atenia probe` subcommand is now the canonical surface). Test count (~1200 / ~370 files) preserved. |

Every commit is on `main` and pushed to `origin/main`. The
six sub-phases each closed with their own commit; no
sub-phase overlapped another's scope.

---

## Architectural decisions locked

Treat as invariants. Future work extends rather than
re-litigates. The M4.5 / M4.6 / M4.7 / M4.8 invariants
(decisions 1–48) remain in force; the list below adds M4.9's
contributions on top.

49. **`clap` 4.x with `derive` + `env` features is the CLI
    framework of record**. Single dep added; transitive
    cost is small (~5 crates). Spec lives in attribute-
    annotated `struct` definitions in `src/bin/atenia.rs`;
    help-text + validation + completions are all generated
    from those structs. Future CLI surface extensions add
    `Subcommand` / `Args` variants there. Manual `std::env::args`
    parsing is no longer the canonical pattern in this
    codebase.

50. **`atenia_engine::demo` is the public reproduction-
    surface module**. Pressure probes, reactive-context
    factory, sharded BF16 load helper, `argmax_row`
    reduction, `cache_dir_for` helper. Behind the
    default-on `demo` Cargo feature. Both the CLI runner
    and the M4.7.6.e integration tests consume the same
    primitives; no test-only / demo-only forks. New
    end-to-end flows that need any of these primitives
    import from `crate::demo` rather than re-implementing
    inline.

51. **Heartbeat dots go to stderr; the final report goes to
    stdout**. The `--output json` path keeps stdout clean
    so `atenia run --output json | jq …` works without
    further redirection. The `Heartbeat` struct
    (`src/cli_run.rs`) spawns a thread that writes a `.`
    every 2 s, flushes stderr, and stops cleanly via
    `AtomicBool` + `JoinHandle`. `--no-progress` skips the
    spawn entirely (CI-friendly). Future progress-UI work
    (indicatif, etc.) lives behind an opt-in flag and does
    not change this stream split.

52. **The `hw-probe` Cargo feature stays opt-in**. Pulling
    wgpu (~50 transitive crates) on a default `cargo
    install --path .` is unacceptable for a
    one-command-reproduction target. The `atenia probe`
    arm is gated at runtime via `cfg!(feature =
    "hw-probe")`; default builds emit an actionable
    rebuild-with-feature message and exit 2. The `[[bin]]
    atenia` entry in `Cargo.toml` does **not** have
    `required-features = ["hw-probe"]`; that would prevent
    default `cargo install` from producing a working
    binary on hosts without wgpu's prerequisites. New
    feature gates that pull heavy deps (a future ROCm /
    Metal probe path) follow this pattern.

53. **The `demo` Cargo feature is default-on**. Three
    consumers need it (the `atenia` binary's `run` arm, the
    M4.7.6.e integration tests, future M5 generation
    tooling). Default-on means `cargo test` / `cargo
    install` work without any explicit `--features` flag.
    Power users wanting a minimal library surface get
    `--no-default-features` and the engine's core public
    API (`crate::amg`, `crate::tensor`, `crate::nn::llama`,
    `crate::v17::loader::*`) compiles without it.

54. **The cache directory auto-cleans only when the CLI
    created it**. If the operator passed `--cache-dir
    <PATH>`, the CLI **does not** delete the directory at
    end of run — the operator owns it. If the CLI fell back
    to `cache_dir_for(label)` (UUID-suffixed temp dir under
    `disk_tier::default_cache_dir()`), the CLI removes the
    directory after the report renders. This rule prevents
    surprising operators with a directory they explicitly
    chose disappearing under them, while still keeping
    one-off runs from leaving 13 GB of detritus in the
    scratch space.

55. **No new env vars introduced in M4.9**. The existing
    set (`ATENIA_LLAMA2_13B_DIR`, `ATENIA_DISK_TIER_DIR`,
    `ATENIA_FORCE_SPILL`, `ATENIA_APX_MODE`,
    `ATENIA_BF16_PRECISION_FLOOR`,
    `ATENIA_TRACE`) covers every override the CLI flags
    need. CLI flags are the canonical override surface;
    env vars are the fallback for shell-scripted
    invocations. New CLI features that need configuration
    add a flag, optionally with `env = "..."` clap
    attribute pointing at one of the existing env vars —
    not a new one.

56. **Exit codes are semantically distinct**. Code 0 =
    success. Code 1 = runtime error (load OOM, kernel
    failure, `deep_degrade_with_lru` failure,
    `ensure_cpu` on output failure). Code 2 =
    configuration / argument error (model dir missing,
    `--features hw-probe` not built, bad flag value).
    Code 3 = mathematical contract violation (Mode C:
    `argmax(pre) != argmax(post)`). CI scripts can
    `case $?` on these to distinguish "demo crashed"
    from "demo proved a contract violation" — the third
    case is the falsifier of the v20 thesis and gets its
    own code.

57. **The CLI's JSON schema is stable across M4.9.+**.
    Adding fields is non-breaking; renaming or removing
    fields is a major bump. Documented in `docs/CLI.md`
    with a JSON-with-Comments shape per mode. M4.9 ships
    two shapes: `DemoReport` (Mode A / Mode C) and
    `ModeBReport` (Mode B; separate because it has no
    argmax / contract). New modes / new fields land
    behind feature flags or via additive evolution; a
    consumer's `jq` query against the M4.9.+ schema must
    keep returning a value across point releases.

---

## Empirical validation results

### Mode C — the canonical killer demo (ADR-004 close criterion)

Hardware: RTX 4070 Laptop, 8 GB VRAM, 32 GB RAM DDR5-5600
dual-channel, NVMe SN770 spill cache. Llama 2 13B Chat at
seq=1, BOS prompt.

| Phase | Wall-clock | Notes |
|-------|-----------:|-------|
| Build graph | ~1 s | 363 parameter nodes |
| Load weights | ~165 s | ~156 MB/s on NVMe (M4.7.1 sharded loader, BF16 storage active) |
| Warmup forward | 200 s | low-pressure context attached, argmax(pre) captured |
| Force LRU spill | 19 s | 866 / 1732 LRU entries migrated (50.0 % exact) at ~150 MB/s, 13 GB written to NVMe |
| Post-spill forward | 23 s | lazy-restore via `ensure_cpu` Disk-arm, argmax(post) captured |
| **Total** | **6.9 minutes** | argmax(pre) == argmax(post) == 1, logit 4.7747 — `[PASS] ✓` |

The argmax `id = 1, logit = 4.7747` matches the M4.7.6.e
Mode C original baseline bit-exactly. The 6.9-minute total
is faster than the original M4.7.6.e first-run
(~24 minutes) because:
- M4.8 perf stack: parallel matmul + matrixmultiply + SIMD
  decode is ~3.5× faster on the warmup forward.
- Cache state: subsequent runs land warmer than a
  cold-build first run.

### Mode A — the no-spill baseline (M4.8.f re-validation)

Same harness as M4.7.6.d; reproduces `argmax id=1, logit=4.7747`
at pos 0 of seq=4 with the M4.8.a–.e perf stack live.

| Phase | Wall-clock |
|-------|-----------:|
| Build | 1.91 s |
| Load | 167.84 s @ 155 MB/s |
| Forward (seq=4) | 278.20 s |
| **Total** | **7.5 minutes** |

Per-position argmax and logit values bit-exact identical to
M4.7.6.d / M4.7.6.e Mode A.

### Mode B — autonomous trigger validation

Numbers match the M4.7.6.e Mode B test exactly. The CLI
runner is a thin clap+report wrapper around the test's
`make_context(cache_dir, /*high_pressure=*/ true)` +
`catch_unwind` core.

| Metric | Value |
|--------|------:|
| DeepDegrade events fired | 4 |
| Bytes spilled to disk | 26 031.7 MB |
| catch_unwind result | panic absorbed (M4.7.5.e gap, expected) |
| Total wall-clock | 8.1 min |
| Exit code | 0 (trigger plumbing OK) |

The verbatim panic message (`Tensor::as_cpu_slice called on
a Disk-resident tensor. Call ensure_cpu() first to materialize
data back in host memory.`) prints under "Forward absorbed:"
in the text output and as `panic_message` in the JSON output
— the documented M4.7.5.e gap is **visible**, not hidden.

### Probe parity vs the legacy `hardware_probe` binary

`atenia probe --output json` vs the pre-M4.9 `hardware_probe
--output json` produce bit-identical output on the dev box,
modulo the timestamp fields (`probed_at_unix_secs` and
`probed_at_iso8601`) which differ across runs by definition.
All hardware / system / GPU / driver fields match.

```
$ diff <(atenia probe --output json) <(hardware_probe --output json)
4,5c4,5
<   "probed_at_unix_secs": 1777497142,
<   "probed_at_iso8601": "2026-04-29T21:12:22Z",
---
>   "probed_at_unix_secs": 1777497149,
>   "probed_at_iso8601": "2026-04-29T21:12:29Z",
```

This is the M4.9.a close criterion. The legacy binary was
removed in M4.9.f.

---

## Hardware prerequisites — what the operator needs

Documented in `README.md` and `docs/CLI.md`; reproduced here
for the handoff record.

| Component | Requirement | Soft / Hard |
|-----------|-------------|:-----------:|
| **CPU** | x86-64 with AVX2 + FMA | Hard at runtime — `is_x86_feature_detected!("avx2")` is consulted; if absent the dispatcher falls back to `scalar_matmul` and the demo wall-clock balloons by 50–100×. |
| **RAM** | 32 GB recommended; 28 GB minimum | Soft — CLI warns below 28 GB, does not abort. Mode A needs ~26 GB resident BF16 + ~2 GB activations + headroom. |
| **Disk for spill cache** (`--cache-dir`) | NVMe at ≥ 200 MB/s sustained write | Soft — CLI runs a 100 MB benchmark at startup, warns below 200 MB/s. Below ~50 MB/s the post-spill forward becomes unusable (~30 min). |
| **Disk space** | ~30 GB for the model + ~14 GB for the spill cache | Hard — `cache_dir_disk_probe` does not check free space directly; load / spill will OOM the disk if exhausted. |
| **GPU** | Not required for the killer demo | The 13 B forward is CPU-bound per HANDOFF M4.7 decisions 34–35 (64 MiB pool ceiling). 1B-class models do exercise the GPU but are not the demo. |
| **OS** | Windows or Linux | macOS not tested in M4.9; Apple Silicon is on the v24 roadmap. Library compiles on macOS but the demo wall-clock has not been validated. |
| **Rust** | stable, 2024 edition | The build script auto-detects CUDA + MSVC; `CUDA_PATH` / `MSVC_TOOLS_PATH` env vars override. |
| **Network** | Required once for `huggingface-cli download` | The model is ~26 GB; one-shot. The CLI itself is offline-only. |

---

## Gaps explicitly closed in M4.9

- **No public reproduction surface for the killer demo**.
  M4.9 ships `atenia run --mode {a|b|c}`. One command to
  reproduce; one command to validate the transparency
  contract; one command to validate the autonomous trigger.
- **Test-only spill orchestration**. M4.7.6.e's pressure
  probes and `make_context` were trapped inside the test
  file; M4.9.b lifts them into `atenia_engine::demo` so the
  CLI and tests share one source of truth.
- **Manual argv parsing in CLI binaries**. The legacy
  `hardware_probe` and `atenia explain` (v13) binaries
  used hand-rolled `std::env::args()` parsers. M4.9.a
  consolidates onto clap; the legacy binary is dropped at
  M4.9.f.
- **The "is this demo legitimate?" question**. The CLI's
  `[PASS] ✓` block on `argmax(pre) == argmax(post)
  bit-exactly` is the operator-facing falsifier of the v20
  thesis. Anyone with the hardware can run the command and
  check the output — no need to read source code or trust
  the dev's claims.
- **JSON output for scripted reproduction**. Stable
  schema, documented in `docs/CLI.md`, with separate
  shapes per mode. CI scripts can grep / `jq` on
  predictable fields (`contract.bit_exact`,
  `deep_degrade_events`, `phases.*_seconds`).
- **Hardware prerequisites are now machine-checked at
  startup**. `ram_soft_warning()` + `cache_dir_disk_probe()`
  surface the most common reproduction failures (too
  little RAM, slow cache disk) **before** the operator
  watches a 30-minute load fail with OOM.
- **Doc bloat from the legacy probe binary**. With
  `hardware_probe` removed, the `docs/HARDWARE_PROBE.md`
  reference now points at `atenia probe` consistently;
  `docs/CLI.md` is the single source of truth for the
  user-facing surface.

---

## Gaps explicitly NOT closed — scope deferred

The M4.5 / M4.6 / M4.7 / M4.8 deferred-scope lists remain in
force. M4.9 added these explicit deferrals to M5+:

- **Token-by-token generation**. The CLI takes float-encoded
  token IDs (`[1.0]` for seq=1; `[1.0, 100.0, 200.0, 300.0]`
  for seq=4). M5 lands the tokenizer integration; M5+ adds
  KV cache and `atenia generate --prompt "<text>"
  --max-tokens N` (or a fourth `atenia run` mode).

- **Auto-download via `atenia run --model auto`**. The CLI
  currently prints a friendly `huggingface-cli download …`
  command when the model is missing. Wiring HF download
  via the `hf-hub` Rust crate is out of scope (drags
  network IO, auth tokens, partial-resume logic) and would
  complicate the demo's single-binary install story. The
  separate `huggingface-cli` step is the canonical path.

- **TUI / `ratatui` interface**. The heartbeat dots are
  the M4.9 progress UX. A real TUI with progress bars,
  per-phase ETAs, and live counters is overkill for a demo
  runner; the M4.7-era observation that "the killer demo
  is one number, not a UI" applies.

- **Telemetry / opt-in usage stats**. Explicitly out — the
  demo is local. No metrics phone home; no version-check
  HTTP requests. Future operational tooling lands as a
  separate `--telemetry-endpoint` flag if it lands at all.

- **Self-update**. `cargo install` is the install path.
  Future versions ship as a new `cargo install --path .`
  on a fresh git checkout.

- **macOS reproduction validation**. The library compiles
  on macOS (Rust + Cargo are the only build prerequisites
  outside CUDA, which macOS does not have anyway). The
  demo wall-clock has not been validated; v24 (Apple
  Silicon support) is the natural milestone for that.

- **Phi 3.5 mini support in `atenia run`**. The CLI's
  `--model` flag is generic — once the engine supports
  Phi's longrope + fused projections (M4.6.2), the CLI
  will run it without code changes. M4.9 ships against
  the M4.6 family + Llama 2 13 B set.

---

## Observations from the M4.9 sprint

Recorded so the in-flight decisions are not lost.

- **Heartbeat dots beat indicatif at M4.9 scope**. A real
  progress bar implies an ETA, and the M4.7.6.d / M4.8.f /
  M4.9.d wall-clocks vary by ±20 % run-to-run depending on
  cache state — an ETA would be misleading more often than
  it would be useful. Heartbeat dots convey "the process
  is alive" without overclaiming. Future polish work that
  lands indicatif should gate it behind `--progress=bar` or
  similar so the dots stay the default.

- **`catch_unwind` + `AssertUnwindSafe` was the right
  choice for Mode B**. The alternative — closing the
  M4.7.5.e activation-arm gap entirely — is M5+ scope
  (substantial work) and would have blocked M4.9 for
  weeks. `catch_unwind` absorbs the panic, the verbatim
  message prints under "Forward absorbed:", and the
  trigger-plumbing contract still holds. The user-facing
  Mode B output is **honest** about what happened
  (`[PASS] ✓ trigger fired`, "panic absorbed"); we did
  not paper over the gap. When M5+ closes the gap, Mode B
  flips from "trigger validated, forward absorbed" to
  "trigger validated, forward completes" with no CLI
  surface change.

- **The "feature not enabled" message is friendlier than
  `required-features`**. clap's `required-features =
  ["hw-probe"]` would refuse to install the binary on
  hosts that cannot build wgpu. The runtime gate emits a
  clear rebuild-instructions message and exits 2 — the
  user knows exactly what to do, the install succeeded,
  and the rest of the CLI works. New optional features
  follow this pattern.

- **The legacy `hardware_probe` parity check was the right
  belt-and-suspenders**. M4.9.a ran `diff <(atenia probe
  --output json) <(hardware_probe --output json)` and
  verified bit-equal output (modulo timestamps) before
  M4.9.f dropped the legacy binary. Anyone updating any
  CLI surface in the future should keep an analogous parity
  check during the migration window.

- **The dev-box wall-clock got faster across reproductions
  as caches warmed**. M4.9.d's first run landed at 12.4 min
  total; subsequent runs on a clean PowerShell shell with
  no other workloads landed at 6.9 min total. Both are
  recorded — 12.4 min in the commit message of the close
  run, 6.9 min in the README and ROADMAP as the canonical
  dev-box reproduction number. Operators on different
  hardware should expect 1.5–3× variance vs the 6.9 min
  figure.

- **No new env vars** is a stronger constraint than it
  sounds. Resisting the urge to add `ATENIA_VERBOSE`,
  `ATENIA_NO_PROGRESS`, `ATENIA_OUTPUT_FORMAT`, etc. and
  instead routing every override through CLI flags keeps
  the operator's mental model small (one config surface,
  not two). The five existing env vars all predate M4.9
  and are documented as fallbacks for shell scripting; new
  configurability lands as flags first.

- **The investigation-previa pattern survived M4.9
  unchanged**. M4.9.a (clap skeleton) opened with a 1500-
  word investigation report; the 6 sub-step plan came out
  of that report and held without revision through .b–.f.
  Two architectural surprises were caught in the report
  before any code landed: (1) the legacy `atenia explain`
  binary needed preservation under clap rather than
  outright removal; (2) `serde` needed to lift from
  optional (gated under `hw-probe`) to mandatory because
  `--output json` is on the default-build hot path. Both
  are visible in the M4.9.a / M4.9.c commit messages.

---

## How to resume on M5

1. **Read this file and HANDOFF M4.7 / M4.8 in order**.
   The M4.7 invariants (decisions 1–39) and M4.8
   invariants (40–48) are still in force; M4.9's are
   layered on top. Pay special attention to **decisions
   49–57** — they bound the design space for M5.

2. **Confirm M4.9 still works on a clean checkout**:
   ```
   atenia probe                                    # exit 2 expected without hw-probe
   atenia probe --features hw-probe                # rebuild + run
   atenia run --mode a --model <PATH> --seq 4      # ~5.4 min, argmax id=1 logit=4.7747
   atenia run --mode c --model <PATH> --cache-dir <NVMe-PATH>  # ~6.9 min, [PASS] ✓
   ```
   The exit codes (0 / 2 / 0 / 0) are the close criterion; any drift surfaces an M4.9.+ regression.

3. **For M5 specifically — tokenizer + KV cache + token-
   by-token generation**:
   - Add `tokenizers = "0.20"` (or `sentencepiece` if Llama
     2's BPE wants the canonical sentencepiece-rs surface)
     as a dep.
   - Wire a `Tokenizer` struct in `crate::nn::llama` (or a
     new `crate::tokenizer` module).
   - The CLI gets a fourth `Mode::G` (generation) or a new
     `atenia generate` subcommand. The killer-demo modes
     A / B / C stay; generation lands alongside.
   - KV cache is the harder piece — needs to integrate
     with the spill / restore primitives so long contexts
     spill cleanly. M4.7's `migrate_selected_cpu_to_disk`
     extends naturally; the new wrinkle is per-token K and
     V tensors that grow incrementally.

4. **For the 13 B GPU acceleration sub-milestone**
   (deferred from M4.7):
   - Non-pooled `cuda_matmul` variant — direct `cudaMalloc`
     per call, bypasses `apx4_12::DEFAULT_BLOCK_SIZE`.
     Cuda runtime may need tuning to avoid alloc thrash.
   - Reactivate `apx4::gpu_context::gpu_available()` to
     consult the real CUDA detection. Lands together with
     the non-pooled variant or rejected tensors corrupt.
   - Close the `ensure_cpu` activation-arm gap (M4.7.5.e).
     Mode B flips from `catch_unwind`-absorbed to clean
     forward; the CLI's panic-message field becomes
     `null` on Mode B success.

5. **For the column-partitioning matrixmultiply
   follow-up** (deferred from M4.8):
   - `rayon::scope` that splits N into column blocks per
     thread; each thread calls `matrixmultiply::sgemm` on
     its slice via raw pointers (unsafe disjoint-slice
     pattern). The bench harness already has the M=1
     numbers needed to validate the speedup.

6. **For the decode-cache pinning follow-up** (deferred
   from M4.8.c): an evictable `Option<Vec<f32>>` alongside
   `TensorStorage::CpuBf16` so the F32 view is reused
   per-tensor without doubling permanent RAM. Spill /
   restore needs to drop the cache; first-call restoration
   cost is unchanged. Substantial work — budget carefully
   and pair with the M5 KV-cache work (both touch the
   storage variant set).

7. **Always honour the vendor-agnostic constraint**
   (decision 46). Every BLAS / SIMD / GPU upgrade must
   work on Intel **and** AMD x86-64 with comparable
   performance, and must keep NEON (ARM, v24) reachable
   without reshaping the public interfaces. The M5 KV
   cache primitives sit on the same Tensor / TensorStorage
   surface that M4.7.4.a's `DiskDtype` extends; new
   variants follow that pattern.

8. **The reproduction recipe in `README.md` is part of
   the test suite, not just docs**. Any change to the
   `atenia run` surface must keep the recipe working. If
   a new flag is added, document it in `docs/CLI.md` and
   leave the canonical `atenia run --mode c --model
   <path> --cache-dir <path>` invocation working without
   the new flag (defaults).
