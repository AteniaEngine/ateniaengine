//! `atenia` — public single-binary CLI for Atenia Engine.
//!
//! M4.9.a establishes the clap-based skeleton with three
//! subcommands:
//!
//!   - **`probe`** — cross-vendor hardware enumeration. Migrates
//!     the pre-existing `hardware_probe` binary's logic into a
//!     subcommand. Behind the `hw-probe` Cargo feature; default
//!     builds emit a friendly "feature not enabled" message.
//!
//!   - **`run`** — the killer-demo runner (Modes A / B / C).
//!     M4.9.a parses cleanly but each mode returns a not-yet-
//!     implemented message; M4.9.c–.e fill in Mode A, Mode C,
//!     and Mode B respectively.
//!
//!   - **`explain`** — pre-existing v13 self-trainer narrative
//!     explainer, preserved to keep its API stable. Wraps the
//!     `LearningContextSnapshot` + `SelfTrainer::explain_decision`
//!     path under a clap surface that matches the previous
//!     manual-argv contract (`--gpu-available=...`,
//!     `--vram-band=...`, `--ram-band=...`).
//!
//! Build:
//! ```powershell
//! # Default build — `run` and `explain` work, `probe` reports
//! # "feature not enabled".
//! cargo build --release --bin atenia
//!
//! # Full build — `probe` operational.
//! cargo build --release --bin atenia --features hw-probe
//! ```
//!
//! Usage:
//! ```text
//! atenia probe   [--output text|json]
//! atenia run     --model <PATH> --mode {a|b|c} [--seq N]
//!                [--output text|json] [--cache-dir PATH] [--no-progress]
//! atenia explain --gpu-available <true|false>
//!                --vram-band <0..3>
//!                --ram-band <0..3>
//! ```
//!
//! The `probe` arm is feature-gated at compile time via
//! `#[cfg(feature = "hw-probe")]` rather than `required-features`
//! on the `[[bin]]` entry so a default `cargo install --path .`
//! still produces a working `atenia` binary on hosts that
//! cannot build wgpu (the heavy dep behind `hw-probe`).
//! Default-build users invoking `atenia probe` get a friendly
//! message pointing them at the rebuild flag.
//!
//! Exit codes shared by every subcommand:
//!
//! | Code | Meaning                                                        |
//! |------|----------------------------------------------------------------|
//! | 0    | Success                                                        |
//! | 1    | Runtime error during execution (load OOM, kernel failure)      |
//! | 2    | Argument / configuration error (bad path, missing feature, …) |
//! | 3    | Mathematical contract violation (Mode C argmax pre != post)   |
//!
//! The legacy `hardware_probe` binary stays in place for one
//! milestone with a deprecation notice; M4.9.f drops it.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use atenia_engine::v13::learning_narrative::build_narrative;
use atenia_engine::v13::learning_snapshot::LearningContextSnapshot;
use atenia_engine::v13::self_trainer::SelfTrainer;

/// `atenia` — beyond-VRAM LLM runtime CLI.
///
/// Entry point for the public reproduction surface of the v20
/// killer demo. The two new subcommands (`probe`, `run`) are
/// intentionally minimal; M4.9 ships the demo reproduction
/// tooling, not a general-purpose inference toolkit. The
/// legacy `explain` subcommand is preserved.
#[derive(Parser)]
#[command(
    name = "atenia",
    version,
    about = "Atenia Engine — hardware-adaptive LLM runtime",
    long_about = "\
Atenia Engine is a hardware-adaptive AI runtime written from \
scratch in Rust. The `atenia` CLI exposes the public reproduction \
surface for the v20 killer demo: Llama 2 13B Chat running \
end-to-end on dev-class hardware (8 GB VRAM + 32 GB RAM) with \
the M3-e reaction loop mediating VRAM ↔ RAM ↔ disk offload.

Subcommands:
  probe      Detect hardware capabilities (requires --features hw-probe)
  run        Run the tri-mode killer demo on a checkpoint
  explain    v13 self-trainer narrative explainer (legacy)

Run `atenia <subcommand> --help` for details."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Detect hardware capabilities (CPU, GPU, RAM).
    ///
    /// Cross-vendor GPU enumeration via wgpu, with NVIDIA
    /// augmentation via NVML when present. Requires the binary
    /// to be built with `--features hw-probe`; default builds
    /// print a friendly "feature not enabled" message and
    /// exit 2.
    Probe(ProbeArgs),

    /// Run the Llama 2 13B Chat killer demo (Modes A / B / C).
    ///
    /// Mode A — clean RAM, no spill, no LRU policy. Baseline
    /// run; produces argmax + logit on the canonical input.
    ///
    /// Mode B — autonomous LRU spill triggered by simulated
    /// memory pressure. Validates the M3-e reaction-loop
    /// trigger plumbing; the forward itself is `catch_unwind`-
    /// absorbed per the M4.7.5.e activation-arm gap (M5+
    /// follow-up).
    ///
    /// Mode C — forced 50 % LRU spill before the post-spill
    /// forward. The canonical *momento guau* path: argmax
    /// must be bit-exactly identical to the pre-spill warmup,
    /// proving the spill + lazy-restore cycle is
    /// mathematically transparent at 13 B parameter scale.
    Run(RunArgs),

    /// v13 self-trainer narrative explainer (legacy).
    ///
    /// Pre-existing API preserved across the M4.9 CLI
    /// migration. Given a `LearningContextSnapshot` (GPU
    /// availability + VRAM/RAM bands), prints the trainer's
    /// learned-decision narrative for that context. Useful
    /// for inspecting the v13 hybrid-execution-engine
    /// scaffolding; not part of the v20 killer-demo surface.
    Explain(ExplainArgs),

    /// **M5.e** — generate text greedily from a Llama-family
    /// checkpoint, streaming each token to stdout as it
    /// lands.
    ///
    /// End-to-end glue around the M5 stack (M5.a tokenizer,
    /// M5.b KV cache, M5.c.2 cache-aware attention + Arc-
    /// shared weights, M5.d.a generation loop, M5.d.b
    /// `GenerationPipeline`). Loads the checkpoint, applies
    /// the model's chat template (unless `--no-chat-template`),
    /// runs greedy decoding until EOS or `--max-tokens`,
    /// prints stats on stderr.
    ///
    /// ```text
    /// atenia generate --prompt "Hello, how are you?" \
    ///     --model models/llama-2-13b-chat \
    ///     --max-tokens 100
    /// ```
    Generate(GenerateArgs),

    /// **Adapter Toolkit v2** — parse a declarative adapter DSL
    /// file (`.yaml` / `.json`), validate it, build the v2
    /// adapter, and print a summary. Does **not** run generation.
    ///
    /// ```text
    /// atenia load config/adapters/llama.yaml
    /// ```
    Load(AdapterFileArgs),

    /// **Adapter Toolkit v2** — like `load`, but prints the
    /// verbose report (v1 capabilities + GGUF→HF tensor-name
    /// sample) and any validation warnings.
    Debug(AdapterFileArgs),

    /// **Adapter Toolkit v2** — auto-detect a model directory
    /// (`config.json` or `*.gguf`) and emit a valid adapter DSL
    /// (YAML) that `atenia load` can consume directly.
    ///
    /// ```text
    /// atenia inspect ./models/llama-3.2-1b-instruct
    /// ```
    Inspect(ModelDirArgs),
}

#[derive(clap::Args)]
struct ProbeArgs {
    /// Output format.
    ///
    /// `text` — human-readable banner. `json` — machine-
    /// readable schema (same shape as the legacy
    /// `hardware_probe --output json`, suitable for `jq`
    /// queries).
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    output: OutputFormat,
}

#[derive(clap::Args)]
struct RunArgs {
    /// Path to the model directory.
    ///
    /// Must contain `config.json`, the safetensors files
    /// (single-file or sharded with
    /// `model.safetensors.index.json`), and tokenizer files.
    /// The CLI does not download the model; see the friendly
    /// error message printed when the path is missing.
    ///
    /// If omitted, falls back to the `ATENIA_LLAMA2_13B_DIR`
    /// environment variable.
    #[arg(long, env = "ATENIA_LLAMA2_13B_DIR")]
    model: PathBuf,

    /// Execution mode.
    ///
    /// `a` — clean RAM baseline.
    /// `b` — autonomous LRU spill trigger validation.
    /// `c` — forced 50 % LRU spill (the canonical *momento
    ///       guau* path).
    #[arg(long, value_enum)]
    mode: Mode,

    /// Sequence length.
    ///
    /// Mode A defaults to 4 (the original M4.7.6.d harness's
    /// canonical input). Modes B and C internally use 1 per
    /// the M4.7.6.e wall-clock budget; overriding here is
    /// permitted but the contracts (B: trigger fires; C:
    /// argmax bit-exact) hold at any seq > 0.
    #[arg(long, default_value_t = 4)]
    seq: usize,

    /// Output format. `text` for humans; `json` for scripted
    /// reproduction (stable schema across M4.9.+).
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    output: OutputFormat,

    /// Disk-tier directory for spill cache (Modes B and C).
    ///
    /// Defaults to the `ATENIA_DISK_TIER_DIR` environment
    /// variable, then to a platform-specific scratch
    /// directory. **Should be on an NVMe drive**: Mode C
    /// writes ~13 GB of BF16 spill at sequential bandwidth.
    /// USB HDDs and slow SD cards will make the post-spill
    /// forward unusable. M4.9.c will add a quick disk-
    /// throughput probe at startup that warns when the
    /// chosen path is below the 200 MB/s practical floor.
    #[arg(long, env = "ATENIA_DISK_TIER_DIR")]
    cache_dir: Option<PathBuf>,

    /// Suppress the live heartbeat dots during long phases.
    ///
    /// Set this when piping the output into a script or CI
    /// log that does not handle live-updating stdout
    /// gracefully. The phase durations and the final report
    /// still print at completion.
    #[arg(long)]
    no_progress: bool,
}

#[derive(clap::Args)]
struct GenerateArgs {
    /// Input prompt. Wrapped in the model's chat template
    /// before encoding (see `--no-chat-template` to bypass).
    #[arg(long)]
    prompt: String,

    /// Path to the model directory (config.json + tokenizer
    /// files + safetensors). Falls back to
    /// `ATENIA_LLAMA2_13B_DIR` for compatibility with the
    /// `run` subcommand's environment.
    #[arg(long, env = "ATENIA_LLAMA2_13B_DIR")]
    model: PathBuf,

    /// Maximum number of new tokens to generate. EOS halts
    /// earlier when emitted.
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,

    /// Output format. `text` streams tokens to stdout and
    /// prints stats on stderr; `json` accumulates and emits
    /// a single structured report on stdout at end-of-run.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    output: OutputFormat,

    /// Disk-tier scratch dir for spill (reserved for M6+;
    /// currently unused by `generate` but kept for API
    /// stability).
    #[arg(long, env = "ATENIA_DISK_TIER_DIR")]
    cache_dir: Option<PathBuf>,

    /// Suppress the heartbeat dots that print during model
    /// load. Stats line still prints when generation
    /// finishes.
    #[arg(long)]
    no_progress: bool,

    /// Skip `apply_chat_template`; feed `--prompt` to the
    /// model verbatim. Useful for completion-style prompts
    /// and for debugging tokenizer / template issues.
    #[arg(long)]
    no_chat_template: bool,
}

#[derive(clap::Args)]
struct ExplainArgs {
    /// Hypothetical "GPU is available" flag for the learning
    /// snapshot. Determines which branch of the v13 self-
    /// trainer's decision table is queried.
    #[arg(long)]
    gpu_available: bool,

    /// VRAM band, 0..=3. Coarse classification used by the
    /// v13 trainer to discretise memory state.
    #[arg(long, value_parser = clap::value_parser!(u8).range(0..=3))]
    vram_band: u8,

    /// RAM band, 0..=3. Coarse classification used by the
    /// v13 trainer to discretise memory state.
    #[arg(long, value_parser = clap::value_parser!(u8).range(0..=3))]
    ram_band: u8,
}

/// Arguments for `load` / `debug` — a single Adapter Toolkit v2
/// DSL file path (`.yaml` / `.yml` / `.json`).
#[derive(clap::Args)]
struct AdapterFileArgs {
    /// Path to the adapter DSL file.
    file: PathBuf,
}

/// Arguments for `inspect` — a model directory containing a
/// `config.json` or a single `*.gguf` file.
#[derive(clap::Args)]
struct ModelDirArgs {
    /// Path to the model directory.
    model_dir: PathBuf,
}

/// Output format shared by `probe` and `run`. `text` is the
/// human-readable banner; `json` is a stable machine-readable
/// schema documented in `docs/CLI.md` (M4.9.f).
#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

/// Killer-demo execution mode. A is the no-spill baseline; B
/// validates the autonomous LRU spill trigger; C exercises
/// the forced LRU spill + lazy-restore transparency contract.
#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    A,
    B,
    C,
}

fn main() {
    let cli = Cli::parse();

    let exit_code = match cli.command {
        Command::Probe(args) => run_probe(args),
        Command::Run(args) => run_demo(args),
        Command::Explain(args) => run_explain(args),
        Command::Generate(args) => run_generate(args),
        Command::Load(args) => run_adapter_load(&args.file, false),
        Command::Debug(args) => run_adapter_load(&args.file, true),
        Command::Inspect(args) => run_adapter_inspect(&args.model_dir),
    };

    std::process::exit(exit_code);
}

/// `atenia load` / `atenia debug` — Adapter Toolkit v2 entry point.
/// Prints the adapter report to stdout, or a typed error to stderr.
/// Exit 0 on success, 2 on any toolkit error (matches the
/// missing-config.json convention used by `generate`).
fn run_adapter_load(file: &std::path::Path, verbose: bool) -> i32 {
    use atenia_engine::adapter_toolkit::{run_debug, run_load};
    let result = if verbose {
        run_debug(file)
    } else {
        run_load(file)
    };
    match result {
        Ok(report) => {
            println!("{report}");
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            2
        }
    }
}

/// `atenia inspect` — Adapter Toolkit v2 auto-detection. Emits a
/// loadable YAML DSL plus a resolved-spec preview to stdout.
fn run_adapter_inspect(model_dir: &std::path::Path) -> i32 {
    use atenia_engine::adapter_toolkit::run_inspect;
    match run_inspect(model_dir) {
        Ok(report) => {
            println!("{report}");
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            2
        }
    }
}

// ============================================================
// `atenia probe`
// ============================================================

/// Dispatch the probe subcommand. Behind the `hw-probe` Cargo
/// feature this calls `atenia_engine::hw_probe::probe()` and
/// renders the report in either text or JSON form. Without
/// the feature the function prints a friendly explanation and
/// returns exit code 2.
#[cfg(feature = "hw-probe")]
fn run_probe(args: ProbeArgs) -> i32 {
    let report = atenia_engine::hw_probe::probe();

    match args.output {
        OutputFormat::Text => {
            print!("{}", report);
        }
        OutputFormat::Json => {
            // Pretty-print so humans can still read the JSON
            // output, at the cost of a few extra bytes. Tools
            // that need compact form can pipe through `jq -c`.
            match serde_json::to_string_pretty(&report) {
                Ok(s) => println!("{}", s),
                Err(e) => {
                    eprintln!("error: failed to serialize report as JSON: {}", e);
                    return 3;
                }
            }
        }
    }

    0
}

#[cfg(not(feature = "hw-probe"))]
fn run_probe(_args: ProbeArgs) -> i32 {
    // M4.9.a contract: default-build users invoking `atenia
    // probe` get a clear, actionable message pointing at the
    // rebuild flag. The `hw-probe` feature pulls in wgpu
    // (~50 transitive crates) and NVML; gating the runtime
    // surface lets `atenia run` ship without those costs.
    eprintln!(
        "error: this build of `atenia` was compiled without the `hw-probe` feature.\n\
         \n\
         The `probe` subcommand needs cross-vendor GPU enumeration via wgpu,\n\
         which is gated behind a Cargo feature so library builds do not pull in\n\
         the wgpu dependency tree. Rebuild with one of:\n\
         \n    \
         cargo install --path . --features hw-probe\n  \
         cargo build --release --bin atenia --features hw-probe\n  \
         cargo run   --release --bin atenia --features hw-probe -- probe\n\
         \n\
         The `atenia run` and `atenia explain` subcommands work on the default\n\
         build and do not require the `hw-probe` feature."
    );
    2
}

// ============================================================
// `atenia run` — dispatches into atenia_engine::cli_run
// ============================================================

#[cfg(feature = "demo")]
fn run_demo(args: RunArgs) -> i32 {
    use atenia_engine::cli_run as cr;
    let translated = cr::RunArgs {
        model: args.model,
        mode: match args.mode {
            Mode::A => cr::Mode::A,
            Mode::B => cr::Mode::B,
            Mode::C => cr::Mode::C,
        },
        seq: args.seq,
        output: match args.output {
            OutputFormat::Text => cr::OutputFormat::Text,
            OutputFormat::Json => cr::OutputFormat::Json,
        },
        cache_dir: args.cache_dir,
        no_progress: args.no_progress,
    };
    cr::run(translated)
}

#[cfg(not(feature = "demo"))]
fn run_demo(_args: RunArgs) -> i32 {
    eprintln!(
        "error: this build of `atenia` was compiled without the `demo` feature.\n\
         \n\
         The `run` subcommand needs the `atenia_engine::demo` and\n\
         `atenia_engine::cli_run` modules. Both are gated behind the `demo`\n\
         Cargo feature (default-enabled). Rebuild with the default feature\n\
         set:\n\
         \n  \
         cargo install --path .\n  \
         cargo build --release --bin atenia\n  \
         cargo run   --release --bin atenia -- run --model <PATH> --mode a"
    );
    2
}

// ============================================================
// `atenia generate` — dispatches into atenia_engine::cli_generate
// ============================================================

#[cfg(feature = "demo")]
fn run_generate(args: GenerateArgs) -> i32 {
    use atenia_engine::cli_generate as cg;
    let translated = cg::GenerateArgs {
        prompt: args.prompt,
        model: args.model,
        max_tokens: args.max_tokens,
        output: match args.output {
            OutputFormat::Text => cg::OutputFormat::Text,
            OutputFormat::Json => cg::OutputFormat::Json,
        },
        cache_dir: args.cache_dir,
        no_progress: args.no_progress,
        no_chat_template: args.no_chat_template,
    };
    cg::run(translated)
}

#[cfg(not(feature = "demo"))]
fn run_generate(_args: GenerateArgs) -> i32 {
    eprintln!(
        "error: this build of `atenia` was compiled without the `demo` feature.\n\
         The `generate` subcommand needs the `atenia_engine::cli_generate` and\n\
         `atenia_engine::nn::llama::pipeline` modules. Rebuild with the default\n\
         feature set:\n  \
         cargo build --release --bin atenia"
    );
    2
}

// ============================================================
// `atenia explain` (legacy v13 surface, preserved)
// ============================================================

fn run_explain(args: ExplainArgs) -> i32 {
    let ctx = LearningContextSnapshot {
        gpu_available: args.gpu_available,
        vram_band: args.vram_band,
        ram_band: args.ram_band,
    };

    let trainer = SelfTrainer::new();

    match trainer.explain_decision(ctx) {
        Some(text) => {
            // Structured explanation should be available for
            // the same context if textual explanation exists;
            // unwrap here is acceptable at the CLI layer.
            let structured = match trainer.explain_decision_structured(ctx) {
                Some(s) => s,
                None => {
                    println!("No learned data available for the given context.");
                    return 0;
                }
            };

            let narrative = build_narrative(&text, &structured);
            println!("{}", narrative.narrative);
            0
        }
        None => {
            println!("No learned data available for the given context.");
            0
        }
    }
}
