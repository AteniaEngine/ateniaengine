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
use std::sync::{Arc, Mutex};

use atenia_engine::cli::logging::{self, CliLogConfig};
use atenia_engine::cli::CliError;
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

    // ---- CLI-2 global logging flags ----
    // `global = true` makes every flag accepted in any position,
    // before or after the subcommand (`atenia --quiet load X` and
    // `atenia load X --quiet` both work).
    /// Quiet mode: only human errors on stderr, no progress, no
    /// banners, no counters.
    #[arg(long, short = 'q', global = true)]
    quiet: bool,

    /// Verbose mode: show useful progress steps (log level `info`).
    #[arg(long, short = 'v', global = true)]
    verbose: bool,

    /// Debug mode: show resolved configuration and internal
    /// decisions (log level `debug`).
    #[arg(long, global = true)]
    debug: bool,

    /// Trace mode: fine-grained CLI-frontier detail (log level
    /// `trace`).
    #[arg(long, global = true)]
    trace: bool,

    /// Explicit log level — overrides --quiet/--verbose/--debug/
    /// --trace. One of: error, warn, info, debug, trace.
    #[arg(long, global = true, value_name = "LEVEL")]
    log_level: Option<String>,

    /// Also write logs to this file (created if absent, parent
    /// directories included). Does not affect stdout.
    #[arg(long, global = true, value_name = "PATH")]
    log_file: Option<PathBuf>,

    /// Use this trace id for the run instead of an auto-generated
    /// one.
    #[arg(long, global = true, value_name = "ID")]
    trace_id: Option<String>,

    /// Disable colored output. Accepted now; colored output is not
    /// implemented yet, so this is currently a no-op.
    #[arg(long, global = true)]
    no_color: bool,
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

    /// **MOE-FULL-14** — controlled, opt-in MoE generation (token-id based).
    ///
    /// Loads a recognised + certified MoE checkpoint (Mixtral / Qwen-MoE /
    /// DeepSeek-MoE) through the controlled production path and generates
    /// greedily from a list of token ids, stopping at EOS or `--max-new`.
    /// Refuses unless `ATENIA_ENABLE_MOE=1` or `--experimental-moe` is set, and
    /// refuses non-certified families / unsupported variants with a clear
    /// message. The dense `generate` path is unaffected. Token-id based (no
    /// tokenizer dependency) — for the experimental MoE runtime only.
    ///
    /// ```text
    /// atenia moe-generate --model models/mixtral-tiny \
    ///     --prompt-ids "22,25,29" --max-new 8 --experimental-moe
    /// ```
    MoeGenerate(MoeGenerateArgs),

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

    /// **CLI-3** — global host + build diagnostics: CPU, RAM,
    /// CUDA, build flavour, supported formats and quants.
    Doctor(JsonArgs),

    /// **CLI-3** — pre-flight diagnosis of a model directory
    /// (format, family, tokenizer, adapter resolution). Does not
    /// run generation.
    ///
    /// ```text
    /// atenia diagnose --model ./models/llama-3.2-1b-instruct
    /// ```
    Diagnose(DiagnoseArgs),

    /// **CLI-3** — list the engine's capabilities: supported
    /// families, out-of-scope architectures, formats, quants,
    /// features.
    Capabilities(JsonArgs),

    /// **CLI-4** — interactive chat REPL against a checkpoint.
    /// Keeps an in-memory history and applies the model chat
    /// template. Commands: /help, /history, /reset, /exit.
    ///
    /// ```text
    /// atenia chat --model ./models/llama-3.2-1b-instruct
    /// ```
    Chat(ChatArgs),

    /// **CLI-6** — download a curated, public Hugging Face
    /// checkpoint and place it under `./models/<alias>/`.
    ///
    /// Use the literal alias `list` to print the available
    /// catalog instead of downloading.
    ///
    /// ```text
    /// atenia download list
    /// atenia download tinyllama
    /// atenia download smollm2-135m --dir ./scratch/sm
    /// atenia download tinyllama --dry-run
    /// ```
    Download(DownloadArgs),

    /// **CLI-7** — first-run guided onboarding. Prints the
    /// recommended four-step flow (`doctor` → `download` →
    /// `diagnose` → `chat`) with the exact commands to run.
    /// Pass `--download` to actually fetch the recommended
    /// model in place of step 2.
    ///
    /// ```text
    /// atenia quickstart
    /// atenia quickstart --download
    /// atenia quickstart --model tinyllama
    /// ```
    Quickstart(QuickstartArgs),

    /// **AQS-10** — render an AQS certification report + manifest
    /// draft from a pre-computed end-to-end results file.
    ///
    /// Experimental. This command does NOT load a model or run a
    /// forward — AQS end-to-end certification needs a per-model F64
    /// reference that only the end-to-end harness can produce. Feed it
    /// that harness output:
    ///
    /// ```text
    /// atenia search --results aqs-results.json --report
    /// atenia search --results aqs-results.json --report --manifest
    /// ```
    Search(SearchArgs),
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

/// Arguments for `doctor` / `capabilities` — only an output toggle.
#[derive(clap::Args)]
struct JsonArgs {
    /// Emit the report as JSON instead of the human checklist.
    #[arg(long)]
    json: bool,
}

/// Arguments for `diagnose` — a model directory plus an output
/// toggle.
#[derive(clap::Args)]
struct DiagnoseArgs {
    /// Path to the model directory to diagnose.
    #[arg(long)]
    model: PathBuf,

    /// Emit the report as JSON instead of the human checklist.
    #[arg(long)]
    json: bool,
}

/// Arguments for `moe-generate` — controlled, opt-in MoE generation.
#[derive(clap::Args)]
struct MoeGenerateArgs {
    /// Path to the MoE model directory (config.json + *.safetensors).
    #[arg(long)]
    model: PathBuf,
    /// Comma-separated prompt token ids, e.g. "22,25,29".
    #[arg(long)]
    prompt_ids: String,
    /// Maximum number of new tokens to generate.
    #[arg(long, default_value_t = 8)]
    max_new: usize,
    /// Opt in to the controlled MoE runtime (equivalent to ATENIA_ENABLE_MOE=1).
    #[arg(long)]
    experimental_moe: bool,
}

/// Arguments for `download` — curated model fetcher.
#[derive(clap::Args)]
struct DownloadArgs {
    /// Curated alias (run `atenia download list` to see the
    /// available ones), or the literal token `list`.
    alias: String,

    /// Destination directory. Defaults to `./models/<default_subdir>`
    /// where `default_subdir` comes from the catalog entry.
    #[arg(long, value_name = "DIR")]
    dir: Option<PathBuf>,

    /// Overwrite an existing destination directory.
    #[arg(long)]
    force: bool,

    /// Resolve the alias and print what would be downloaded
    /// without writing any files.
    #[arg(long)]
    dry_run: bool,

    /// Skip the post-download "Next:" footer that suggests
    /// `atenia diagnose` / `atenia chat`.
    #[arg(long)]
    no_suggest: bool,
}

/// Arguments for `quickstart` — first-run UX.
#[derive(clap::Args)]
struct QuickstartArgs {
    /// Actually run the recommended download instead of just
    /// printing the suggested commands.
    #[arg(long)]
    download: bool,

    /// Curated alias to recommend / download. Defaults to the
    /// smallest entry in the catalog so the first run finishes
    /// in a coffee break on any connection.
    #[arg(long, default_value = atenia_engine::cli::quickstart::DEFAULT_MODEL)]
    model: String,

    /// Custom destination directory passed through to
    /// `atenia download` when `--download` is set.
    #[arg(long, value_name = "DIR")]
    dir: Option<PathBuf>,

    /// Suppress the "Next:" / "Want the shortest path?" footer.
    #[arg(long)]
    no_suggest: bool,
}

/// Arguments for `search` — the AQS-10 certification front-end.
#[derive(clap::Args)]
struct SearchArgs {
    /// Path to a pre-computed AQS end-to-end results JSON file
    /// (produced by the AQS end-to-end harness). Required.
    #[arg(long, value_name = "FILE")]
    results: Option<PathBuf>,

    /// Print the human-readable certification report table.
    #[arg(long)]
    report: bool,

    /// Print the experimental `3.0.0-draft` manifest.
    #[arg(long)]
    manifest: bool,

    /// Reserved: opt into real GPTQ on a future model-driving path.
    /// No effect on the results-file path.
    #[arg(long)]
    include_gptq: bool,
}

/// Arguments for `chat` — the interactive REPL.
#[derive(clap::Args)]
struct ChatArgs {
    /// Path to the model directory (config.json + weights, or a
    /// directory with a single .gguf file).
    #[arg(long)]
    model: PathBuf,

    /// Maximum new tokens per assistant turn.
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Sampling temperature. Accepted for forward compatibility;
    /// chat currently uses greedy decoding, so a non-zero value is
    /// ignored with a warning.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Skip the model chat template; use a plain User:/Assistant:
    /// transcript instead.
    #[arg(long)]
    no_chat_template: bool,
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

    // **CLI-2 logging.** Resolve the effective level from the
    // global flags, then initialise the CLI logger. A bad
    // `--log-file` is the only failure here; render it like any
    // other CLI error and exit.
    let level = match logging::resolve_level(
        cli.quiet,
        cli.verbose,
        cli.debug,
        cli.trace,
        cli.log_level.as_deref(),
    ) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(e.exit.code());
        }
    };
    if let Err(e) = logging::init_cli_logging(CliLogConfig {
        level,
        log_file: cli.log_file.clone(),
        trace_id: cli.trace_id.clone(),
        no_color: cli.no_color,
    }) {
        eprintln!("{e}");
        std::process::exit(e.exit.code());
    }

    // **CLI-1 panic boundary.** Install a panic hook that captures
    // the panic message + location into a slot, then run the
    // dispatch under `catch_unwind`. An internal panic becomes a
    // rendered `E-INTERNAL-PANIC` (exit 101) instead of a raw Rust
    // backtrace dumped at the user. The hook replaces the default
    // one, so the backtrace is not printed; the captured message
    // is surfaced in the error's technical details.
    let panic_slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    {
        let slot = Arc::clone(&panic_slot);
        std::panic::set_hook(Box::new(move |info| {
            let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = info.payload().downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic payload".to_string()
            };
            let located = match info.location() {
                Some(l) => format!("{payload} (at {}:{})", l.file(), l.line()),
                None => payload,
            };
            if let Ok(mut guard) = slot.lock() {
                *guard = Some(located);
            }
        }));
    }

    let dispatch = std::panic::AssertUnwindSafe(|| match cli.command {
        Command::Probe(args) => run_probe(args),
        Command::Run(args) => run_demo(args),
        Command::Explain(args) => run_explain(args),
        Command::Generate(args) => run_generate(args),
        Command::MoeGenerate(args) => run_moe_generate(args),
        Command::Load(args) => run_adapter_load(&args.file, false),
        Command::Debug(args) => run_adapter_load(&args.file, true),
        Command::Inspect(args) => run_adapter_inspect(&args.model_dir),
        Command::Doctor(args) => {
            atenia_engine::cli::diagnostics::run_doctor(args.json)
        }
        Command::Diagnose(args) => {
            atenia_engine::cli::diagnostics::run_diagnose(&args.model, args.json)
        }
        Command::Capabilities(args) => {
            atenia_engine::cli::diagnostics::run_capabilities(args.json)
        }
        Command::Chat(args) => {
            atenia_engine::cli::chat::run_chat(atenia_engine::cli::chat::ChatArgs {
                model: args.model,
                max_tokens: args.max_tokens,
                temperature: args.temperature,
                no_chat_template: args.no_chat_template,
            })
        }
        Command::Download(args) => {
            atenia_engine::cli::download::run_download(
                atenia_engine::cli::download::DownloadArgs {
                    alias: args.alias,
                    dir: args.dir,
                    force: args.force,
                    dry_run: args.dry_run,
                    no_suggest: args.no_suggest,
                },
            )
        }
        Command::Quickstart(args) => {
            atenia_engine::cli::quickstart::run_quickstart(
                atenia_engine::cli::quickstart::QuickstartArgs {
                    download: args.download,
                    model: args.model,
                    dir: args.dir,
                    no_suggest: args.no_suggest,
                },
            )
        }
        Command::Search(args) => {
            atenia_engine::cli::search::run_search(atenia_engine::cli::search::SearchArgs {
                results: args.results,
                report: args.report,
                manifest: args.manifest,
                include_gptq: args.include_gptq,
            })
        }
    });

    let exit_code = match std::panic::catch_unwind(dispatch) {
        Ok(code) => code,
        Err(_) => {
            let detail = panic_slot
                .lock()
                .ok()
                .and_then(|mut g| g.take())
                .unwrap_or_else(|| "unknown internal panic".to_string());
            let err = CliError::internal_panic(detail);
            eprintln!("{err}");
            err.exit.code()
        }
    };

    std::process::exit(exit_code);
}

/// `atenia load` / `atenia debug` — Adapter Toolkit v2 entry point.
/// Prints the adapter report to stdout on success, or a rendered
/// [`CliError`] to stderr. The exit code is set by the CLI error
/// layer: 2 for spec/adapter/validation faults, 1 for I/O faults.
fn run_adapter_load(file: &std::path::Path, verbose: bool) -> i32 {
    use atenia_engine::adapter_toolkit::{run_debug, run_load};
    let cmd = if verbose { "debug" } else { "load" };
    logging::info(&format!("command start: {cmd}"));
    logging::debug(&format!("adapter spec file: {}", file.display()));
    let result = if verbose {
        run_debug(file)
    } else {
        run_load(file)
    };
    match result {
        Ok(report) => {
            println!("{report}");
            logging::info(&format!("command completed: {cmd}"));
            0
        }
        Err(e) => {
            // Boundary translation: ToolkitError -> CliError. The
            // toolkit itself is unchanged.
            let err = CliError::from(e);
            eprintln!("{err}");
            err.exit.code()
        }
    }
}

/// `atenia inspect` — Adapter Toolkit v2 auto-detection. Emits a
/// loadable YAML DSL plus a resolved-spec preview to stdout, or a
/// rendered [`CliError`] to stderr.
/// **MOE-FULL-14** — `atenia moe-generate`: controlled, opt-in MoE generation.
fn run_moe_generate(args: MoeGenerateArgs) -> i32 {
    logging::info("command start: moe-generate");
    if args.experimental_moe {
        // Equivalent to ATENIA_ENABLE_MOE=1 for this process.
        // SAFETY: single-threaded CLI startup; no other thread reads env here.
        unsafe {
            std::env::set_var(atenia_engine::moe::ENABLE_MOE_ENV, "1");
        }
    }
    let prompt_ids: Result<Vec<u32>, _> =
        args.prompt_ids.split(',').map(|s| s.trim().parse::<u32>()).collect();
    let prompt_ids = match prompt_ids {
        Ok(v) if !v.is_empty() => v,
        _ => {
            eprintln!("error: --prompt-ids must be a non-empty comma-separated list of token ids");
            return 2;
        }
    };
    match atenia_engine::moe::controlled_moe_generate(&args.model, &prompt_ids, args.max_new) {
        Ok(ids) => {
            // MOE-PROD-3: optional expert-cache hit-ratio report (disk tier).
            if std::env::var("ATENIA_MOE_CACHE_STATS").as_deref() == Ok("1") {
                let s = atenia_engine::moe::graph_op::aggregate_resident_cache_stats();
                let total = s.hits + s.misses;
                let ratio = if total > 0 { 100.0 * s.hits as f64 / total as f64 } else { 0.0 };
                eprintln!(
                    "[ATENIA] MoE expert cache: hits={} misses={} hit_ratio={:.1}% tier_bytes_read={} (~{:.2} GiB)",
                    s.hits,
                    s.misses,
                    ratio,
                    s.tier_bytes_read,
                    s.tier_bytes_read as f64 / (1024.0 * 1024.0 * 1024.0),
                );
            }
            let csv = ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
            println!("{csv}");
            0
        }
        Err(e) => {
            // Clear, actionable message; exit 2 (the CLI's "input/usage" code).
            eprintln!("{e}");
            2
        }
    }
}

fn run_adapter_inspect(model_dir: &std::path::Path) -> i32 {
    use atenia_engine::adapter_toolkit::{run_inspect, ToolkitError};
    logging::info("command start: inspect");
    logging::debug(&format!("model directory: {}", model_dir.display()));
    match run_inspect(model_dir) {
        Ok(report) => {
            println!("{report}");
            logging::info("command completed: inspect");
            0
        }
        Err(e) => {
            // `inspect` reads a model *directory*, not a spec file.
            // The generic `ToolkitError::Io` -> `CliError` mapping
            // is worded for spec files, so re-word the I/O case
            // here for the inspect context. Other variants keep
            // the shared mapping.
            let err = match e {
                ToolkitError::Io(msg) => CliError::adapter_inspect_failed(msg),
                other => CliError::from(other),
            };
            eprintln!("{err}");
            err.exit.code()
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
