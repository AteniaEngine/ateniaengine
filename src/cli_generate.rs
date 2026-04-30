//! M5.e — implementation of the `atenia generate` subcommand.
//!
//! Loads a Llama-family checkpoint via [`GenerationPipeline`]
//! and runs greedy decoding on `--prompt`, streaming each
//! generated token to stdout as it lands.
//!
//! Lives in the library (not `src/bin/atenia.rs`) so the
//! binary stays small and the orchestration is integration-
//! testable.
//!
//! Exit codes:
//!
//! | Code | Meaning                                                        |
//! |------|----------------------------------------------------------------|
//! | 0    | Generation finished cleanly (EOS or `--max-tokens` reached).   |
//! | 1    | Runtime error during load or generation (OOM, kernel fail).    |
//! | 2    | Argument / configuration error (model dir missing, bad flag). |

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;

use crate::nn::llama::{
    CollectingTokenSink, GenerationPipeline, GeneratedToken, StdoutTokenSink, TokenSink,
};

/// Output format for `atenia generate`. `text` prints the
/// generated tokens to stdout as they land + a stats line on
/// stderr; `json` accumulates into a structured report
/// printed at the end.
#[derive(Clone, Copy, Debug)]
pub enum OutputFormat {
    Text,
    Json,
}

/// Mirrors the `clap::Args` struct in `src/bin/atenia.rs`.
/// Decoupled so the implementation lives in the lib and the
/// binary just adapts.
pub struct GenerateArgs {
    pub prompt: String,
    pub model: PathBuf,
    pub max_tokens: usize,
    pub output: OutputFormat,
    pub cache_dir: Option<PathBuf>,
    pub no_progress: bool,
    /// When `true`, skip [`AteniaTokenizer::apply_chat_template`]
    /// and feed `prompt` to the model verbatim. Used for
    /// completion-style prompting and for tests that want
    /// deterministic input bytes.
    pub no_chat_template: bool,
}

#[derive(Serialize)]
struct GenerateReport {
    prompt: String,
    generated_text: String,
    /// Decoded text fragments, one per generated token. Empty
    /// strings denote special tokens (BOS/EOS).
    tokens: Vec<String>,
    token_ids: Vec<u32>,
    tokens_generated: usize,
    total_seconds: f64,
    tokens_per_second: f32,
    /// True iff the loop halted because the model emitted
    /// the EOS sentinel (rather than hitting `--max-tokens`).
    eos_reached: bool,
}

/// Heartbeat dots emitted to stderr while the (slow) model
/// load happens. Output goes to stderr so JSON / piped
/// stdout stays clean.
struct Heartbeat {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Heartbeat {
    fn start(interval_ms: u64) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            while !stop_clone.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(interval_ms));
                if stop_clone.load(Ordering::Relaxed) { break; }
                eprint!(".");
                let _ = std::io::stderr().flush();
            }
        });
        Heartbeat { stop, handle: Some(handle) }
    }
    fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        eprintln!();
    }
}

/// Soft-warn when system RAM is below the 28 GiB practical
/// floor for the 13B target. Mirrors `cli_run.rs`'s pattern.
fn warn_low_ram() {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let ram_gib = sys.total_memory() as f64 / (1024.0_f64.powi(3));
    if ram_gib < 28.0 {
        eprintln!(
            "warning: detected {ram_gib:.1} GiB of system RAM. The 13B target needs \
             ~28 GiB minimum at BF16. Smaller models (TinyLlama, SmolLM2, Llama 3.2 1B) \
             work fine on this host."
        );
    }
}

/// Entry point for `atenia generate`. Returns the process
/// exit code (0/1/2) per the contract documented above.
pub fn run(args: GenerateArgs) -> i32 {
    // ---- Validate paths up front (exit 2) ----
    if !args.model.exists() {
        eprintln!(
            "error: model directory does not exist: {}",
            args.model.display()
        );
        return 2;
    }
    if !args.model.join("config.json").exists() {
        eprintln!(
            "error: model directory {} is missing config.json — does it actually contain a HuggingFace checkpoint?",
            args.model.display()
        );
        return 2;
    }
    if args.max_tokens == 0 {
        eprintln!("error: --max-tokens must be > 0");
        return 2;
    }
    let _ = args.cache_dir;

    warn_low_ram();

    // ---- Load model with heartbeat dots ----
    if matches!(args.output, OutputFormat::Text) {
        eprintln!("Loading model from {} ...", args.model.display());
    }
    let heartbeat = if !args.no_progress && matches!(args.output, OutputFormat::Text) {
        Some(Heartbeat::start(2000))
    } else {
        None
    };
    let load_start = std::time::Instant::now();
    let pipe = match GenerationPipeline::from_model_dir(&args.model) {
        Ok(p) => p,
        Err(e) => {
            if let Some(hb) = heartbeat { hb.stop(); }
            eprintln!("error: failed to load model: {e}");
            return 1;
        }
    };
    let load_secs = load_start.elapsed().as_secs_f32();
    if let Some(hb) = heartbeat { hb.stop(); }

    if matches!(args.output, OutputFormat::Text) {
        let resident_gib = pipe.store.resident_bytes() as f64 / (1024.0_f64.powi(3));
        eprintln!(
            "Model loaded in {:.1}s ({} parameters, {:.2} GiB resident).",
            load_secs, pipe.store.len(), resident_gib
        );
        eprintln!();
        eprintln!("> {}", args.prompt);
        eprintln!();
    }

    // ---- Generate ----
    //
    // Two sinks composed: a `CollectingTokenSink` that builds
    // the report (used by both text and json paths), and a
    // `StdoutTokenSink` that streams text-mode output as
    // tokens land. JSON mode skips the stdout streaming so
    // the final JSON is the only thing on stdout.
    let mut collected = CollectingTokenSink::default();
    let gen_start = std::time::Instant::now();

    let result = match args.output {
        OutputFormat::Text => {
            // **M5.e UX fix.** Prefill on 13B takes 3-5
            // minutes on CPU with no intermediate output;
            // before this fix users naturally interpreted
            // the silent gap as a hang and ctrl-C'd. Show a
            // visible "thinking" indicator that stops the
            // moment the first token lands.
            if !args.no_progress {
                eprintln!("Prefilling prompt and generating ...");
            }
            let prefill_hb = if !args.no_progress {
                Some(Heartbeat::start(2000))
            } else {
                None
            };
            let mut combined = TextAndCollectSink {
                stdout: StdoutTokenSink,
                collect: &mut collected,
                prefill_heartbeat: prefill_hb,
            };
            let r = run_generate(&pipe, &args, &mut combined);
            // If generation errored before any token landed,
            // make sure the heartbeat is stopped.
            if let Some(hb) = combined.prefill_heartbeat.take() {
                hb.stop();
            }
            r
        }
        OutputFormat::Json => {
            run_generate(&pipe, &args, &mut collected)
        }
    };
    let total_secs = gen_start.elapsed().as_secs_f64();

    let text = match result {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: generation failed: {e}");
            return 1;
        }
    };

    let eos_reached = collected.tokens.last()
        .map(|t| t.is_eos)
        .unwrap_or(false);
    let tps = if total_secs > 0.0 {
        collected.tokens.len() as f32 / total_secs as f32
    } else { 0.0 };

    match args.output {
        OutputFormat::Text => {
            // Final newline so the prompt returns to the
            // user's shell on its own line.
            println!();
            eprintln!();
            eprintln!("---");
            eprintln!(
                "Generated: {} tokens in {:.1}s ({:.2} tok/s){}",
                collected.tokens.len(), total_secs, tps,
                if eos_reached { " [EOS]" } else { " [max-tokens reached]" },
            );
        }
        OutputFormat::Json => {
            let report = GenerateReport {
                prompt: args.prompt.clone(),
                generated_text: text,
                tokens: collected.tokens.iter().map(|t| t.text.clone()).collect(),
                token_ids: collected.tokens.iter().map(|t| t.token_id).collect(),
                tokens_generated: collected.tokens.len(),
                total_seconds: total_secs,
                tokens_per_second: tps,
                eos_reached,
            };
            match serde_json::to_string_pretty(&report) {
                Ok(s) => println!("{}", s),
                Err(e) => {
                    eprintln!("error: failed to serialise report as JSON: {e}");
                    return 1;
                }
            }
        }
    }

    0
}

fn run_generate<S: TokenSink>(
    pipe: &GenerationPipeline,
    args: &GenerateArgs,
    sink: &mut S,
) -> Result<String, String> {
    if args.no_chat_template {
        pipe.generate_raw(&args.prompt, args.max_tokens, sink)
            .map_err(|e| e.to_string())
    } else {
        pipe.generate(&args.prompt, args.max_tokens, sink)
            .map_err(|e| e.to_string())
    }
}

/// Helper sink that mirrors events to both a stdout streamer
/// and a collecting buffer. Used by the text path so the
/// final stats line can be computed off the collected events
/// while the user sees tokens stream live.
///
/// **M5.e UX fix.** Holds an optional [`Heartbeat`] that ticks
/// dots to stderr during the long prefill phase (no output
/// for 3-5 min on 13B is indistinguishable from a hang for
/// the user). The first `on_token` call stops the heartbeat
/// and clears the dots line so the streamed text starts on a
/// fresh line. From the second token onward the heartbeat
/// stays off and tokens stream live.
struct TextAndCollectSink<'a> {
    stdout: StdoutTokenSink,
    collect: &'a mut CollectingTokenSink,
    prefill_heartbeat: Option<Heartbeat>,
}

impl<'a> TokenSink for TextAndCollectSink<'a> {
    fn on_token(&mut self, t: &GeneratedToken) {
        if let Some(hb) = self.prefill_heartbeat.take() {
            // First token landed → prefill finished. Stop the
            // dots and emit a newline so the generated text
            // starts on a clean line.
            hb.stop();
        }
        self.stdout.on_token(t);
        self.collect.on_token(t);
    }
}
