//! Implementation of `atenia run` (Modes A / B / C).
//!
//! M4.9.c lands Mode A. M4.9.d adds Mode C. M4.9.e adds Mode
//! B. The three modes share the report struct, the heartbeat-
//! dots progress UX, the hardware soft-warning, and the
//! disk-throughput probe; only the per-mode forward orchestration
//! differs.
//!
//! This module is private to the `atenia` binary — its public
//! API is `pub fn run(args: RunArgs) -> i32` consumed by
//! `src/bin/atenia.rs`.

use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::demo::{
    argmax_row, build_and_load_llama, cache_dir_for, make_context,
};
use crate::nn::llama::LlamaRuntime;
use crate::tensor::tensor::Tensor;

/// Args mirror of `crate::RunArgs` from `atenia.rs`. Kept as
/// a private re-declaration so this module compiles without
/// the parent's clap derives leaking through.
pub struct RunArgs {
    pub model: PathBuf,
    pub mode: Mode,
    pub seq: usize,
    pub output: OutputFormat,
    pub cache_dir: Option<PathBuf>,
    pub no_progress: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    A,
    B,
    C,
}

#[derive(Clone, Copy)]
pub enum OutputFormat {
    Text,
    Json,
}

// ============================================================
// Output schema
// ============================================================

/// Top-level demo report. Stable across M4.9.+ — adding
/// fields is non-breaking, removing fields is a major bump.
/// Documented in `docs/CLI.md` (added in M4.9.f).
#[derive(Serialize)]
struct DemoReport {
    version: String,
    mode: &'static str,
    seq: usize,
    model: ModelReport,
    phases: PhasesReport,
    /// Per-position argmax results. Length = `seq`.
    argmax: Vec<ArgmaxEntry>,
    /// Logit-magnitude sanity checks; non-finite count
    /// catches numerical blowup along the layer chain.
    logit_stats: LogitStats,
    /// Mode A leaves this `None`. Modes B / C populate it
    /// (M4.9.d / .e).
    contract: Option<TransparencyContract>,
    total_seconds: f64,
}

#[derive(Serialize)]
struct ModelReport {
    path: PathBuf,
    layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    param_count: usize,
    storage: &'static str,
}

#[derive(Serialize)]
struct PhasesReport {
    build_seconds: f32,
    load_seconds: f32,
    /// MB/s estimate based on the BF16 on-disk size of a
    /// canonical 13B-class checkpoint (~26 GB). Approximate
    /// for non-13B models — the field name carries `_estimate`
    /// so JSON consumers do not over-trust it.
    load_throughput_mb_s_estimate: f32,
    forward_seconds: f32,
    /// Mode B/C only. `None` for Mode A.
    spill_seconds: Option<f32>,
    /// Mode B/C only.
    spill_tensors_migrated: Option<usize>,
    /// Mode C only (Mode B does not warmup before its panic).
    warmup_forward_seconds: Option<f32>,
}

#[derive(Serialize)]
struct ArgmaxEntry {
    position: usize,
    token_id: usize,
    logit: f32,
}

#[derive(Serialize)]
struct LogitStats {
    max_abs: f32,
    mean_abs: f32,
    finite: usize,
    total: usize,
}

#[derive(Serialize)]
struct TransparencyContract {
    name: &'static str,
    pre: ArgmaxEntry,
    post: ArgmaxEntry,
    bit_exact: bool,
    description: &'static str,
}

// ============================================================
// Heartbeat-dots progress UX
// ============================================================

/// Spawn a heartbeat thread that prints `.` to stderr every
/// `interval_ms` until `stop` is set. Returns a join handle.
/// Output goes to stderr so JSON on stdout stays clean.
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
                std::thread::sleep(std::time::Duration::from_millis(
                    interval_ms,
                ));
                if stop_clone.load(Ordering::Relaxed) {
                    break;
                }
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

// ============================================================
// Hardware soft-warning + disk-throughput probe
// ============================================================

/// Read total system RAM via `sysinfo`. Returns the value in
/// gigabytes (decimal — 1 GB = 10^9 bytes).
fn total_ram_gb() -> f64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    // `sysinfo` 0.30 returns `total_memory()` in bytes
    // (the `*Ext` traits were merged into the inherent impls).
    (sys.total_memory() as f64) / 1_000_000_000.0
}

fn ram_soft_warning() {
    let ram_gb = total_ram_gb();
    if ram_gb < 28.0 {
        eprintln!(
            "warning: detected {:.1} GB total RAM. Llama 2 13B Chat in BF16\n\
             needs ~26 GB resident parameters; the forward will likely fail\n\
             with OOM during the load step. Recommended: at least 32 GB RAM.\n",
            ram_gb,
        );
    }
}

/// Quick sequential-write benchmark on `cache_dir`. Writes a
/// 100 MB scratch file, measures wall-clock, deletes the file.
/// Warns if throughput is below the 200 MB/s floor.
///
/// Returns the measured throughput in MB/s (or `None` if the
/// probe could not run, e.g. cache_dir is not writable).
fn cache_dir_disk_probe(cache_dir: &Path) -> Option<f64> {
    use std::fs;

    let _ = fs::create_dir_all(cache_dir);
    let probe_path = cache_dir.join(".atenia_throughput_probe");

    let payload: Vec<u8> = vec![0xAA; 100 * 1024 * 1024];

    let t0 = Instant::now();
    let result = fs::write(&probe_path, &payload);
    let elapsed = t0.elapsed().as_secs_f64();

    let _ = fs::remove_file(&probe_path);

    match result {
        Ok(_) if elapsed > 0.0 => {
            let mb_s = 100.0 / elapsed;
            if mb_s < 200.0 {
                eprintln!(
                    "warning: cache directory throughput is {:.1} MB/s\n\
                     ({}). Modes B / C will be very slow. Use an NVMe-backed\n\
                     directory; pass `--cache-dir <PATH>` or set\n\
                     ATENIA_DISK_TIER_DIR.\n",
                    mb_s,
                    cache_dir.display(),
                );
            }
            Some(mb_s)
        }
        _ => None,
    }
}

// ============================================================
// Output renderer
// ============================================================

fn render(report: &DemoReport, format: OutputFormat) {
    match format {
        OutputFormat::Json => {
            match serde_json::to_string_pretty(report) {
                Ok(s) => println!("{}", s),
                Err(e) => {
                    eprintln!("error: failed to serialise report as JSON: {}", e);
                }
            }
        }
        OutputFormat::Text => {
            render_text(report);
        }
    }
}

fn render_text(r: &DemoReport) {
    println!();
    println!("=== Atenia v20 Killer Demo — Llama-family — Mode {} ===", r.mode.to_uppercase());
    println!();
    println!("Run:");
    println!("  atenia version: {}", r.version);
    println!("  Model:          {}", r.model.path.display());
    println!(
        "                  {} layers × hidden {} × intermediate {} (vocab {})",
        r.model.layers, r.model.hidden_size, r.model.intermediate_size, r.model.vocab_size,
    );
    println!(
        "  Parameters:     {} ({} storage)",
        r.model.param_count, r.model.storage,
    );
    println!("  Sequence:       {}", r.seq);
    println!();
    println!("Phases:");
    println!(
        "  Build graph .................... {:>6.2}s",
        r.phases.build_seconds,
    );
    println!(
        "  Load weights ................. {:>8.2}s   (~{:.0} MB/s)",
        r.phases.load_seconds, r.phases.load_throughput_mb_s_estimate,
    );
    if let Some(warmup) = r.phases.warmup_forward_seconds {
        println!(
            "  Warmup forward ............... {:>8.2}s",
            warmup,
        );
    }
    if let Some(spill_secs) = r.phases.spill_seconds {
        let migrated = r.phases.spill_tensors_migrated.unwrap_or(0);
        println!(
            "  Force LRU spill ................ {:>6.2}s   ({} tensors migrated)",
            spill_secs, migrated,
        );
    }
    println!(
        "  Forward ...................... {:>8.2}s",
        r.phases.forward_seconds,
    );
    println!();
    println!("Per-position argmax:");
    for entry in &r.argmax {
        println!(
            "  Pos {}: argmax id = {:>5}   logit = {:.4}",
            entry.position, entry.token_id, entry.logit,
        );
    }
    println!();
    println!(
        "Logit stats: max |v| = {:.4}   mean |v| = {:.4}   finite = {}/{}",
        r.logit_stats.max_abs,
        r.logit_stats.mean_abs,
        r.logit_stats.finite,
        r.logit_stats.total,
    );
    if let Some(contract) = &r.contract {
        println!();
        println!("Transparency contract:");
        println!(
            "  argmax(pre)  = {}, logit {:.4}",
            contract.pre.token_id, contract.pre.logit,
        );
        println!(
            "  argmax(post) = {}, logit {:.4}",
            contract.post.token_id, contract.post.logit,
        );
        println!(
            "  {} {}",
            if contract.bit_exact { "[PASS] ✓" } else { "[FAIL] ✗" },
            contract.description,
        );
    }
    println!();
    println!(
        "Total wall-clock: {:.1} seconds ({:.1} minutes).",
        r.total_seconds,
        r.total_seconds / 60.0,
    );
    println!();
}

// ============================================================
// Mode dispatcher
// ============================================================

pub fn run(args: RunArgs) -> i32 {
    if !args.model.exists() {
        eprintln!(
            "error: model directory not found: {}\n\
             \n\
             The killer demo expects a Llama-family checkpoint at the path you\n\
             pass via --model (or the ATENIA_LLAMA2_13B_DIR environment\n\
             variable). For Llama 2 13B Chat:\n\
             \n  \
             huggingface-cli download meta-llama/Llama-2-13b-chat-hf \\\n    \
             --local-dir <model_dir> \\\n    \
             --include '*.safetensors' '*.json' 'tokenizer*'\n\
             \n\
             The cache / spill directory should be on an NVMe drive; see\n\
             --cache-dir / ATENIA_DISK_TIER_DIR.",
            args.model.display(),
        );
        return 2;
    }

    ram_soft_warning();

    match args.mode {
        Mode::A => run_mode_a(args),
        Mode::B => run_mode_b(args),
        Mode::C => run_mode_c(args),
    }
}

// ============================================================
// Mode A — clean RAM, no spill, no LRU
// ============================================================

fn run_mode_a(args: RunArgs) -> i32 {
    let total_start = Instant::now();
    let runtime = LlamaRuntime { batch: 1, seq: args.seq };

    // Canonical M4.6 / M4.7.6.d input. The CLI hard-codes
    // this to keep the demo's argmax reproducible against
    // documented baselines; Mode A's scientific value is the
    // wall-clock + argmax stability across builds, not a
    // library API for arbitrary inputs.
    let token_pattern: Vec<f32> = match args.seq {
        1 => vec![1.0],
        4 => vec![1.0, 100.0, 200.0, 300.0],
        n => {
            // Reasonable extension: BOS + ascending integers.
            let mut v = vec![1.0];
            for i in 1..n {
                v.push((i as f32) * 100.0);
            }
            v
        }
    };

    eprintln!("=== Atenia v20 Killer Demo — Mode A (clean RAM) ===");
    eprintln!();
    eprintln!("Loading {} (this typically takes ~3 min on NVMe) ...",
        args.model.display());

    let hb = if !args.no_progress {
        Some(Heartbeat::start(2_000))
    } else {
        None
    };
    let (mut graph, metrics) = build_and_load_llama(
        &args.model,
        runtime,
        /*verbose=*/ false,
    );
    if let Some(hb) = hb {
        hb.stop();
    }

    eprintln!(
        "Loaded {} parameters in {:.1}s. Running forward (CPU; M4.8 stack: \
         3.5x baseline) ...",
        metrics.param_count, metrics.load_secs,
    );

    let hb = if !args.no_progress {
        Some(Heartbeat::start(2_000))
    } else {
        None
    };
    let tokens = Tensor::new_cpu(vec![1, args.seq], token_pattern);
    let fwd_start = Instant::now();
    let outputs = graph.execute(vec![tokens]);
    let fwd_secs = fwd_start.elapsed().as_secs_f32();
    if let Some(hb) = hb {
        hb.stop();
    }

    let logits = &outputs[0];
    let slice = logits.as_cpu_slice();
    let vocab = metrics.config.vocab_size;
    assert_eq!(slice.len(), args.seq * vocab, "logit slice mismatch");

    let max_abs = slice.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let mean_abs: f32 =
        slice.iter().map(|v| v.abs()).sum::<f32>() / slice.len() as f32;
    let finite = slice.iter().filter(|v| v.is_finite()).count();

    let mut argmax_entries = Vec::with_capacity(args.seq);
    for pos in 0..args.seq {
        let row = &slice[pos * vocab..(pos + 1) * vocab];
        let (id, logit) = argmax_row(row, vocab);
        argmax_entries.push(ArgmaxEntry { position: pos, token_id: id, logit });
    }

    let total_seconds = total_start.elapsed().as_secs_f64();

    let report = DemoReport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        mode: "a",
        seq: args.seq,
        model: ModelReport {
            path: args.model.clone(),
            layers: metrics.config.num_hidden_layers,
            hidden_size: metrics.config.hidden_size,
            intermediate_size: metrics.config.intermediate_size,
            vocab_size: metrics.config.vocab_size,
            param_count: metrics.param_count,
            storage: "bf16",
        },
        phases: PhasesReport {
            build_seconds: metrics.build_secs,
            load_seconds: metrics.load_secs,
            load_throughput_mb_s_estimate: 26_000.0
                / metrics.load_secs.max(0.01),
            forward_seconds: fwd_secs,
            spill_seconds: None,
            spill_tensors_migrated: None,
            warmup_forward_seconds: None,
        },
        argmax: argmax_entries,
        logit_stats: LogitStats {
            max_abs,
            mean_abs,
            finite,
            total: slice.len(),
        },
        contract: None,
        total_seconds,
    };

    render(&report, args.output);

    0
}

// ============================================================
// Mode B / C stubs (M4.9.d / .e)
// ============================================================

fn run_mode_b(args: RunArgs) -> i32 {
    let _ = args;
    eprintln!("error: `atenia run --mode b` lands in M4.9.e. Until then, use:");
    eprintln!(
        "  cargo test --release --test m4_7_6_e_llama2_13b_modes_b_c_test \\"
    );
    eprintln!("    -- --ignored --nocapture --test-threads=1 \\");
    eprintln!("       llama2_13b_mode_b_autonomous_trigger_fires_under_high_pressure");
    2
}

// ============================================================
// Mode C — forced 50 % LRU spill ("momento guau" canonical)
// ============================================================

fn run_mode_c(args: RunArgs) -> i32 {
    let total_start = Instant::now();

    // Mode C uses seq=1 by default per M4.7.6.e to cap
    // wall-clock per mode. Operator can override via --seq;
    // the contract `argmax(pre) == argmax(post)` holds at any
    // seq > 0 because attention pos 0 is purely
    // self-attention.
    let effective_seq = if args.seq == 4 { 1 } else { args.seq };
    if args.seq == 4 {
        eprintln!(
            "note: --seq defaulted from 4 to 1 for Mode C (M4.7.6.e\n\
             canonical setting; --seq <other> overrides)."
        );
    }
    let runtime = LlamaRuntime { batch: 1, seq: effective_seq };

    // Token pattern: BOS for seq=1; ascending integers for
    // seq>1. Same convention as Mode A.
    let token_pattern: Vec<f32> = match effective_seq {
        1 => vec![1.0],
        n => {
            let mut v = vec![1.0];
            for i in 1..n {
                v.push((i as f32) * 100.0);
            }
            v
        }
    };

    // Resolve cache directory.
    let cache_dir = match args.cache_dir.as_ref() {
        Some(p) => p.clone(),
        None => cache_dir_for("atenia_run_mode_c"),
    };
    if let Err(e) = std::fs::create_dir_all(&cache_dir) {
        eprintln!(
            "error: could not create cache directory {}: {}",
            cache_dir.display(), e,
        );
        return 1;
    }

    eprintln!("=== Atenia v20 Killer Demo — Mode C (forced 50% LRU spill) ===");
    eprintln!();
    eprintln!("Cache dir:      {}", cache_dir.display());

    let _ = cache_dir_disk_probe(&cache_dir);

    eprintln!(
        "Loading {} (this typically takes ~3 min on NVMe) ...",
        args.model.display(),
    );

    let hb = if !args.no_progress {
        Some(Heartbeat::start(2_000))
    } else {
        None
    };
    let (mut graph, metrics) = build_and_load_llama(
        &args.model,
        runtime,
        /*verbose=*/ false,
    );
    if let Some(hb) = hb {
        hb.stop();
    }

    eprintln!(
        "Loaded {} parameters in {:.1}s.",
        metrics.param_count, metrics.load_secs,
    );

    // Attach a low-pressure reactive context so the LRU
    // populates during warmup but no autonomous trigger
    // fires. Mode C's spill must be *forced* by the explicit
    // `deep_degrade_with_lru` call below.
    let ctx = make_context(cache_dir.clone(), /*high_pressure=*/ false);
    graph.set_reactive_context(ctx);

    let tokens = Tensor::new_cpu(vec![1, effective_seq], token_pattern);

    // ---- Warmup forward ----
    eprintln!("[1/3] Warmup forward (no spill yet) ...");
    let hb = if !args.no_progress {
        Some(Heartbeat::start(2_000))
    } else {
        None
    };
    let warmup_start = Instant::now();
    let warmup_outputs = graph.execute(vec![tokens.clone()]);
    let warmup_secs = warmup_start.elapsed().as_secs_f32();
    if let Some(hb) = hb {
        hb.stop();
    }
    let warmup_logits = warmup_outputs[0].as_cpu_slice().to_vec();
    let vocab = metrics.config.vocab_size;
    assert_eq!(
        warmup_logits.len(),
        effective_seq * vocab,
        "warmup logits length mismatch"
    );
    let (pre_id, pre_logit) = argmax_row(&warmup_logits[..vocab], vocab);
    eprintln!(
        "Warmup forward: {:.1}s   argmax(pos 0) = {} logit {:.4}",
        warmup_secs, pre_id, pre_logit,
    );

    // ---- Force the M4.7.5.d 50 % LRU spill ----
    eprintln!("[2/3] Forcing deep_degrade_with_lru (SPILL_FRACTION = 0.5) ...");
    let spill_start = Instant::now();
    let migration_result = graph.deep_degrade_with_lru(&cache_dir);
    let spill_secs = spill_start.elapsed().as_secs_f32();
    let migration = match migration_result {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: deep_degrade_with_lru failed: {:?}", e);
            return 1;
        }
    };
    eprintln!(
        "Forced spill: {} migrated, {} skipped, {:.1}s",
        migration.tensors_migrated, migration.tensors_skipped, spill_secs,
    );

    // ---- Post-spill forward ----
    eprintln!("[3/3] Post-spill forward (lazy restore through ensure_cpu) ...");
    let hb = if !args.no_progress {
        Some(Heartbeat::start(2_000))
    } else {
        None
    };
    let post_start = Instant::now();
    let mut post_outputs = graph.execute(vec![tokens]);
    let post_secs = post_start.elapsed().as_secs_f32();
    if let Some(hb) = hb {
        hb.stop();
    }
    // Output may have been migrated to disk during the
    // forward by a guard re-trigger; lazy-restore before
    // reading. M4.7.6.e Mode B's `ensure_cpu` defensive call
    // generalised to Mode C as well.
    if let Err(e) = post_outputs[0].ensure_cpu() {
        eprintln!("error: ensure_cpu on post-spill output failed: {:?}", e);
        return 1;
    }
    let post_logits = post_outputs[0].as_cpu_slice().to_vec();
    assert_eq!(
        post_logits.len(),
        effective_seq * vocab,
        "post-spill logits length mismatch",
    );
    let (post_id, post_logit) = argmax_row(&post_logits[..vocab], vocab);
    eprintln!(
        "Post-spill forward: {:.1}s   argmax(pos 0) = {} logit {:.4}",
        post_secs, post_id, post_logit,
    );

    // ---- Transparency contract ----
    let bit_exact = pre_id == post_id && pre_logit == post_logit;

    // Compose per-position argmax for the JSON / text report
    // (post-spill row is the "live" one for downstream
    // consumers; the contract block carries the pre/post
    // comparison explicitly).
    let mut argmax_entries = Vec::with_capacity(effective_seq);
    for pos in 0..effective_seq {
        let row = &post_logits[pos * vocab..(pos + 1) * vocab];
        let (id, logit) = argmax_row(row, vocab);
        argmax_entries.push(ArgmaxEntry { position: pos, token_id: id, logit });
    }

    let max_abs = post_logits.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let mean_abs: f32 =
        post_logits.iter().map(|v| v.abs()).sum::<f32>() / post_logits.len() as f32;
    let finite = post_logits.iter().filter(|v| v.is_finite()).count();

    let total_seconds = total_start.elapsed().as_secs_f64();

    let report = DemoReport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        mode: "c",
        seq: effective_seq,
        model: ModelReport {
            path: args.model.clone(),
            layers: metrics.config.num_hidden_layers,
            hidden_size: metrics.config.hidden_size,
            intermediate_size: metrics.config.intermediate_size,
            vocab_size: metrics.config.vocab_size,
            param_count: metrics.param_count,
            storage: "bf16",
        },
        phases: PhasesReport {
            build_seconds: metrics.build_secs,
            load_seconds: metrics.load_secs,
            load_throughput_mb_s_estimate: 26_000.0
                / metrics.load_secs.max(0.01),
            forward_seconds: post_secs,
            spill_seconds: Some(spill_secs),
            spill_tensors_migrated: Some(migration.tensors_migrated),
            warmup_forward_seconds: Some(warmup_secs),
        },
        argmax: argmax_entries,
        logit_stats: LogitStats {
            max_abs,
            mean_abs,
            finite,
            total: post_logits.len(),
        },
        contract: Some(TransparencyContract {
            name: "transparency",
            pre: ArgmaxEntry {
                position: 0,
                token_id: pre_id,
                logit: pre_logit,
            },
            post: ArgmaxEntry {
                position: 0,
                token_id: post_id,
                logit: post_logit,
            },
            bit_exact,
            description: if bit_exact {
                "argmax(pre-spill) == argmax(post-spill) bit-exactly — \
                 the LRU spill + lazy-restore cycle is mathematically \
                 transparent at this parameter scale."
            } else {
                "argmax(pre-spill) != argmax(post-spill) — TRANSPARENCY \
                 VIOLATION. The spill + restore cycle changed the model's \
                 output. This indicates a regression in the M4.7.4.d / \
                 M4.7.5 spill primitives."
            },
        }),
        total_seconds,
    };

    render(&report, args.output);

    // Best-effort cleanup of the cache directory if we
    // created it ourselves (i.e. user did not pass --cache-dir).
    if args.cache_dir.is_none() {
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    if bit_exact {
        0
    } else {
        // Exit code 3: mathematical contract violation.
        // Distinguished from runtime errors (1) and config
        // errors (2) so CI scripts can detect the difference.
        3
    }
}
