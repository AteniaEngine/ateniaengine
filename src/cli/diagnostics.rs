//! **CLI diagnostics (CLI-3): `doctor`, `diagnose`, `capabilities`.**
//!
//! Three read-only diagnostic commands. They inspect the host, a
//! model directory, or the engine's static capability set, and
//! print a human checklist (or JSON with `--json`). They never run
//! generation, never mutate anything, and touch no runtime core,
//! graph builder or loader — only passive reads of helpers that
//! already exist (`cpu_features`, `cuda::cuda_available`, the
//! Adapter Toolkit v2 inspect/resolve path).
//!
//! Output contract: the report goes to **stdout**; logs and errors
//! go to **stderr** via CLI-2 logging and [`CliError`].

use std::path::Path;

use serde::Serialize;

use crate::adapter_toolkit::{inspect_model_dir, validate, ResolvedAdapterSpec};

use super::error::CliError;
use super::logging;

/// GGUF tensor quantisations the engine's `decode_tensor` can
/// currently decode. Kept in sync with `v17::loader::gguf_decode`.
const SUPPORTED_QUANTS: &[&str] = &["F32", "F16", "Q8_0", "Q5_0", "Q4_K", "Q5_K", "Q6_K"];

/// Model families the engine supports (v1 adapters + Llama-compatible
/// families that ride the Llama adapter).
const SUPPORTED_FAMILIES: &[&str] = &[
    "llama", "qwen2", "qwen3", "gemma2", "gemma3", "phi3", "mistral", "smollm (llama-compatible)",
    "falcon3 (llama-compatible)",
];

/// Architectures explicitly out of scope — listed so the report is
/// honest about what will fail.
const UNSUPPORTED_ARCHITECTURES: &[&str] = &[
    "falcon (classic FalconForCausalLM / RWForCausalLM)",
    "mixture-of-experts (Mixtral, Mistral-MoE)",
    "multimodal / vision",
    "encoder-decoder",
];

// ================================================================
// Shared check model
// ================================================================

/// The severity of a single diagnostic check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Status {
    /// The check passed.
    Ok,
    /// The check passed but something is sub-optimal.
    Warn,
    /// The check failed — the thing being diagnosed is not usable.
    Fail,
    /// Neutral information, not a pass/fail.
    Info,
}

impl Status {
    fn tag(self) -> &'static str {
        match self {
            Status::Ok => "[ ok ]",
            Status::Warn => "[warn]",
            Status::Fail => "[fail]",
            Status::Info => "[info]",
        }
    }
}

/// One line of a diagnostic report.
#[derive(Clone, Debug, Serialize)]
struct Check {
    status: Status,
    label: String,
    detail: String,
}

fn check(status: Status, label: &str, detail: impl Into<String>) -> Check {
    Check {
        status,
        label: label.to_string(),
        detail: detail.into(),
    }
}

/// Render a list of checks as an aligned human checklist.
fn render_checks(title: &str, checks: &[Check]) -> String {
    let mut out = format!("{title}\n");
    let width = checks.iter().map(|c| c.label.len()).max().unwrap_or(0);
    for c in checks {
        out.push_str(&format!(
            "  {} {:width$}  {}\n",
            c.status.tag(),
            c.label,
            c.detail,
            width = width
        ));
    }
    out
}

/// `true` when any check failed — used to pick the exit code.
fn any_failed(checks: &[Check]) -> bool {
    checks.iter().any(|c| c.status == Status::Fail)
}

// ================================================================
// `atenia doctor`
// ================================================================

#[derive(Serialize)]
struct DoctorReport {
    checks: Vec<Check>,
    warnings: Vec<String>,
}

/// `atenia doctor` — global host + build diagnostics.
pub fn run_doctor(json: bool) -> i32 {
    logging::info("command start: doctor");
    let checks = collect_doctor_checks();
    let warnings: Vec<String> = checks
        .iter()
        .filter(|c| c.status == Status::Warn)
        .map(|c| format!("{}: {}", c.label, c.detail))
        .collect();

    if json {
        let report = DoctorReport {
            checks: checks.clone(),
            warnings,
        };
        match serde_json::to_string_pretty(&report) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                let err = CliError::generation_failed(
                    "failed to serialise the doctor report",
                    e.to_string(),
                );
                eprintln!("{err}");
                return err.exit.code();
            }
        }
    } else {
        print!("{}", render_checks("Atenia Engine — system diagnostics", &checks));
    }
    logging::info("command completed: doctor");
    // `doctor` reports the host; it does not fail just because the
    // host has a warning. Only a hard check failure exits non-zero.
    if any_failed(&checks) {
        2
    } else {
        0
    }
}

fn collect_doctor_checks() -> Vec<Check> {
    let mut checks = Vec::new();

    checks.push(check(Status::Info, "version", env!("CARGO_PKG_VERSION")));
    checks.push(check(
        Status::Info,
        "os",
        format!("{} / {}", std::env::consts::OS, std::env::consts::ARCH),
    ));

    // CPU — from the existing cpu_features helper.
    let cpu = crate::cpu_features::cpu_features();
    let mut isa = Vec::new();
    if cpu.avx2 {
        isa.push("AVX2");
    }
    if cpu.avx512f {
        isa.push("AVX512F");
    }
    if cpu.fma {
        isa.push("FMA");
    }
    let isa_str = if isa.is_empty() {
        "no SIMD".to_string()
    } else {
        isa.join(", ")
    };
    checks.push(check(
        if cpu.avx2 { Status::Ok } else { Status::Warn },
        "cpu",
        format!("{} threads, {isa_str}", cpu.threads),
    ));

    // RAM — passive read via sysinfo (already a dependency).
    {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let total = sys.total_memory() as f64 / 1024.0_f64.powi(3);
        let avail = sys.available_memory() as f64 / 1024.0_f64.powi(3);
        // 16 GiB is the practical floor for the small/mid models;
        // below that, only the tiny models are comfortable.
        let status = if total >= 16.0 {
            Status::Ok
        } else {
            Status::Warn
        };
        checks.push(check(
            status,
            "ram",
            format!("{avail:.1} GiB available / {total:.1} GiB total"),
        ));
    }

    // Build flavour — CPU-only vs CUDA-enabled.
    let cuda_build = cfg!(atenia_cuda);
    checks.push(check(
        Status::Info,
        "build",
        if cuda_build {
            "CUDA-enabled"
        } else {
            "CPU-only (vendor-agnostic)"
        },
    ));

    // CUDA runtime availability.
    let cuda = crate::cuda::cuda_available();
    checks.push(check(
        if cuda { Status::Ok } else { Status::Info },
        "cuda",
        if cuda {
            "available"
        } else if cuda_build {
            "not available (no NVIDIA driver / GPU detected)"
        } else {
            "not available (CPU-only build)"
        },
    ));

    checks.push(check(
        Status::Ok,
        "backends",
        if cuda { "cpu, cuda" } else { "cpu" },
    ));
    checks.push(check(Status::Ok, "formats", "safetensors, GGUF"));
    checks.push(check(Status::Ok, "gguf quants", SUPPORTED_QUANTS.join(", ")));

    // Cache/log write permission — try a temp-dir round trip.
    {
        let probe = std::env::temp_dir().join(format!(
            "atenia_doctor_write_probe_{}",
            std::process::id()
        ));
        let writable = std::fs::write(&probe, b"ok").is_ok();
        let _ = std::fs::remove_file(&probe);
        checks.push(check(
            if writable { Status::Ok } else { Status::Warn },
            "temp dir writable",
            std::env::temp_dir().display().to_string(),
        ));
    }

    checks
}

// ================================================================
// `atenia diagnose --model <dir>`
// ================================================================

#[derive(Serialize)]
struct DiagnoseReport {
    model_dir: String,
    checks: Vec<Check>,
    ready: bool,
}

/// `atenia diagnose --model <dir>` — pre-flight diagnosis of a
/// model directory. Never generates; it resolves the adapter as a
/// dry run only.
pub fn run_diagnose(model_dir: &Path, json: bool) -> i32 {
    logging::info("command start: diagnose");
    logging::debug(&format!("model directory: {}", model_dir.display()));

    // A non-existent path cannot be diagnosed at all — hard error.
    if !model_dir.exists() {
        let err = CliError::io_not_found("the model directory", model_dir);
        eprintln!("{err}");
        return err.exit.code();
    }

    let checks = collect_diagnose_checks(model_dir);
    let ready = !any_failed(&checks);

    if json {
        let report = DiagnoseReport {
            model_dir: model_dir.display().to_string(),
            checks: checks.clone(),
            ready,
        };
        match serde_json::to_string_pretty(&report) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                let err = CliError::generation_failed(
                    "failed to serialise the diagnose report",
                    e.to_string(),
                );
                eprintln!("{err}");
                return err.exit.code();
            }
        }
    } else {
        print!(
            "{}",
            render_checks(
                &format!("Atenia Engine — model diagnosis: {}", model_dir.display()),
                &checks
            )
        );
        if ready {
            println!("\nResult: model is ready to generate.");
        } else {
            println!("\nResult: model is NOT ready — see the [fail] lines above.");
        }
    }

    logging::info("command completed: diagnose");
    if ready {
        0
    } else {
        2
    }
}

fn collect_diagnose_checks(model_dir: &Path) -> Vec<Check> {
    let mut checks = Vec::new();
    checks.push(check(Status::Ok, "model path", "exists"));

    // Auto-detect the model via the Adapter Toolkit v2 inspector.
    // A failure here means the directory is not a supported model.
    let report = match inspect_model_dir(model_dir) {
        Ok(r) => r,
        Err(e) => {
            checks.push(check(Status::Fail, "model detection", e.to_string()));
            return checks;
        }
    };
    let dsl = report.dsl;

    let format = dsl.format.as_deref().unwrap_or("unknown");
    checks.push(check(Status::Ok, "format", format));
    checks.push(check(Status::Ok, "family", dsl.family.clone()));

    // Tokenizer presence — GGUF embeds it; HF needs tokenizer.json.
    if format == "gguf" {
        checks.push(check(Status::Info, "tokenizer", "embedded in the GGUF file"));
    } else if model_dir.join("tokenizer.json").is_file() {
        checks.push(check(Status::Ok, "tokenizer", "tokenizer.json present"));
    } else {
        checks.push(check(
            Status::Fail,
            "tokenizer",
            "tokenizer.json is missing from the model directory",
        ));
    }

    // Validate the inferred adapter spec (Adapter Toolkit v2).
    let report_v = validate(&dsl);
    if report_v.is_ok() {
        checks.push(check(Status::Ok, "adapter spec", "valid"));
    } else {
        checks.push(check(
            Status::Fail,
            "adapter spec",
            report_v.errors.join("; "),
        ));
    }
    for w in &report_v.warnings {
        checks.push(check(Status::Warn, "adapter spec", w.clone()));
    }

    // Dry-run adapter resolution — resolve to a v1 family adapter
    // WITHOUT loading weights or generating.
    match ResolvedAdapterSpec::resolve(&dsl) {
        Ok(spec) => {
            checks.push(check(
                Status::Ok,
                "adapter resolved",
                format!("{:?} ({})", spec.family, spec.architecture),
            ));
        }
        Err(e) => {
            checks.push(check(Status::Fail, "adapter resolved", e.to_string()));
        }
    }

    // Surface the inspector's own notes (e.g. GGUF RoPE caveat).
    for note in &report.notes {
        checks.push(check(Status::Info, "note", note.clone()));
    }

    checks
}

// ================================================================
// `atenia capabilities`
// ================================================================

#[derive(Serialize)]
struct CapabilitiesReport {
    supported_families: Vec<String>,
    unsupported_architectures: Vec<String>,
    /// **MODEL-INTAKE-1** — non-native architecture strings the
    /// curated allowlist maps to a compatible base family (UNCERTIFIED).
    compatible_architectures: Vec<String>,
    /// **MODEL-INTAKE-1** — env opt-in for the generic decoder path.
    generic_intake_opt_in: String,
    formats: Vec<String>,
    gguf_quants: Vec<String>,
    features: Vec<String>,
}

/// `architecture -> base (evidence)` lines for the curated
/// known-compatible allowlist. Single source of truth: the compat
/// layer's [`crate::model_adapters::compat::LLAMA_COMPATIBLE_ALLOWLIST`].
fn compatible_architecture_lines() -> Vec<String> {
    crate::model_adapters::compat::LLAMA_COMPATIBLE_ALLOWLIST
        .iter()
        .map(|e| format!("{} -> {} (uncertified)", e.architecture, e.base_architecture))
        .collect()
}

const FEATURES: &[&str] = &[
    "KV cache (incremental decoding)",
    "CPU fallback (vendor-agnostic, no CUDA required)",
    "VRAM / RAM / disk memory tiering",
    "Adapter Toolkit v2 declarative adapters",
    "GGUF and HuggingFace safetensors loading",
];

/// `atenia capabilities` — the engine's static capability set.
/// Honest: it lists what is out of scope as plainly as what works.
pub fn run_capabilities(json: bool) -> i32 {
    logging::info("command start: capabilities");

    if json {
        let report = CapabilitiesReport {
            supported_families: SUPPORTED_FAMILIES.iter().map(|s| s.to_string()).collect(),
            unsupported_architectures: UNSUPPORTED_ARCHITECTURES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            compatible_architectures: compatible_architecture_lines(),
            generic_intake_opt_in: crate::model_adapters::compat::GENERIC_INTAKE_ENV.to_string(),
            formats: vec!["GGUF".to_string(), "safetensors".to_string()],
            gguf_quants: SUPPORTED_QUANTS.iter().map(|s| s.to_string()).collect(),
            features: FEATURES.iter().map(|s| s.to_string()).collect(),
        };
        match serde_json::to_string_pretty(&report) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                let err = CliError::generation_failed(
                    "failed to serialise the capabilities report",
                    e.to_string(),
                );
                eprintln!("{err}");
                return err.exit.code();
            }
        }
    } else {
        let mut out = String::from("Atenia Engine — capabilities\n\n");
        out.push_str("Supported model families:\n");
        for f in SUPPORTED_FAMILIES {
            out.push_str(&format!("  - {f}\n"));
        }
        out.push_str("\nNot supported (out of scope):\n");
        for a in UNSUPPORTED_ARCHITECTURES {
            out.push_str(&format!("  - {a}\n"));
        }
        out.push_str("\nKnown-compatible (allowlist, mapped + uncertified):\n");
        for line in compatible_architecture_lines() {
            out.push_str(&format!("  - {line}\n"));
        }
        out.push_str(&format!(
            "  (unknown architectures: set {}=1 to attempt the generic Llama-compatible \
             decoder path, topology-checked + uncertified)\n",
            crate::model_adapters::compat::GENERIC_INTAKE_ENV
        ));
        out.push_str("\nWeight formats:\n  - GGUF\n  - safetensors\n");
        out.push_str("\nGGUF quantisations:\n");
        for q in SUPPORTED_QUANTS {
            out.push_str(&format!("  - {q}\n"));
        }
        out.push_str("\nFeatures:\n");
        for feat in FEATURES {
            out.push_str(&format!("  - {feat}\n"));
        }
        print!("{out}");
    }

    logging::info("command completed: capabilities");
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_tags_are_fixed_width() {
        assert_eq!(Status::Ok.tag().len(), 6);
        assert_eq!(Status::Warn.tag().len(), 6);
        assert_eq!(Status::Fail.tag().len(), 6);
        assert_eq!(Status::Info.tag().len(), 6);
    }

    #[test]
    fn doctor_checks_cover_the_core_items() {
        let checks = collect_doctor_checks();
        let labels: Vec<&str> = checks.iter().map(|c| c.label.as_str()).collect();
        for expected in ["version", "os", "cpu", "ram", "build", "cuda", "formats"] {
            assert!(labels.contains(&expected), "missing doctor check: {expected}");
        }
    }

    #[test]
    fn any_failed_detects_a_fail() {
        let ok = vec![check(Status::Ok, "a", "x"), check(Status::Warn, "b", "y")];
        assert!(!any_failed(&ok));
        let bad = vec![check(Status::Ok, "a", "x"), check(Status::Fail, "b", "y")];
        assert!(any_failed(&bad));
    }

    #[test]
    fn render_checks_aligns_and_tags() {
        let checks = vec![
            check(Status::Ok, "short", "x"),
            check(Status::Fail, "a-longer-label", "y"),
        ];
        let r = render_checks("Title", &checks);
        assert!(r.starts_with("Title\n"));
        assert!(r.contains("[ ok ]"));
        assert!(r.contains("[fail]"));
    }

    #[test]
    fn diagnose_checks_fail_for_unsupported_model() {
        // A synthetic classic-Falcon directory must produce a
        // failing detection check, not a panic.
        let dir = std::env::temp_dir().join(format!(
            "atenia_diag_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["FalconForCausalLM"],"model_type":"falcon"}"#,
        )
        .unwrap();
        let checks = collect_diagnose_checks(&dir);
        assert!(any_failed(&checks), "classic Falcon must fail diagnosis");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
